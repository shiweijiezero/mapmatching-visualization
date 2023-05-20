import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from neural import *
import logging
from utils import *
import dgl

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


#
# # Compute log sum exp in a numerically stable way for the forward algorithm
# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
#
#


def log_sum_exp(smat):
    """
    参数: smat 是 "status matrix", DP状态矩阵; 其中 smat[i][j] 表示 上一帧为i状态且当前帧为j状态的分值
    作用: 针对输入的【二维数组的每一列】, 各元素分别取exp之后求和再取log; 物理意义: 当前帧到达每个状态的分值(综合所有来源)
    例如: smat = [[ 1  3  9]
                 [ 2  9  1]
                 [ 3  4  7]]
         其中 smat[:, 2]= [9,1,7] 表示当前帧到达状态"2"有三种可能的来源, 分别来自上一帧的状态0,1,2
         这三条路径的分值求和按照log_sum_exp法则，展开 log_sum_exp(9,1,7) = log(exp(9) + exp(1) + exp(7)) = 3.964
         所以，综合考虑所有可能的来源路径，【当前帧到达状态"2"的总分值】为 3.964
         前两列类似处理，得到一个行向量作为结果返回 [ [?, ?, 3.964] ]
    注意数值稳定性技巧 e.g. 假设某一列中有个很大的数
    输入的一列 = [1, 999, 4]
    输出     = log(exp(1) + exp(999) + exp(4)) # 【直接计算会遭遇 exp(999) = INF 上溢问题】
            = log(exp(1-999)*exp(999) + exp(999-999)*exp(999) + exp(4-999)*exp(999)) # 每个元素先乘后除 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)] * exp(999)) # 提取公因式 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + log(exp(999)) # log乘法拆解成加法
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + 999 # 此处exp(?)内部都是非正数，不会发生上溢
            = log([exp(smat[0]-vmax) + exp(smat[1]-vmax) + exp(smat[2]-vmax)]) + vmax # 符号化表达
    代码只有两行，但是涉及二维张量的变形有点晦涩，逐步分析如下, 例如:
    smat = [[ 1  3  9]
            [ 2  9  1]
            [ 3  4  7]]
    smat.max(dim=0, keepdim=True) 是指【找到各列的max】，即: vmax = [[ 3  9  9]] 是个行向量
    然后 smat-vmax, 两者形状分别是 (3,3) 和 (1,3), 相减会广播(vmax广播复制为3*3矩阵)，得到:
    smat - vmax = [[ -2  -6  0 ]
                   [ -1  0   -8]
                   [ 0   -5  -2]]
    然后.exp()是逐元素求指数
    然后.sum(axis=0, keepdim=True) 是"sum over axis 0"，即【逐列求和】, 得到的是行向量，shape=(1,3)
    然后.log()是逐元素求对数
    最后再加上 vmax; 两个行向量相加, 结果还是个行向量
    """
    vmax = smat.max(dim=0, keepdim=True).values  # 每一列的最大数
    return (smat - vmax).exp().sum(axis=0, keepdim=True).log() + vmax


class BiLSTM_CRF(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, graph, n_layers=2):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = graph.number_of_nodes()
        self.tagset_size = graph.number_of_nodes()
        self.g = dgl.DGLGraph()
        self.g.from_networkx(nx_graph=graph)
        self.dropout = nn.Dropout(p=0.1)
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim))

        # Maps the output of the LSTM into tag space.
        # self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2tag = nn.Linear(hidden_dim, hidden_dim)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.

        self.transition_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Linear(2 * hidden_dim, embedding_dim)
        )

        self.emit_prob = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.emit_liner = nn.Linear(self.embedding_dim, 1)
        self.trans_prob = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.trans_liner = nn.Linear(self.embedding_dim, 1)

        # self.transitions = nn.Parameter(
        #     torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

        # self.merge_liner = nn.Linear(self.embedding_dim,k_neighbour)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def E(self, label_vec, node_vec):
        vec = self.emit_prob(torch.stack([label_vec, node_vec]).view(1, 2, self.embedding_dim)).view(1,
                                                                                                     self.embedding_dim)
        return torch.sigmoid(self.emit_liner(vec).view(1))

    def T(self, node_vec1, node_vec2):
        vec = self.trans_prob(torch.stack([node_vec1, node_vec2]).view(1, 2, self.embedding_dim)).view(1,
                                                                                                       self.embedding_dim)
        return torch.sigmoid(self.trans_liner(vec).view(1))

    def get_candidatae_feat_by_id(self, candidate_id):
        return self.word_embeds(torch.tensor(candidate_id).long())

    def _forward_alg(self, observed_feats, sentence_candidates):

        layer_result = []  # 用于存放上一层的概率
        for i, feat in enumerate(observed_feats):  # 从左到右遍历每一层
            if (i == 0):
                for candidate_id in sentence_candidates[i]:
                    layer_result.append(self.E(feat, self.get_candidatae_feat_by_id(candidate_id)))  # 第一层只有发射概率
            else:
                new_layer_result = []
                for candidate_id in sentence_candidates[i]:  # 遍历当前层中的每个候选人
                    temp_result = []
                    emit_p = self.E(feat, self.get_candidatae_feat_by_id(candidate_id))
                    for j, candidate_id_prev in enumerate(sentence_candidates[i - 1]):  # 遍历上一层中的每个候选人
                        # 上一层某个候选人a的概率 * (a到b的转移概率)
                        temp_result.append(layer_result[j] \
                                           + self.T(self.get_candidatae_feat_by_id(candidate_id_prev),
                                                    self.get_candidatae_feat_by_id(candidate_id)) + emit_p)
                    new_layer_result.append(log_sum_exp(torch.stack(temp_result)).view(1))
                layer_result = new_layer_result  # 保存概率

        # logsumexp_
        # {上层每个候选人}（上层每个候选人的累加概率对数 + 该候选人到当前层候选人的迁移概率对数）+当前层候选人的发射概率对数
        #

        return log_sum_exp(torch.stack(layer_result)).view(1)

    def get_merged_feat(self, neighbours):
        embed = self.word_embeds(torch.Tensor(neighbours).long())
        embed = torch.mean(embed, 0)
        return embed

    def _get_embeded_features(self, sentence):

        h = self.word_embeds.weight
        # print("节点数量:",g.number_of_nodes())
        for i, layer in enumerate(self.gcn_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

        # self.hidden = self.init_hidden()
        embeds = []
        for neighbours in sentence:
            embeds.append(self.get_merged_feat(neighbours))
        return torch.stack(embeds)
        # embeds = torch.stack(embeds, 0)  # seq_len * embed_dim
        # embeds = embeds.view(len(sentence), 1, -1)  # seq_len *1* embed_dim
        # # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # lstm_feats = self.hidden2tag(lstm_out)
        # return lstm_feats  # seq_len * embed_dim

    def _score_sentence(self, observed_feats, hidden_states):
        """
                求路径pair: frames->tags 的分值
                index:      0   1   2   3   4   5   6
                frames:     F0  F1  F2  F3  F4
                tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
                """

        score = torch.zeros(1)

        for i, frame in enumerate(observed_feats):  # 沿途累加每一帧的转移和发射
            # print(labels[i])
            score += self.E(self.get_merged_feat(hidden_states[i]), frame)
            if (i > 0):
                score += self.T(self.get_merged_feat(hidden_states[i]), self.get_merged_feat(hidden_states[i - 1]))
        return score

    def _viterbi_decode(self, observed_feats, observed_neighbours):

        result_path = []
        final_path = []

        layer_result = []

        for i, feat in enumerate(observed_feats):
            if (i == 0):
                for candidate_id in observed_neighbours[i]:
                    layer_result.append(self.E(feat, self.get_candidatae_feat_by_id(candidate_id)))
            else:
                result_path.append([])
                new_layer_result = []
                for candidate_id in observed_neighbours[i]:
                    temp_result = []
                    for j, candidate_id_prev in enumerate(observed_neighbours[i - 1]):
                        temp_result.append(layer_result[j] \
                                           + self.T(self.get_candidatae_feat_by_id(candidate_id_prev),
                                                    self.get_candidatae_feat_by_id(candidate_id)))
                    maxpos = torch.argmax(torch.tensor(temp_result))
                    result_path[-1].append(maxpos)
                    current_point_score = temp_result[maxpos]

                    current_point_score += self.E(feat, self.get_candidatae_feat_by_id(candidate_id))

                    new_layer_result.append(current_point_score)

                layer_result = new_layer_result

        layer_id = len(observed_neighbours) - 2
        current_ptr = torch.argmax(torch.tensor(layer_result))
        final_path.append(observed_neighbours[layer_id + 1][current_ptr])
        while (layer_id >= 0):
            final_path.append(observed_neighbours[layer_id][current_ptr])
            current_ptr = result_path[layer_id][current_ptr]
            layer_id -= 1

        return layer_result[torch.argmax(torch.tensor(layer_result))], final_path

        #     for next_tag in range(self.tagset_size):
        #         # next_tag_var[i] holds the viterbi variable for tag i at the
        #         # previous step, plus the score of transitioning
        #         # from tag i to next_tag.
        #         # We don't include the emission scores here because the max
        #         # does not depend on them (we add them in below)
        #         next_tag_var = forward_var + self.T()
        #
        #         best_tag_id = argmax(next_tag_var)
        #         bptrs_t.append(best_tag_id)
        #         viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
        #     # Now add in the emission scores, and assign forward_var to the set
        #     # of viterbi variables we just computed
        #     forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
        #     backpointers.append(bptrs_t)
        #
        # # Transition to STOP_TAG
        # terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # best_tag_id = argmax(terminal_var)
        # path_score = terminal_var[0][best_tag_id]
        #
        # # Follow the back pointers to decode the best path.
        # best_path = [best_tag_id]
        # for bptrs_t in reversed(backpointers):
        #     best_tag_id = bptrs_t[best_tag_id]
        #     best_path.append(best_tag_id)
        # # Pop off the start tag (we dont want to return that to the caller)
        # start = best_path.pop()
        # assert start == self.tag_to_ix[START_TAG]  # Sanity check
        # best_path.reverse()
        # return path_score, best_path

    def neg_log_likelihood(self, observed, hiddenstates):
        observed_feats = self._get_embeded_features(observed)
        forward_score = self._forward_alg(observed_feats, observed)
        # print(forward_score)
        gold_score = self._score_sentence(observed_feats, hiddenstates)
        # print(gold_score)
        return forward_score - gold_score

    def forward(self, observed_neighbours):  # dont confuse this with _forward_alg above.
        print(f"Do infer:{str(observed_neighbours)[:20]}")
        # Get the emission scores from the BiLSTM
        observed_feats = self._get_embeded_features(observed_neighbours)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(observed_feats, observed_neighbours)
        return score, tag_seq


# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for x, y in training_data:
    for word in x:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

opt = DefaultConfig()
train_dataloader, valid_dataloader, GRAPH = prepareDataloader(opt)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}


def RMF(path1, path2_):
    path2 = []
    for item in path2_:
        path2 += item

    p1_segs = set()
    p2_segs = set()
    for p1, p2 in zip(path1[0:-1], path[1:]):
        p1_segs.add((p1, p2))
    for p1, p2 in zip(path2[0:-1], path2[1:]):
        p2_segs.add((p1, p2))

    rmf = len(p1_segs.intersection(p2_segs)) / len(p1_segs.union(p2_segs))
    return rmf

EMBEDDING_DIM = 10
HIDDEN_DIM = 10

model = BiLSTM_CRF(EMBEDDING_DIM, HIDDEN_DIM, graph=GRAPH)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

iter = 0
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10):  # again, normally you would NOT do 300 epochs, it is toy data
    for x, y in train_dataloader.dataset:
        jump = False
        for one in x + y:
            for two in one:
                if (two > 67200):
                    jump = True
        if (jump):
            continue

        if (iter % 5 == 0):
            with torch.no_grad():
                precheck_sent = train_dataloader.dataset.data[0][0]
                print(str(precheck_sent)[:20] + '...')
                score, path = model(precheck_sent)
                true_path = train_dataloader.dataset.data[0][1]
                rmf = RMF(path, true_path)
                print(f"RMF:{rmf}, Score:{score}\nPATH:{str(path)[:100]}")

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(x, y)
        print(f"Iter:{iter} loss:{loss}")

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        iter += 1
