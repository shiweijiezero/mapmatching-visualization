from torch import nn
import dgl
from dgl.nn.pytorch import GraphConv
import torch
from torch.nn import functional as F
import torchsnooper
import numpy


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Sequential(
            nn.Linear(output_size, hidden_size * 2),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Linear(inchannel, outchannel, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Linear(outchannel, outchannel, bias=False),
            nn.BatchNorm1d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_size, n_layers=2, dropout=0.01):

        self.hidden_size = hidden_size

        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=F.relu))
        self.layers.append(GraphConv(hidden_size, hidden_size))
        self.dropout = nn.Dropout(p=dropout)

        # self.conv_seq = nn.Sequential(
        #     nn.Conv2d(1, 2, 20),
        #     nn.MaxPool2d((6, 6)),
        #     nn.Conv2d(2, 3, 3),
        #     nn.MaxPool2d((2, 2)),
        #     nn.ReLU(),
        #
        #     # nn.Linear(in_feats, hidden_size * 2),
        #     # nn.BatchNorm1d(hidden_size * 2),
        #     # nn.Linear(hidden_size * 2, hidden_size),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_size, out_size),
        # )

        self.conv_seq = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3)
        )

        # self.seq = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 4),
        #     # self._make_layer(hidden_size * 10, hidden_size * 4, 3),
        #     self._make_layer(hidden_size * 4, hidden_size * 4, 3),
        #     nn.Linear(hidden_size * 4, out_size),
        #     nn.Sigmoid()
        #     # nn.Linear(in_feats, hidden_size * 2),
        #     # nn.BatchNorm1d(hidden_size * 2),
        #     # nn.Linear(hidden_size * 2, hidden_size),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_size, out_size),
        # )

        self.seq = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            # nn.Linear(hidden_size * 10, out_size),
            # self._make_layer(hidden_size * 10, hidden_size * 4, 3),
            # self._make_layer(hidden_size * 4, hidden_size * 4, 3),
            # nn.Linear(hidden_size * 3000, out_size),
            # nn.Sigmoid()
            # nn.Linear(in_feats, hidden_size * 2),
            # nn.BatchNorm1d(hidden_size * 2),
            # nn.Linear(hidden_size * 2, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, out_size),
        )

        # self.encoder = EncoderRNN(input_size=hidden_size, hidden_size=self.hidden_size)
        # self.decoder = DecoderRNN(hidden_size=hidden_size, output_size=hidden_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _make_layer(self, inchannel, outchannel, block_num):
        '''
        构建layer，包含多个residual
        '''
        shortcut = nn.Sequential(
            nn.Linear(inchannel, outchannel, bias=False),
            nn.BatchNorm1d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    # @torchsnooper.snoop()
    def forward(self, X, g, feat):
        # do the prediction using X which contains 4 id: user,region,time,poi
        # device = torch.device('cuda:%d' % torch.cuda.current_device())
        # h = self.feat
        # self.g.to(device)
        h = feat
        # print("节点数量:",g.number_of_nodes())
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, g)

        final_rst = []
        # for x_line in X:
        #     # x_line: 长为 N 的vector
        #     x_new_line = torch.stack((x_line,), 0)
        #     x_new_line = torch.t(x_new_line).expand_as(h)
        #     x_new_line = h.mul(x_new_line)
        #     # x_new_line = torch.flatten(x_new_line)
        #     x_new_line = torch.stack((x_new_line,), 0)
        #     mask_result.append(x_new_line)
        #
        # input_tensor = torch.stack(mask_result, 0)  # BatchSize*1*(N*HiddenSize) 10*1*(301754*128)
        #
        # conv_result = self.conv_seq(input_tensor)
        # print(conv_result.shape)
        # conv_result = conv_result.view(-1, self.num_flat_features(X)) #BatchSize*(N*HiddenSize)
        # print(conv_result.shape)
        # return self.seq(conv_result)


        for x_line in X:
            # x_line: 长为 N 的vector
            x_new_line = h[x_line != 0]
            x_new_line = x_new_line.T
            x_new_line = torch.stack((x_new_line,), 0)
            x_new_line = self.conv_seq(x_new_line)

            # x_new_line 是变长序列
            x_new_line = x_new_line[0].T  # sql_len * hidden_size


            each_predict = self.seq(x_new_line) #seq*output
            one_predict = torch.max(each_predict,0)[0] # 不定长序列到定长分类结果
            final_rst.append(F.sigmoid(one_predict))

        return torch.stack(final_rst,0)
            # encoder_outputs = torch.zeros(len(x_new_line), self.encoder.hidden_size)
            # decoder_outputs = []
            # decoder_hidden = encoder_hidden = self.encoder.initHidden()
            #
            # for ei in range(len(x_new_line)):
            #     encoder_output, encoder_hidden = self.encoder.forward(x_new_line[ei], encoder_hidden)
            #     encoder_outputs[ei] = encoder_output[0, 0]
            #
            # for di, point_vec in enumerate(encoder_outputs):
            #     decoder_output, decoder_hidden = self.decoder.forward(point_vec, decoder_hidden)
            #     decoder_outputs.append(decoder_output)
            #
            # final_rst.append(torch.cat(decoder_outputs[-10:], 1))

        # return self.seq(torch.cat(final_rst, 0))

        # input_tensor = torch.stack(mask_result, 0)  # BatchSize*1*(N*HiddenSize) 10*1*(301754*128)

        # conv_result = self.conv_seq(input_tensor)
        # print(conv_result.shape)
        # conv_result = conv_result.view(-1, self.num_flat_features(X))  # BatchSize*(N*HiddenSize)
        # print(conv_result.shape)
        # return self.seq(conv_result)
    # def save_node_embedding(self, feat, g):
    #     h = feat
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #         h = layer(h, g)
    #     return h
