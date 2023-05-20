from datetime import datetime
import warnings


class DefaultConfig(object):
    nrows = None
    java_src_path = '/home/gmjin/Projects/map-service'
    edge_path = 'data/20190930/hangzhou_road_network_20190906/EdgeInfo.csv'
    node_path = 'data/20190930/hangzhou_road_network_20190906/NodeInfo.csv'
    gps_path = 'data/20190930/hz_taxi_gps_trace_0629/all.txt'
    mee_path = 'data/20190930/hz_taxi_mee_trace_0629/000000_0'
    encoder_path = 'output/encoder_epoch-3299_loss-49.30.pkl'
    decoder_path = 'output/decoder_epoch-3299_loss-49.30.pkl'
    nxgraph_path = 'data/nxgraph.obj'

    gcn_train_file = 'data/train-k20.txt'

    output_dir = 'output'
    web_server_port = 5000

    # experimental setting
    java_args = '-mmOF-HMM -mc20 -sa3 -ms10 -tw110|-mmOF-CRF -mc65 -sa1 -ms200 -ce1 -dc40'
    interval = 65
    speed = 130
    angle = 100
    roadmapK = 0
    windowsize = 1
    sigmaM = 50
    density = 0
    ignore_sparse = False
    orderwindowsize = 0
    orderstep = 3
    cutguest = 0
    enhance_mee = False
    use_rnn = False
    # mee_url = 'http://127.0.0.1:8090/netmatch'
    mee_url = 'http://127.0.0.1:8090/netmatch'
    # mee_url = 'http://192.168.126.193:8090/netmatch'
    # gps_url = 'http://192.168.126.193:9090/netmatch'
    gps_url = 'http://127.0.0.1:8090/netmatch'
    netmatch_url = ''
    SHUFFLE_DATASET = True
    BATCH_SIZE = 3
    USE_CUDA = False
    corridor_width = 30

    GRAPH_NODE_NUM = 301754
    GCN_LAYER = 2
    LEARNING_RATE = 0.0001
    EPOCH_NUM = 10000
    CUDA_DEVICE_NUM = 3
    VALID_EPOCH = 10
    HIDDEN_SIZE = 64
    BATCH_SIZE = 100
    SAVE_MODEL = True
    MATCHER = 'net'  # net|py|java


    def __init__(self):
        pass

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k in kwargs:
            v = kwargs[k]
            if not hasattr(self, k):
                warnings.warn(f"Warning: opt has not attribut {k}")
            setattr(self, k, v)
        # 打印配置信息
        print('user config:')
        for k in self.__dict__:
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def get_mee_file_name(self) -> str:
        return self.mee_path

    def get_mee_obj_name(self) -> str:
        return 'data/mee_nrow-%s.obj' % (str(self.nrows))

    def get_gps_file_name(self) -> str:
        return self.gps_path

    def get_gps_obj_name(self) -> str:
        return 'data/gps_nrow-%s.obj' % (str(self.nrows))

    def get_edge_file_name(self) -> str:
        return self.edge_path

    def get_node_file_name(self) -> str:
        return self.node_path

    def get_map_obj_name(self) -> str:
        return 'data/map_nrow-%s.obj' % str(self.nrows)

    def get_train_pairs_obj(self):
        return 'data/pairs.obj'

    def get_feat_file_name(self):
        return 'data/feat-hiddensize%d.obj' % int(self.HIDDEN_SIZE)

    def get_model_file_name(self, epoch=0, loss=0):
        return "data/model-epoch%d-loss-%.2f.pkl" % (int(epoch), float(loss))

    @property
    def java_match_target_dir_name(self):
        return self.java_src_path + '/data/Beijing-S/input/trajectory/L180_I120_N-1'

    @property
    def java_match_result_dir_name(self):
        return self.java_src_path + '/data/Beijing-S/output/matchResult/L180_I120_N-1'
