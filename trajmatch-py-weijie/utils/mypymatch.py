from .mybase import dis_between_pos
from .mydb import dbsession
import networkx
import pickle
import math
import numpy as np
import redis

nxgraph = networkx.DiGraph()
route_matrix = {}

try:
    nxgraph = pickle.load(open('data/nxgraph.obj', mode='rb'))
    # route_matrix = networkx.all_pairs_dijkstra_path(nxgraph, weight='length', cutoff=1000)
except:
    print("No nxgraph obj.")

redis_pool = redis.ConnectionPool(host='192.168.126.193', port=6379)


#

# def mymatch(lmap: InMemMap, opt, gps_pos_line: list):
#     trip = [x[::-1] for x in gps_pos_line]
#
#     matcher = DistanceMatcher(lmap, max_dist_init=100, obs_noise=10, obs_noise_ne=10,
#                               non_emitting_states=False, only_edges=False)
#     states, _ = matcher.match(trip)
#     nodes = matcher.path_pred_onlynodes
#
#     return nodes

def gaussian(sigma, x, u):
    # y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    y = 1.0 / sigma * math.exp(-(x - u) / sigma)
    return y


def emit_prob(observed_pos, hidden_state_pos, opt):
    dis = dis_between_pos(observed_pos, hidden_state_pos)
    # print(f"DIS:{dis}")
    return gaussian(opt.sigmaM, dis, 0)
    # up = math.exp(-1 * (dis_between_pos(observed_pos, hidden_state_pos) ** 2) / (2 * opt.sigmaM ** 2))
    # down = (2 * math.pi) ** 0.5 * opt.sigmaM
    #
    # if (up == 0):
    #     up = 0.00001
    # return up / down


def route_dis_by_nid(prev_nid, this_nid):
    # redis_connection = redis.Redis(connection_pool=redis_pool)
    a, b = (prev_nid, this_nid) if prev_nid < this_nid else (this_nid, prev_nid)

    key = str((a, b))
    rst = list(dbsession.db['shortest'].find({'_id': key}))
    if (len(rst) == 0):
        # if ((a, b) not in route_matrix):
        try:
            dis = networkx.astar_path_length(nxgraph, source=a, target=b, weight='length')
            # dis = dis_between_pos(dbsession.get_node_pos_by_nid(a), dbsession.get_node_pos_by_nid(b))
        except:
            dis = 1000000
        dbsession.db['shortest'].insert({'_id': key, 'length': dis})
        route_matrix[(a, b)] = dis
    else:
        dis = rst[0]['length']
        # dis = route_matrix[(a, b)]
    # route_matrix[(a, b)] = dis
    # dis = networkx.shortest_path_length(nxgraph, source=a, target=b, weight='length')
    return dis

    key = f"{a}_{b}"

    if (redis_connection.exists(key)):
        return float(redis_connection.get(key))
    else:
        # print(f"Not found key {key}")
        try:
            dis = networkx.astar_path_length(nxgraph, source=a, target=b, weight='length')
        except:
            dis = 1000000
        redis_connection.set(key, dis)
        return dis

    raise Exception("not found key in redis")


# route_dis = networkx.shortest_path_length(nxgraph, prev_nid, this_nid, 'length')


def trans_prob(prev_nid, this_nid, opt, cache=None):
    # global floyd_matrix
    # if(floyd_matrix is None):
    #     floyd_matrix = networkx.floyd_warshall(nxgraph)
    if ((prev_nid, this_nid) in cache):
        return cache[(prev_nid, this_nid)]
    else:
        route_dis = route_dis_by_nid(prev_nid, this_nid)
        dis_between = dis_between_pos(dbsession.get_node_pos_by_nid(prev_nid), dbsession.get_node_pos_by_nid(this_nid))
        prob = gaussian(opt.sigmaM, route_dis, dis_between)
        cache[(prev_nid, this_nid)] = prob
        # print(f"routedis:{route_dis},transprob:{prob}")
        return prob


def mymatch(opt, gps_pos_lines: list):
    cache = {}

    for gps_pos_line in gps_pos_lines:
        print(f"Len of this line: {len(gps_pos_line)}")

        canditate_range = 10
        for para_str in opt.java_args.split(' '):
            if (para_str[0:3] == '-mc'):
                canditate_range = int(para_str.lstrip('-mc'))

        # candidate_layers = [MAP.query_by_pos_lines([[pos]], k=canditate_range, return_type='nids') for pos in
        #                     gps_pos_line]
        candidate_layers = [dbsession.query_ball_points(pos, canditate_range) for pos in gps_pos_line]

        prob_matrix = []

        for layer_id in range(len(candidate_layers)):
            print(f"Constructing {layer_id}-th layer...")
            prob_matrix.append({})
            this_pos = gps_pos_line[layer_id]
            layer = candidate_layers[layer_id]
            for canditate_nid in layer:
                candidate_pos = dbsession.get_node_pos_by_nid(canditate_nid)

                eprob = emit_prob(this_pos, candidate_pos, opt)

                if (layer_id == 0):
                    prob_matrix[layer_id][canditate_nid] = {'prob': eprob,
                                                            'path': [canditate_nid]}
                else:

                    # prob_matrix[layer_id][canditate_nid] = {'prob': 0, 'path': []}
                    max_prob = 0
                    max_path = []
                    for prev_nid in prob_matrix[layer_id - 1].keys():
                        new_prob = trans_prob(prev_nid=prev_nid, this_nid=canditate_nid, opt=opt, cache=cache) * \
                                   eprob
                        new_prob = prob_matrix[layer_id - 1][prev_nid]['prob'] + new_prob
                        # prob = trans_prob(prev_nid=prev_nid, this_nid=canditate_nid, opt=opt) * \
                        #        prob_matrix[layer_id - 1][prev_nid]['prob']
                        if (new_prob > max_prob):
                            max_prob = new_prob
                            max_path = prob_matrix[layer_id - 1][prev_nid]['path'] + [canditate_nid]
                    prob_matrix[layer_id][canditate_nid] = {'prob': max_prob,
                                                            'path': max_path}
                # print(f"{layer_id},{canditate_nid},{prob_matrix[layer_id][canditate_nid]}")

        LAST_LAYER = len(candidate_layers) - 1

        # 遍历最后一层
        max_prob = 0
        max_path = []
        for canditate_nid in candidate_layers[-1]:
            if prob_matrix[LAST_LAYER][canditate_nid]['prob'] > max_prob:
                max_prob = prob_matrix[LAST_LAYER][canditate_nid]['prob']
                max_path = prob_matrix[LAST_LAYER][canditate_nid]['path']

        # rst += max_path
        rst = set()

        for prev_nid, this_nid in zip(max_path[:-1], max_path[1:]):
            try:
                seg_nids = networkx.shortest_path(nxgraph, prev_nid, this_nid, 'length')
                for prev_nid_in_seg, this_nid_in_seg in zip(seg_nids[:-1], seg_nids[1:]):
                    seg_pos_tuple = dbsession.get_node_pos_by_nid(prev_nid_in_seg), dbsession.get_node_pos_by_nid(
                        this_nid_in_seg)
                    rst.add(seg_pos_tuple)
            except:
                print(f"No path from {prev_nid} ot {this_nid}")
            # try:
            #     rst.append([MAP.get_node_by_id(nid) for nid in networkx.shortest_path(nxgraph, prev_nid, this_nid, 'length')])
            # except:
            #     print(f"No path from {prev_nid} ot {this_nid}")
    return list(rst)  # [rids]
