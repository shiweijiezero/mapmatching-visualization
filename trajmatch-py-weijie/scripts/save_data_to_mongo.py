# 将各类数据存入Mongodb

from utils import *
from config import DefaultConfig
import pymongo
import pyprind

opt = DefaultConfig()


def process_pos(pos):
    return _roundpos(pos)


def _roundpos(pos: tuple):
    PRECISION_ = 12
    return (round(float(pos[0]), PRECISION_), round(float(pos[1]), PRECISION_))


def save_all_endpoint():
    """
    解析并保存所有端点
    """
    nidtable = dbsession.db['nid2pos']
    node_df = pd.read_csv(opt.get_node_file_name(), sep=',', nrows=None, header=0, float_precision='high')
    bar = pyprind.ProgBar(iterations=len(node_df))
    for index, row in node_df.iterrows():
        nid = int(row["Tower Node ID"])
        lat = row["Latitude"]
        lng = row["Longitude"]
        lng, lat = process_pos((lng, lat))
        nidtable.insert_one({'_id': nid, 'lat': lat, 'lng': lng})
        bar.update()


def save_all_road(full=False):
    """
    解析并保存道路信息
    :param full: 是否保存折现中间节点
    """
    mapdb = MYDB()

    road_df = pd.read_csv(opt.get_edge_file_name(), sep=',', nrows=opt.nrows, header=0, float_precision='high')
    bar = pyprind.ProgBar(iterations=92911)

    if (full):
        print("Saving data to table[roadinfo_full]")
        roadinfotable = dbsession.db['roadinfo_full']
    else:
        print("Saving data to table[roadinfo]")
        roadinfotable = dbsession.db['roadinfo']

    for index, row in road_df.iterrows():
        rid = int(row["edge ID"])
        speed = float(row['maxSpeed'])
        nodesinfo = row['way']
        fa = row['forward access']
        ba = row['backward access']
        poss = []
        nodesinfo = nodesinfo.split(';')
        for lat, lng in zip(nodesinfo[0::2], nodesinfo[1::2]):
            pos = process_pos((lng, lat))
            poss.append(pos)

        bid = mapdb.get_nid_by_pos(poss[0])
        eid = mapdb.get_nid_by_pos(poss[-1])
        if not full:
            nids = [bid, eid]
        else:
            nids = poss
            nids[0] = bid
            nids[-1] = eid

        roadinfotable.insert_one({'_id': rid, 'nids': nids, 'speed': speed, 'fa': fa, 'ba': ba})
        bar.update()


from multiprocessing import Pool, cpu_count



def parse_mee_df(df):
    traj_dict = {}

    bar = pyprind.ProgBar(iterations=len(df))
    for _, line in df.iterrows():
        bar.update()
        vid, tstamp, operator, longti, lati = line

        longti, lati = (float(longti), float(lati))

        vid = int(vid)
        if (vid not in traj_dict):
            traj_dict[vid] = []
        # traj_dict[vid].append(Traj(longti, lati, tstamp))
        traj_dict[vid].append({'lat': lati, 'lng': longti, 'ts': tstamp})

    return traj_dict


def parse_gps_df(df):
    traj_dict = {}
    bar = pyprind.ProgBar(iterations=len(df))
    for _, line in df.iterrows():
        bar.update()
        vid, guest_status, tstamp, longti, lati, speed, direction = line
        vid = int(vid)
        if (vid not in traj_dict):
            traj_dict[vid] = []
        longti, lati = gcj02_to_wgs84(float(longti), float(lati))
        traj_dict[vid].append(
            {'lat': lati, 'lng': longti, 'ts': tstamp + datetime.timedelta(hours=-8), 'has_guest': guest_status})
    return traj_dict


def save_gps_trace_to_mongo():
    gps_df = pd.read_csv(opt.get_gps_file_name(), sep=',', nrows=opt.nrows, header=None, parse_dates=[2])
    print("Readed gps file.")
    gps_df.columns = ['vehicle_id', 'useless1', 'tstamp', 'lati', 'longti', 'useless2', 'useless3']
    # gps_df.sort_values('tstamp', inplace=True)

    traj_dict = {}

    n_cores = cpu_count()
    df_split = split_dataframe(gps_df, n_cores)

    pool = Pool(n_cores)
    results = pool.map_async(parse_gps_df, iterable=df_split)
    pool.close()
    pool.join()
    print("Start combining...")
    for part_of_result_dict in results.get():
        for k in part_of_result_dict:
            if (k not in traj_dict):
                traj_dict[k] = part_of_result_dict[k]
            else:
                traj_dict[k] += part_of_result_dict[k]
    print("Start sorting...")
    for k in traj_dict:
        traj_dict[k].sort(key=lambda x: x['ts'])

    bar = pyprind.ProgBar(iterations=len(traj_dict.keys()))
    gps_table = dbsession.db['gps_trace']
    for vid, value in traj_dict.items():
        bar.update()
        gps_table.insert_one({"_id": vid, 'trace': value})


import datetime


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def save_mee_to_mongo():
    print("Reading mees file from [%s]" % opt.get_mee_file_name())
    mee_df = pd.read_csv(
        opt.get_mee_file_name(),
        sep=',', nrows=opt.nrows,
        header=None, parse_dates=[1],
        infer_datetime_format=True,
        date_parser=dateparse)
    mee_df.columns = ['vehicle_id', 'tstamp', 'operator', 'lati', 'longti']

    mee_df.sort_values("tstamp", inplace=True)

    traj_dict = {}

    n_core = cpu_count()

    dfs = split_dataframe(mee_df, n_core)

    pool = Pool(n_core)

    results = pool.map_async(parse_mee_df, iterable=dfs)

    pool.close()
    pool.join()

    print("Start combining...")
    for part_of_result_dict in results.get():
        for k in part_of_result_dict:
            if (k not in traj_dict):
                traj_dict[k] = part_of_result_dict[k]
            else:
                traj_dict[k] += part_of_result_dict[k]
    print("Start sorting...")
    for k in traj_dict:
        traj_dict[k].sort(key=lambda x: x['ts'])

    bar = pyprind.ProgBar(iterations=len(traj_dict.keys()))
    mee_table = dbsession.db['mee_trace']
    for vid, value in traj_dict.items():
        bar.update()
        mee_table.insert_one({"_id": vid, 'trace': value})


# def export_shortest_graph_obj():
#     # from scipy import sparse
#     #
#     # mapdb = MYDB()
#     # C = mapdb.db['roadinfo'].count()
#     #
#     # import numpy as np
#     # adjmatrix = np.empty((C, C))
#     #
#     # bar = pyprind.ProgBar(iterations=mapdb.db['roadinfo'].count())
#     # for item in mapdb.db['roadinfo'].find():
#     #     bar.update()
#     #     nids = item['nids']
#     #     bid = nids[0]
#     #     eid = nids[-1]
#     #     if (item['fa']):
#     #         adjmatrix[bid][eid] = dis_between_pos(mapdb.get_node_pos_by_nid(bid), mapdb.get_node_pos_by_nid(eid))
#     #
#     #     if (item['ba']):
#     #         adjmatrix[bid][eid] = dis_between_pos(mapdb.get_node_pos_by_nid(eid), mapdb.get_node_pos_by_nid(bid))
#     #
#     # adjmatrix = sparse.csr_matrix(adjmatrix)
#     #
#     # with open('data/adjmatrix.obj', mode='wb') as f:
#     #     pickle.dump(adjmatrix, f, protocol=2)
#
#     import scipy
#     from scipy import sparse
#
#     with open('data/adjmatrix.obj', mode='rb') as f:
#         adjmatrix = pickle.load(f)
#     print("Start routing ...")
#     adjmatrix= sparse.csgraph.dijkstra(adjmatrix, directed=True, unweighted=False, min_only=False,
#                                              return_predecessors=False,limit=0.0001)
#
#     np.save('data/adjmatrix2.obj'
#             '',adjmatrix)
#
#     # adjmatrix = sparse.csr_matrix(adjmatrix)
#     #
#     #
#     #
#     # with open('data/adjmatrix2.obj', mode='wb') as f:
#     #     pickle.dump(adjmatrix, f, protocol=2)
#


# def export_route_matrix():
#     import networkx
#     nxgraph = networkx.DiGraph()
#     nxgraph = pickle.load(open('data/nxgraph.obj', mode='rb'))
#     print("Start routing the entire graph...")
#     route_matrix = networkx.all_pairs_dijkstra_path(nxgraph, weight='length', cutoff=5)
#
#     route_matrix = dict(route_matrix)
#
#     with open('data/route_matrix.obj', mode='wb') as f:
#         pickle.dump(route_matrix, f)


if (__name__ == "__main__"):
    # save_all_endpoint()
    # save_all_road()
    # export_txt_file()
    # save_gps_trace_to_mongo()
    # save_mee_to_mongo()
    # export_networkx_obj()
    # export_kd_tree()

    # export_route_matrix()
    save_all_road(full=True)
