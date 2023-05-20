import pymongo
from .mybase import TrajPoint
import pytz
import pickle
from numpy import array
import numpy as np


# DBURL = 'mongodb://127.0.0.1:27017/'
# DBURL = 'mongodb://192.168.126.193:27017/'


class MYDB():

    def create_instance(self,dbname):
        return pymongo.MongoClient(self.DBURL, tz_aware=True,
                                   tzinfo=pytz.timezone('Asia/Shanghai'))[
            dbname]

    def __init__(self, DBURL='mongodb://192.168.134.122:27017/',dbname = "spark"):
        # self.db = \
        #     pymongo.MongoClient('mongodb://192.168.126.193:27017/', tz_aware=True,
        #                         tzinfo=pytz.timezone('Asia/Shanghai'))[
        #         "trajmatch"]
        self.DBURL = DBURL

        self.db = self.create_instance(dbname)
        try:
            with open('data/kdtree.obj', mode='rb') as f:
                self.kdtree = pickle.load(f)
        except Exception as e:
            print(e)
            print("no kdtree file.")

    def get_nid_by_pos(self, pos):
        """
        :param pos: lng,lat using bd09-coord-system with precision 13
        :return:
        """
        res = self.db['nid2pos'].find({'lng': pos[0], 'lat': pos[1]})
        res = list(res)
        if (len(res) > 0):
            return res[0]["_id"]
        else:
            return None

    def get_node_pos_by_nid(self, nid):
        """
        :param nid: int
        :return: (lng,lat)
        """
        res = self.db['nid2pos'].find({'_id': nid})
        res = list(res)
        if (len(res) > 0):
            return res[0]['lng'], res[0]['lat']
        else:
            return None

    def get_origin_gps_line_with_ts(self, vid):
        obj = self.db['gps_trace'].find_one({"_id": vid}, {'trace': 1})
        return [TrajPoint(lat=x['lat'], lng=x['lng'], ts=x['ts'], has_guest=x['has_guest']) for x in obj['trace']]

    def get_gps_line_with_ts(self, vid):

        if ('_' in str(vid)):
            obj = self.db['gps_trace_has_guest'].find_one({"_id": str(vid)}, {'trace': 1})
            return [TrajPoint(lat=x['lat'], lng=x['lng'], ts=x['ts'], has_guest=x['has_guest']) for x in obj['trace']]
        else:
            obj = self.db['gps_trace'].find_one({"_id": int(vid)}, {'trace': 1})
            return [TrajPoint(lat=x['lat'], lng=x['lng'], ts=x['ts'], has_guest=x['has_guest']) for x in obj['trace']]

    def get_mee_line_with_ts(self, vid):
        if ('_' in str(vid)):
            vid = int(vid.split('_')[0])
        obj = self.db['mee_trace'].find_one({"_id": int(vid)}, {'trace': 1})
        return [TrajPoint(lat=x['lat'], lng=x['lng'], ts=x['ts']) for x in obj['trace']]

    def get_gps_vids(self, num=5000):
        # T = self.db['gps_trace_has_guest'].aggregate([{"$sample": {"size": num}}], allowDiskUse=True)
        T = self.db['gps_trace_has_guest'].find({"$where": "this.trace.length > 5"}, {'_id': 1}).limit(num)
        from random import sample
        return list(sample([item['_id'] for item in T], num))

    def get_road_poss(self, rid):
        obj = self.db['roadinfo'].find_one({"_id": rid}, {'nids': 1})
        nids = obj['nids']
        return [self.get_node_pos_by_nid(nid) for nid in nids]

    # def query_column_by_id_order(self, col_name, ids):
    #
    #     querys = {}
    #     for i, _id in enumerate(ids):
    #         querys[f'query{i}'] = [{'$match': {'_id': _id}}]
    #
    #     objs = self.db[col_name].aggregate([
    #         {
    #             '$facet': querys
    #         }
    #     ])
    #
    #     return list(objs)[0]

    def get_muti_road_poss(self, rids: list):

        # ques = self.query_column_by_id_order('roadinfo',rids)
        #
        # nids = []
        #
        # for que in ques.values:
        #     for roadinfo in que:
        #         nids+=roadinfo['nids']

        rids = [int(abs(x)) for x in rids]
        rids = list(set(rids))

        obj = self.db['roadinfo'].find({
            "_id": {
                "$in":
                    rids
            }
        })
        obj = list(obj)
        obj.sort(key=lambda thing: rids.index(thing['_id']))

        nids = []
        for x in obj:
            nids += x['nids']
        nids = list(set(nids))

        obj = self.db['nid2pos'].find({
            "_id": {
                "$in":
                    nids
            }
        })
        obj = list(obj)
        obj.sort(key=lambda thing: nids.index(thing['_id']))

        return [[(p['lng'], p['lat']) for p in obj]]

    def query_ball_points(self, pos, range=100):

        nids = self.kdtree.query(array(pos), k=range)[1].tolist()

        # nids = self.kdtree.query_ball_point(pos,range)
        return list(nids)

    def query_k_nearest(self, pos, k=5):
        nids = self.kdtree.query(pos, k)[1].tolist()
        return list(nids)

    def nid_number(self):
        return self.db['nid2pos'].count()

    def query_ball_roads(self, pos_lines, k):

        if (k <= 0):
            return []

        query_data = []

        # if (type(traj_list[0][0]) is float):
        #     query_data = traj_list  # just one traj line
        # else:
        for line in pos_lines:
            query_data += line

        nids_ = self.kdtree.query(array(query_data), k=k)[1].tolist()
        nids = []
        for line in nids_:
            nids += line
        print(f"{len(nids)}:{nids}")
        items = list(self.db['roadinfo'].find({'nids': {'$in': nids}}))
        return items

    def query_by_pos_lines(self, pos_lines: list, k=10, offset=(0, 0)):
        if (k <= 0):
            return []

        query_data = []

        # if (type(traj_list[0][0]) is float):
        #     query_data = traj_list  # just one traj line
        # else:
        for line in pos_lines:
            query_data += line

        data = self.kdtree.query(array(query_data) + array(offset), k=k)[1].tolist()
        print(f'{data}')

        for nid in data:
            print(nid)
            pos = self.get_node_pos_by_nid(nid)
            yield pos


# dbsession = MYDB(DBURL='mongodb://icodelab.cn:27018/')
# dbsession = MYDB(DBURL='mongodb://192.168.126.193:27017/')
dbsession = MYDB(DBURL='mongodb://192.168.134.122:27017/')
