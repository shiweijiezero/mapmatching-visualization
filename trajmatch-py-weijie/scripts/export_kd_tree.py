
from utils import *
from config import DefaultConfig
import pymongo
import pyprind

opt = DefaultConfig()



def export_kd_tree():
    from scipy.spatial.kdtree import KDTree
    import numpy as np

    mapdb = MYDB()
    bar = pyprind.ProgBar(iterations=mapdb.db['nid2pos'].count())
    poss = np.zeros((mapdb.db['nid2pos'].count() + 1, 2))

    for item in mapdb.db['nid2pos'].find():
        bar.update()
        _id = int(item['_id'])
        poss[_id, :] = item['lng'], item['lat']

    kdtree = KDTree(data=poss)
    with open('data/kdtree.obj', mode='wb') as f:
        pickle.dump(kdtree, f, protocol=2)
