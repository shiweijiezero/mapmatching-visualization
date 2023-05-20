from utils import *
from config import DefaultConfig
import pdb
import matplotlib.pyplot as plt
from numpy import array

opt = DefaultConfig()

opt.nrows = None
MEE = load_or_create_mee_obj(opt)

# m = load_or_create_map_obj(opt)
#
# G = load_or_create_gps_obj(opt)
#
# import random
#
# vid = random.choice(list(G.traj_dict.keys()))
# # vid = 5670
# vid = 243204
# # vid = 9803  # very long
# print(vid)
#
# t0 = G.get_raw_gps_line(vid)
#
# t1 = gps_interval_filter(vid, G.traj_dict[vid])
# t1 = [x.gps_pos_line() for x in t1]
# # print(t1)
#
# roads = m.query_by_traj([t0], k=500)
#
# # c = show_in_bd_map(t1, roads, road_offset=(0.01121, 0.003835), gps_offset=(0.01121 - 0.00475, 0.003835 + 0.0025))
# c = show_in_bd_map(data=[
#     {
#         'name': '路网',
#         'data': roads,
#         'color': 'purple',
#         'offset': (0.01121, 0.003835)
#     },
#     {
#         'name': '原始GPS轨迹',
#         'data': [t0],
#         'color': 'red',
#         'offset': (0.01121 - 0.00475, 0.003835 + 0.00168)
#     },
#     {
#         'name': '分段后GPS轨迹',
#         'data': t1,
#         'color': 'green',
#         'offset': (0.01121 - 0.00475, 0.003835 + 0.00168)
#     },
# ], center=(t1[0][0]))
# c.render()
