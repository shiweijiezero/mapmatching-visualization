from utils import *
from config import DefaultConfig

opt = DefaultConfig()

from pyprind import ProgBar

# bar1 = ProgBar(iterations=67330)
# jumpfirst = True
# with open(opt.node_path, mode='r', encoding='utf-8') as fin:
#     with open(opt.node_path + '.bd', mode='w', encoding='utf-8') as fout:
#         for line in fin.readlines():
#             bar1.update()
#             if (jumpfirst):
#                 jumpfirst = False
#                 fout.write("Tower Node ID,Latitude,Longitude\n")
#                 continue
#             line = line.split(',')
#             pos = wgs84_to_bd09(lon=line[2], lat=line[1])
#             s = ",".join([str(x) for x in [line[0], pos[1], pos[0]]]) + '\n'
#             fout.write(s)

bar2 = ProgBar(iterations=92913)
jumpfirst = True
with open(opt.edge_path, mode='r', encoding='utf-8') as fin:
    with open(opt.edge_path + '.bd', mode='w', encoding='utf-8') as fout:
        for line in fin.readlines():
            bar2.update()
            if (jumpfirst):
                jumpfirst = False
                fout.write("edge ID,base node ID,adj node ID,dist,Name,maxSpeed,way,forward access,backward access\n")
                continue
            line = line.split(',')
            way = line[6].split(';')
            poss = zip(way[::2],way[1::2])
            new_pos_list = []
            for raw_pos in poss:
                new_pos = wgs84_to_bd09(lon=raw_pos[1], lat=raw_pos[0])
                new_pos_list.append(new_pos[1])
                new_pos_list.append(new_pos[0])
            line[6] = ';'.join([str(x) for x in new_pos_list])
            s = ",".join(line)
            fout.write(s)
