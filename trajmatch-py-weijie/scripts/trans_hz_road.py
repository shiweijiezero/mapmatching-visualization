from utils import *
from config import DefaultConfig

opt = DefaultConfig()

noded2idict = {}
id2nodedict = {}

transed_hz_road_path = 'edges_0.txt'
transed_hz_node_path = 'vertices_0.txt'


def write_road(rid, beginid, endid, othernodes):
    with open(transed_hz_road_path, mode='a', encoding='utf-8') as f:
        bnode = id2nodedict[beginid]
        enode = id2nodedict[endid]
        binfo = ' '.join([str(x) for x in [beginid, bnode[1], bnode[0], 'nodeType:0']])
        einfo = ' '.join([str(x) for x in [endid, enode[1], enode[0], 'nodeType:0']])

        oinfo = [binfo]

        for othernode in othernodes:
            if (othernode not in noded2idict):
                oid = len(noded2idict)
                noded2idict[oid] = othernode
                print("add new node: " + str(othernode))
            else:
                oid = noded2idict[othernode]

            oinfo.append("%d- %f %f" % (oid, othernode[1], othernode[0]))

        oinfo.append(einfo)

        content_list = [
            rid,
            ','.join([str(x) for x in oinfo]),
            'false'
        ]
        f.write('|'.join([str(x) for x in content_list]))
        f.write('\n')


def tramsform_roadmap():
    with open(opt.node_path, mode='r') as f:
        for line in f.readlines():
            id, lat, lng = line.split(',')
            id, lat, lng = int(id), float(lat), float(lng)
            id2nodedict[id] = (lat, lng)
            noded2idict[(lat, lng)] = id
    print("Finish node info")
    with open(opt.edge_path, mode='r') as f:
        for line in f.readlines():
            rid, beginid, endid, rleng, rname, rspeed, nodesinfo, type1, type2 = line.split(',')
            nodesinfo = nodesinfo.split(';')
            i = 1  # !!
            othernodes = []
            while (i < (len(nodesinfo) / 2) - 1):
                lat = float(nodesinfo[2 * i])
                lng = float(nodesinfo[2 * i + 1])
                i += 1
                othernodes.append((lat, lng))
            write_road(int(rid), int(beginid), int(endid), othernodes)
    print("Finih road info.")

    with open(transed_hz_node_path, mode='a') as f:
        for nid in id2nodedict:
            f.write(' '.join([str(x) for x in [
                nid,
                id2nodedict[nid][1],
                id2nodedict[nid][0],
                'nodeType:0'
            ]]) + '\n')

    print("Finish")


tramsform_roadmap()
