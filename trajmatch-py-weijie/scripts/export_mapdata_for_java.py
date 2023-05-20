from utils import *
from config import DefaultConfig
import pymongo
import pyprind

opt = DefaultConfig()

import fire


def export(full=False):
    def get_content(rid, binfo, einfo, othernodes, otherids, speed):
        oinfo = [binfo]
        for oid, othernode in zip(otherids, othernodes):
            # oid = self.nodepos2nid[othernode]
            oinfo.append("%d- %f %f" % (oid, othernode[0], othernode[1]))
        oinfo.append(einfo)

        content_list = [
            rid,
            ','.join([str(x) for x in oinfo]),
            'false',
            f'speed:{speed}'
        ]
        return '|'.join([str(x) for x in content_list])

    beginend_pair = set()

    nidtable = dbsession.db['nid2pos']
    if (full == True):
        roadtable = dbsession.db['roadinfo_full']
    else:
        roadtable = dbsession.db['roadinfo']

    edge_fname = os.path.join(opt.output_dir, 'edges_0.txt')
    vertice_fname = os.path.join(opt.output_dir, 'vertices_0.txt')

    bar = pyprind.ProgBar(iterations=roadtable.count())

    last_interpoint_id = 67331

    with open(edge_fname, mode='w', encoding='utf-8') as f:
        for document in roadtable.find():
            bar.update()
            rid = document['_id']
            nids = document['nids']
            fa = document['fa']
            ba = document['ba']
            bid = nids[0]
            eid = nids[-1]
            speed = document['speed']
            if (len(nids) > 2):
                otherposs = nids[1:-1]
            else:
                otherposs = []
            # if (len(nids) > 2):
            #     otherids = nids[1:-1]
            bnode = dbsession.get_node_pos_by_nid(bid)
            enode = dbsession.get_node_pos_by_nid(eid)
            binfo = ' '.join([str(x) for x in [bid, bnode[0], bnode[1], 'nodeType:0']])
            einfo = ' '.join([str(x) for x in [eid, enode[0], enode[1], 'nodeType:0']])
            otherids = list(range(last_interpoint_id, last_interpoint_id + len(otherposs)))
            # print(otherids)
            last_interpoint_id += len(otherposs)
            if fa:
                if ((bid, eid) not in beginend_pair):
                    content = get_content(rid, binfo, einfo, otherposs, otherids, speed)
                    beginend_pair.add((bid, eid))
                    f.write(content)
                    f.write('\n')
            if ba and rid != 0:
                if ((eid, bid) not in beginend_pair):
                    content = get_content(-1 * rid, einfo, binfo, otherposs[::-1],
                                          [-1 * int(x) for x in otherids[::-1]], speed)
                    beginend_pair.add((eid, bid))
                    f.write(content)
                    f.write('\n')

    bar = pyprind.ProgBar(iterations=nidtable.count())
    with open(vertice_fname, mode='w', encoding='utf-8') as f1:
        for document in nidtable.find():
            bar.update()
            nid = document['_id']
            lat = document['lat']
            lng = document['lng']
            f1.write(f"{nid} {lng} {lat} nodeType:0\n")

    print("Write result to %s,%s" % (edge_fname, vertice_fname))


if (__name__ == '__main__'):
    fire.Fire()
