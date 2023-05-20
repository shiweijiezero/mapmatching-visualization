from flask import Flask
import random
from utils import *
from config import DefaultConfig
from flask_cors import CORS
from functools import partial
import fire
from itertools import chain
import traceback
from functools import reduce
import pdb
import matplotlib
import re

matplotlib.use('AGG')

opt = DefaultConfig()

# matchresult_collection_name = '2020-09-25 10:34:13-matchResult'
# matchresult_collection_name = '2020-11-01 21:27:51-matchResult'

# matchresult_collection_name = '2020-11-02 17:10:13-matchResult'
dbsession = MYDB()
# dbsession = MYDB(DBURL='mongodb://192.168.134.122:27017/')

# dblist=myclient.list_database_names()
collist = list(filter(lambda x:
                      x[-11:] == "matchResult",
                      dbsession.db.list_collection_names()))

# collist=mydb.list_collection_names()
collist.sort()
matchresult_collection_name = collist[-1]
raw_data_collection_name = "pairedData"

from pyecharts.globals import CurrentConfig
from utils.mydb import MYDB


# try:
#     MAP
# except:
#     MAP = load_or_create_map_obj(opt)
#     GPS = load_or_create_gps_obj(opt)
#     MEE = load_or_create_mee_obj(opt)

# MAP, GPS, MEE = None, None, None

# COLOR_SET = ['#003399', '#996600', '#6600cc', '#009900', '#000000cc', '#990000', '#993366', '#996633', '#996833']
COLOR_SET = ['#003399', '#996600', '#6600cc', '#009900', '#000000cc', '#009933', '#993366', '#996633', '#996833']

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/getvids', methods=['GET'])
def getvids():
    num = 500
    T = list(dbsession.db[matchresult_collection_name].find({}, {'id': 1}).limit(num))
    from random import sample
    ids = list(sample([item['id'] for item in T], num if len(T) > num else len(T)))
    import json
    return json.dumps(ids)


def cat_lines(lines):
    new_lines = []
    temp_line = []
    for line in lines:
        if (len(temp_line) == 0):
            temp_line += line
        else:
            if (temp_line[-1] == line[0]):
                temp_line += line[1:]
            else:
                new_lines.append(temp_line)
                temp_line = line
    if (len(temp_line) != 0):
        new_lines.append(temp_line)
    return new_lines


def concat_lines(lines):
    new_lines = []
    temp_line = []
    for line in lines:
        if (len(temp_line) == 0):
            temp_line += line
        else:
            if (temp_line[-1] == line[0]):
                temp_line += line[1:]
            else:
                new_lines.append(temp_line)
                temp_line = line
    if (len(temp_line) != 0):
        new_lines.append(temp_line)
    return new_lines


def raw_strToTrajectory(s):
    res = []
    for p in s.split(';'):
        t = p.split(' ')
        res.append((float(t[0]), float(t[1])))
    return res


@app.route(
    '/domatch/<int:cutguest>/<int:javaport>/<int:result_type>/<string:vid>/<int:interval>/<int:speed>/<int:angle>/<int:roadmapK>/<int:windowsize>/<int:sigmaM>/<int:corridor_width>/<string:java_args>/<int:orderwindow>/<int:orderstep>/<int:ignore_sparse>',
    methods=['GET', 'POST'])
def render_match_result(vid, cutguest, javaport, result_type, interval, speed, angle, roadmapK,
                        windowsize,
                        sigmaM,
                        corridor_width, java_args, orderwindow, orderstep, ignore_sparse):
    # opt.netmatch_url = opt.netmatch_url_temp % (javaport)

    res = dbsession.db[matchresult_collection_name].find_one({"id": vid})

    raw_data = dbsession.db[raw_data_collection_name].find_one({"id": vid})
    rawGPS = raw_strToTrajectory(raw_data["gpsString"])
    rawMEE = raw_strToTrajectory(raw_data["meeString"])

    GPSemitTable = json.loads(res['gpsEmitTableStr'])
    MEEemitTable = json.loads(res['meeEmitTableStr'])

    filteredGPS = strToTrajectory(str(res['filtered_gps']))
    filteredMEE = strToTrajectory(str(res['filtered_mee']))

    gps_nodes = res['gpsmatch_Nodes']
    mee_nodes = res['meematch_Nodes']
    corridor_lines = [[
        (x['_1'], x['_2']) for x in res['corridor_lines']
    ]]
    gps_matched_lines_ = [
        reduce
            (
            lambda x, y: x + y,
            map(
                lambda x: [(x[0], x[1]), (x[2], x[3])],
                [
                    [float(x) for x in p.split(' ')] for p in gps_nodes
                ]
            )
        )
    ]
    gps_matched_lines = concat_lines(gps_matched_lines_)

    mee_matched_lines_ = [
        reduce
            (
            lambda x, y: x + y,
            map(
                lambda x: [(x[0], x[1]), (x[2], x[3])],
                [
                    [float(x) for x in p.split(' ')] for p in mee_nodes
                ]
            )
        )
    ]
    mee_matched_lines = concat_lines(mee_matched_lines_)
    mee_coor = list(map(
        lambda x: [(x[0], x[1]), (x[2], x[3])],
        [
            [float(x) for x in p.split(' ')] for p in mee_nodes
        ]
    ))

    # corridor_lines = list(map(
    #     partial(get_corridor_lines, corridor_width=corridor_width)
    #     ,
    #     mee_matched_lines
    # ))

    # -----------------------------------------
    corridor_lines, gps_contours, hierarchy = draw_contours(mee_coor, dpi=200, width=corridor_width, k=0.5)

    level_contours = operate_tree(gps_contours, hierarchy[0])
    shapely_CMF = shapely_cmf(gps_matched_lines[0], level_contours)

    # -----------------------------------------

    print("Do processing after matching...")

    # filteredGPS = [gcj02_to_wgs84(pos[0], pos[1]) for pos in filteredGPS]

    data_ = [
        [filteredGPS],
        [filteredMEE],
        gps_matched_lines,
        mee_matched_lines,
        corridor_lines,
        [rawGPS],
        [rawMEE]
    ]

    # # remove dup points for plot
    # for i, target in enumerate(data_):
    #     data_[i] = list(map(
    #         remove_duplicated_points,
    #         target
    #     ))

    # moded wgs84 to bd09
    for i, target in enumerate(data_):
        data_[i] = list(map(
            lambda line: [wgs84_to_bd09(pos[0], pos[1]) for pos in line],
            target
        ))

    filtered_gps_data, \
    filtered_mee_data, \
    gps_matched_lines, \
    mee_matched_lines, \
    corridor_lines, \
    raw_gps_data, \
    raw_mee_data = data_

    data = [
        {
            'name': '原始GPS轨迹',
            'data': raw_gps_data,
        },
        {
            'name': '原始蜂窝轨迹',
            'data': raw_mee_data,
        },
        # {
        #     'name': 'GPS过滤',
        #     'data': filtered_gps_data,
        # },
        {
            'name': 'GPS匹配结果',
            'data': gps_matched_lines,
        },
        {
            'name': '过滤后蜂窝轨迹',
            'data': filtered_mee_data,
        },
        {
            'name': '蜂窝轨迹匹配结果',
            'data': mee_matched_lines,
        },
        {
            'name': '匹配结果走廊',
            'data': corridor_lines,
        }
    ]

    for i, d in enumerate(data):
        d['color'] = COLOR_SET[i]

    print("Caculating RMF ...")

    rmf = RMF(gps_matched_lines, mee_matched_lines)
    cmf = CMF(gps_matched_lines, mee_matched_lines, corridor_width=corridor_width)
    print(f"RMF:{rmf:.3f} CMF:{cmf:.3f} shapely_CMF:{shapely_CMF:.3f}")
    print("Rendering...")

    # interested_features = ['meePointsIntervalMaxDistance', 'meePointsIntervalAvgDistance', "meePointsNum",
    #                        "meePointsIntervalMaxTime", "meePointsIntervalAvgTime",
    #                        "meePointMaxDistanceProportionInLength", "gpsPointMaxDistanceProportionInLength", "id",
    #                        "rmf", 'cmf-radius50']
    #
    #
    # title_dict = {"ShapelyCMF": f"{shapely_CMF:.3f}"}
    #
    # for k in interested_features:
    #     if (type(res[k]) is float):
    #         title_dict[k] = f"{res[k]:.3f}"
    #     else:
    #         title_dict[k] = res[k]
    # conf = res['conf'].split(',')
    # title_dict['======'] = '|'
    # for p in conf:
    #     k, v = p.split(':')
    #     title_dict[k] = v

    # c = show_in_bd_map(data=data, center=(gps_matched_lines[0][0]), title=make_titile(title_dict))
    c = show_in_bd_map(data=data, center=(gps_matched_lines[0][0]), title="")


    # ========================add road point====================================
    if (roadmapK > 0):

        try:
            print(f"MEE CandiNum:{[len(v) for sample_pos_str, v in dict(MEEemitTable).items()]}")
            add_emit_data(c, MEEemitTable, "MEE发射概率")

            print(f"GPS CandiNum:{[len(v) for sample_pos_str, v in dict(GPSemitTable).items()]}")
            add_emit_data(c, GPSemitTable, "GPS发射概率")

            # add_road_speed_scatter(c,mee_result_rids,"MEE������������")

        except Exception as e:
            traceback.print_exc()
            pass

    # =========================================================================================
    print("Done, return result.")
    # return c.render_embed()
    return replace_static_cdn(c.render_embed())


def make_titile(data_dict: dict):
    res = "           "
    for k, v in data_dict.items():
        res += str(k) + ":" + str(v) + "\n           "
    return res


#
def replace_static_cdn(html_s: str):
    return str(html_s). \
        replace(
        "https://assets.pyecharts.org/assets/echarts.min.js",
        "https://cdn.bootcss.com/echarts/4.4.0-rc.1/echarts.js"
    ). \
        replace(
        "https://assets.pyecharts.org/assets/bmap.min.js",
        "https://cdn.bootcdn.net/ajax/libs/echarts/4.4.0-rc.1/extension/bmap.min.js"
    )


def add_emit_data(c, emittable, col_name):
    # add emit prob line
    # emitdatapair = []

    for sample_pos_str, v in list(dict(emittable).items()):
        emitdatapair = []
        extension_data = []
        sample_position = eval(sample_pos_str)
        sample_position = wgs84_to_bd09(*sample_position)
        for cand in v:
            seg_begin_position, seg_end_position, match_point_position, probInfo, segid = eval(cand)

            seg_begin_position = myround(wgs84_to_bd09(*seg_begin_position))
            seg_end_position = myround(wgs84_to_bd09(*seg_end_position))
            match_point_position = myround(wgs84_to_bd09(*match_point_position))

            emitColor = "#730310"
            roadsegColor = "#C60681"

            re_match_result = re.findall("mc(\d*)-ts(\d*)-tag(-*\d*)(-st\w*)*", probInfo)
            sourceType = re_match_result[0][3]

            if (sourceType != "-storigin" and sourceType != ''):
                # continue
                if (sourceType == "-stextension"):
                    emitColor = "#0066ff"
                    roadsegColor = "#880335"
                if (sourceType == "-stpickup"):
                    emitColor = "#333399"
                    roadsegColor = "#7FA757"

                extension_data.append({
                    'coords': [
                        list(match_point_position),
                        list(sample_position),
                    ],
                    # 'value': f"{[seg_begin_position, seg_end_position, match_point_position, segid, probInfo]}",
                    'value': f"{[segid, probInfo]}",
                    'lineStyle': {
                        'opacity': 1,
                        'width': 3,
                        'color': emitColor
                    },
                })

                extension_data.append({
                    'coords': [
                        list(seg_begin_position),
                        list(seg_end_position),
                    ],
                    'value': f"{segid}",
                    'lineStyle': {
                        'opacity': 1,
                        'width': 3,
                        'color': roadsegColor
                    },
                })

            else:
                emitdatapair.append({
                    'coords': [
                        list(match_point_position),
                        list(sample_position),
                    ],
                    # 'value': f"{[seg_begin_position, seg_end_position, match_point_position, segid, probInfo]}",
                    'value': f"{[segid, probInfo]}",
                    'lineStyle': {
                        'opacity': 1,
                        'width': 1.5,
                        'color': emitColor
                    },
                })

                emitdatapair.append({
                    'coords': [
                        list(seg_begin_position),
                        list(seg_end_position),
                    ],
                    'value': f"{segid}",
                    'lineStyle': {
                        'opacity': 1,
                        'width': 1.5,
                        'color': roadsegColor
                    },
                })

        # print(emitdatapair)

        c.add(series_name=col_name,
              type_='lines',
              data_pair=emitdatapair,
              is_large=True,
              effect_opts=opts.EffectOpts(is_show=False, symbol_size=0.1),
              is_polyline=False,
              symbol='none',
              label_opts=opts.LabelOpts(is_show=False, formatter="{c}", font_size=8, position='insideBottomLeft'),
              # is_selected=False
              )

        if (len(extension_data) > 0):
            c.add(series_name=col_name + "拓展点",
                  type_='lines',
                  data_pair=extension_data,
                  is_large=True,
                  effect_opts=opts.EffectOpts(is_show=False, symbol_size=0.1),
                  is_polyline=False,
                  symbol='none',
                  label_opts=opts.LabelOpts(is_show=False, formatter="{c}", font_size=8, position='insideBottomLeft'),
                  # is_selected=False
                  )


@app.route('/cleardata/<string:vid>', methods=['GET'])
def clear_data(vid):
    dirnames = list(os.listdir(opt.output_dir))
    target_dirs = list(filter(lambda x: str(x).split('_')[0] == str(vid), dirnames))
    if (len(target_dirs) == 0):
        return "Nothing to clear."
    else:
        for targetdir in target_dirs:
            os.system("rm %s -rf" % os.path.join(opt.output_dir, targetdir))
        return "remove: [%s]" % ','.join(target_dirs)


def runserver(**kwargs):
    global matchresult_collection_name
    opt.parse(kwargs)
    if ('tablename' in kwargs):
        matchresult_collection_name = kwargs['tablename']
    app.run(host='0.0.0.0', port=opt.web_server_port, debug=True, use_reloader=False)
    # app.run(host='127.0.0.1', port=opt.web_server_port, debug=True, use_reloader=False)


if (__name__ == '__main__'):
    fire.Fire()
    # runserver(tablename="2021-03-05 06:36:57-matchResult")
