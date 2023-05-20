import numpy as np
from utils import load_or_create_map_obj

import json
import os

from pyecharts import options as opts
from pyecharts.charts import BMap, Page
from pyecharts.faker import Collector, Faker
from pyecharts.globals import BMapType
from config import DefaultConfig
from utils.mymap import MapDataset

# ------------------------------
from .polygon_union.draw_rectangle import draw_contours


# -------------------------------


def plot_traj(plt, traj, label, marker='.', c=None, linewidth=2, markersize=12):
    x = np.array([x.lat for x in traj])
    y = np.array([x.lng for x in traj])
    # plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(x, y, marker=marker, c=c if c else np.random.rand(3, ), linewidth=linewidth, markersize=markersize,
             label=label)

def show_in_bd_map(data: list, center=(), title="Result"):
    # data = [
    #     {'name': 'roadmap',
    #      'data': trajs,
    #      'color': 'purple',
    #      'offset': (0, 0)
    #      }
    # ]

    BAIDU_MAP_AK = 'dXsIyeWSfumKGtGsI9ALfwNeG1zHXukk'
    # data_from_roadmap = trajlines_to_bd_json(roads, offset=road_offset)
    # data_from_trajs = trajlines_to_bd_json(trajs, offset=gps_offset)

    # if (center == None):
    #     center = data_from_trajs[0]['coords'][0]
    c = (
        # BMap(init_opts=opts.InitOpts(width="1280px", height="675px"))
        BMap(init_opts=opts.InitOpts(width="1300px", height="730px"))
            .add_schema(
            baidu_ak=BAIDU_MAP_AK,
            center=center,
            zoom=14,
            is_roam='move',
            map_style={
                "styleJson": [
                    {
                        "featureType": "water",
                        "elementType": "all",
                        "stylers": {"color": "#d1d1d1"},
                    },
                    {
                        "featureType": "land",
                        "elementType": "all",
                        "stylers": {"color": "#f3f3f3"},
                    },
                    {
                        "featureType": "railway",
                        "elementType": "all",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "highway",
                        "elementType": "all",
                        "stylers": {"color": "#fdfdfd"},
                    },
                    {
                        "featureType": "highway",
                        "elementType": "labels",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "geometry",
                        "stylers": {"color": "#fefefe"},
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "geometry.fill",
                        "stylers": {"color": "#fefefe"},
                    },
                    {
                        "featureType": "poi",
                        "elementType": "all",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "green",
                        "elementType": "all",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "subway",
                        "elementType": "all",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "manmade",
                        "elementType": "all",
                        "stylers": {"color": "#d1d1d1"},
                    },
                    {
                        "featureType": "local",
                        "elementType": "all",
                        "stylers": {"color": "#d1d1d1"},
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "labels",
                        "stylers": {"visibility": "off"},
                    },
                    {
                        "featureType": "boundary",
                        "elementType": "all",
                        "stylers": {"color": "#fefefe"},
                    },
                    {
                        "featureType": "building",
                        "elementType": "all",
                        "stylers": {"color": "#d1d1d1"},
                    },
                    {
                        "featureType": "label",
                        "elementType": "labels.text.fill",
                        "stylers": {"color": "#999999"},
                    },
                ]
            },
        )
            #     .add(
            #     "roadmap",
            #     type_="lines",
            #     data_pair=data_from_roadmap,
            #     is_polyline=True,
            #     is_large=True,
            #     linestyle_opts=opts.LineStyleOpts(color="purple", opacity=0.5, width=1),
            # )
            #     .add(
            #     "rawgps",
            #     type_="lines",
            #     data_pair=data_from_trajs,
            #     is_polyline=True,
            #     is_large=True,
            #     linestyle_opts=opts.LineStyleOpts(color="red", opacity=0.7, width=1.2),
            # )
            .add_control_panel(
            maptype_control_opts=opts.BMapTypeControlOpts(
                type_=BMapType.MAPTYPE_CONTROL_DROPDOWN
            ),
            scale_control_opts=opts.BMapScaleControlOpts(),
            overview_map_opts=opts.BMapOverviewMapControlOpts(is_open=True),
            navigation_control_opts=opts.BMapNavigationControlOpts(),

        )
            .set_global_opts(title_opts=opts.TitleOpts(title=title), tooltip_opts=opts.TooltipOpts(formatter="{c}"))

    )

    for item in data:

        c.add(
            series_name=item['name'],
            type_="lines",
            data_pair=trajlines_to_bd_json(item['data'], offset=item['offset'] if 'offset' in item else (0, 0)),
            is_polyline=True,
            is_large=True,
            effect_opts=opts.EffectOpts(is_show=False, symbol='arrow', period=10, symbol_size=6),
            # label_opts=opts.LabelOpts(is_show=True,formatter="{@name}"),
            linestyle_opts=opts.LineStyleOpts(color=item['color'], opacity=1, width=3),
            itemstyle_opts={
                'normal': {
                    'color': 'black'
                }
            }
        )

    return c


def trajlines_to_bd_json(trajs, offset=(0, 0)):
    rst = []
    for line in trajs:
        if (len(line) > 1):
            rst.append({
                # 'label': 'label',
                # 'name': 'name',
                'coords': [
                    [float(x[0]) + offset[0], float(x[1]) + offset[1]] for x in line
                ]
            })
        else:
            print(f"line only have one point.ignored.")
    return rst

#
#
# def add_point_to_chart(c,point_pos:list):
#     c.add(
#         series_name=item['name'],
#         type_="lines",
#         data_pair=trajlines_to_bd_json(item['data'], offset=item['offset']),
#         is_polyline=True,
#         is_large=True,
#         linestyle_opts=opts.LineStyleOpts(color=item['color'], opacity=0.5, width=1
#     )
