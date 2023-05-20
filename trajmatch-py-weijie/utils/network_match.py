import requests

import json


def netmatch(traj_str, java_args, opt, show_emittable=False, url=None):
    data = {
        'line': traj_str,
        'java_args': java_args
    }

    s = requests.session()
    s.keep_alive = False  # 关闭多余连接
    # print(f"NetMatch from {opt.netmatch_url}")
    res = s.post(
        opt.netmatch_url if url is None else url,
        # 'http://192.168.126.193:8090/netmatch',
        data=data
    )
    # print(res.text)

    splitTag = '#'

    if (splitTag not in res.text):
        rids = json.loads(res.text)
    else:
        a, b = res.text.split(splitTag)
        # print(b)
        rids = json.loads(a)
        emitTable = json.loads(b)
        if (show_emittable):
            return rids, emitTable
    return rids

    # rids = list(set(list(rids)))
    # print(f"Received {len(rids)} matched rids from net.")

