from utils import *
from config import DefaultConfig

import random
import datetime
opt = DefaultConfig()
opt.nrows = 10000

def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


df = pd.read_csv(
            opt.get_mee_file_name(),
            sep=',', nrows=opt.nrows,
            header=None, parse_dates=[1],
            infer_datetime_format=True,
            date_parser=dateparse)
df.columns = ['vehicle_id', 'tstamp', 'operator', 'lati', 'longti']
df.sort_values('tstamp', inplace=True)
traj_dict = {}

for linenum in range(1, opt.nrows):
    print(linenum)
    vid, tstamp, operator, lati, longti  = df.iloc[linenum, :]
    vid = vid
    if (vid not in traj_dict):
        traj_dict[vid] = []
    traj_dict[vid].append(TrajPoint(longti, lati, tstamp))

i = 0
for tid in random.sample(traj_dict.keys(), 10):
    with open('trajs/trip_%d.txt' % i, mode='a') as f:
        for trajoj in traj_dict[tid]:
            f.write(' '.join([str(x) for x in [trajoj.lat, trajoj.lng, int(trajoj.ts.value/1000000000)]]))
            f.write('\n')
    i += 1
