import fire


def a(**kwargs):
    fname = kwargs['fname']

    res = []
    with open(fname, 'r') as f:
        for line in f:
            if ('##' in line):
                res.append(line.lstrip('##RMF avg: ').rstrip('\n'))

    results = []
    for tu in zip(res[0::2], res[1::2]):
        tu = list(tu)
        tu[0] = float(tu[0])
        results.append(tu)

    results.sort(key=lambda x: x[0])

    for one in results:
        print(one)


if (__name__ == '__main__'):
    fire.Fire()
