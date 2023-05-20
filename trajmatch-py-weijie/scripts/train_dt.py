# mongoexport -d spark -c "2020-12-09 18:22:48-matchResult" -f meePointsIntervalMaxDistance,meePointsIntervalAvgDistance,meePointsIntervalMedianDistance,meePointsNum,meePointsIntervalMaxTime,meePointsIntervalAvgTime,meePointMaxDistanceProportionInLength,gpsPointMaxDistanceProportionInLength,cmf-radius50  --csv -o ./all-for-dt.csv
import graphviz
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree


#
# def trans(x):
#     if (x >= 0.5):
#         return '0'
#     if (x < 0.2):
#         return '1'
#     return '2'

def trans(x):
    if (x >= 0.2):
        return '0'
    return '1'


def load_data(fname):
    data = pd.read_csv(fname, sep=',', header=0)

    data['label'] = data['cmf-radius50'].apply(trans)

    print("good line number:", data[data['label'] == 1].count())

    return data


def do_decision_tree(data: pd.DataFrame):
    x = data.iloc[:, :-2]
    y = data.label
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=20,
                                 max_leaf_nodes=20)
    dtc.fit(x, y)
    print('准确率：', dtc.score(x, y))

    pred = dtc.predict(x)

    from io import StringIO
    dot_data = StringIO()
    export_graphviz(dtc, label='root', filled=True, rounded=True, special_characters=True,
                    class_names=["bad", "good"], proportion=True,
                    feature_names=x.columns,
                    out_file=dot_data)

    import pydotplus

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('diabetes.png')
    import matplotlib.pyplot as plt

    import pydotplus


# def see_dot():
#     import pydot
#     (graph,) = pydot.graph_from_dot_file('data/tree.dot')
#     graph.write_png('data/tree.png')
#     graph.write_pdf('data/tree.pdf')

if (__name__ == '__main__'):
    data = load_data("data/all-for-dt.csv")
    do_decision_tree(data)
    # see_dot()
