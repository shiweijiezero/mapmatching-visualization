from typing import List, Tuple, Set, Dict

#顶点
class Vertex:
    def __init__(self, x: float, y: float, s_num = None):
        self.x = x
        self.y = y
        self.s_num = s_num

    #序号是不重要的, 因此必然不能参与判等和哈希
    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        s_num = ""
        if type(self.s_num) == list:
            for i in range(len(self.s_num)):
                s_num += str(self.s_num[i]) + ","
            s_num = s_num[:-1]
        elif type(self.s_num) == int:
            s_num += str(self.s_num)

        return "({:0>4.2f}, {:0>4.2f})[{: <4}]".format(self.x, self.y, s_num)
    
    def copy(self,s_num=None):
        return Vertex(self.x, self.y, s_num)