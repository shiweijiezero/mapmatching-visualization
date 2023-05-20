from typing import List, Tuple, Set, Dict
#三维向量
class D3Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    #返回一个新向量, 是自己的归一化向量
    def get_normalizeion(self):
        module = self.get_module()
        return D3Vector(self.x / module, self.y / module, self.z / module)

    #返回本向量的模长
    def get_module(self):
        return (pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2))**0.5

    #将自身放缩到模长为n
    def zoom_to_module(self, n):
        module = self.get_module()
        self.x = self.x / module * n
        self.y = self.y / module * n
        self.z = self.z / module * n

#二维向量(其实是哑化了z轴的三维向量)
class D2Vector(D3Vector):
    def __init__(self, x, y):
        super().__init__(x, y, 0)
