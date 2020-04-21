import numpy as np


class Node(object):
    __slots__ = ["_id", "_type", "parent", "data", "model"]

    def __init__(self, _id=None, _type=None, parent=None, data=None, model=None):
        self._type = _type
        self._id = _id
        self.data = data or []
        self.model = model or []
        self.parent = parent

    def __getitem__(self, item):
        return getattr(self, item, 0)

    def add_data(self, data=None):
        self.data = data

    def del_data(self):
        self.data = None

    def add_model(self, model=None):
        self.model = model

    def __repr__(self):
        return "id: %s, type: %s, parent: %s;" % (self._id, self._type, self.parent)


class Tree(object):
    __slots__ = ["elements", "nums", "levels"]

    def __init__(self, nums=None, elements=None, levels=None):
        self.nums = nums
        self.elements = elements
        self.levels = levels

    def __getitem__(self, item):
        return getattr(self, item, 0)

    def get_node(self, _id=None):
        i = 0
        while i < self.nums and self.elements[i]["_id"] != _id:
            i = i + 1
        if i < self.nums:
            return self.elements[i]
        else:
            return None


class Level(object):
    __slots__ = ["num_level", "members"]

    def __init__(self, num_level=None, members=None):
        self.num_level = num_level  # number of levels
        self.members = members  # number of clients


if __name__ == '__main__':
    # unittest.main()
    num_node = 5
    list_node = []
    for i in range(num_node):
        list_node.append(Node(_id=i))
        print(list_node[i])