import numpy as np


class Node(object):
    __slots__ = ["_id", "_type", "parent", "data", "model", "childs", "level"]

    def __init__(self, _id=None, _type="Group", parent=None, data=None, model=None, childs=None, level=None ):

        self._type = _type
        self._id = _id
        self.data = data or []
        self.model = model or []
        self.parent = parent or "Empty"
        self.childs = childs or []
        self.level = level or 0

    def __getitem__(self, item):
        return getattr(self, item, 0)

    def add_data(self, data=None):
        self.data = data

    def del_data(self):
        self.data = None

    def add_model(self, model=None):
        self.model = model

    def add_parent(self, parent=None):
        self.parent = parent

    def get_clients(self):
        if self._type.upper() == "GROUP":
            return self.childs
        else:
            return False

    def count_clients(self):
        counts = 0
        if self._type=="Client":
            return 1
        elif self.level ==1:
            return len(self.childs)
        else:
            for c in self.childs:
                counts += c.count_clients()
        return counts




    def __repr__(self):
        return "id: %s, type: %s, parent: %s;\n" % (self._id, self._type, self.parent)


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
    print("a")




