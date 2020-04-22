
import numpy as np
import tensorflow as tf
from tqdm import trange

class Node(object):
    __slots__ = ["_id", "_type", "parent", "data", "gmodel","ggrad", "childs", "level", "numb_clients"]

    def __init__(self, _id=None, _type="Group", parent=None, data=None, gmodel=None, ggrad= None, childs=None, level=None ):

        self._type = _type
        self._id = _id
        self.data = data or []
        self.gmodel = gmodel or []
        self.ggrad= ggrad or []
        self.parent = parent or "Empty"
        self.childs = childs or []
        self.level = level or 0
        self.numb_clients = 1

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

    def get_hierrachical_info(self):
        print("Checking at id:", self._id)
        # print("Parent",self.parent)
        if (self._type.upper() == "CLIENT"):
            return self.parent.get_hierrachical_info()
        else:
            if(self.parent != "Empty"):
                parent_md = self.parent.get_hierrachical_info()
                return (self.gmodel[0]/self.numb_clients + parent_md[0], self.gmodel[1]/self.numb_clients + parent_md[1])
            elif(self.parent == "Empty"):  #root node
                return (self.gmodel[0]/self.numb_clients, self.gmodel[1]/self.numb_clients)

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

    #
    # features = tf.placeholder(tf.float32, shape=[None, 784], name='features')
    # labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
    # input_layer = tf.reshape(features, [-1, 28, 28, 1])
    # conv1 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=32,
    #     kernel_size=[5, 5],
    #     padding="same",
    #     activation=tf.nn.relu)
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)



