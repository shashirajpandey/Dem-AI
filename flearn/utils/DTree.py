
import numpy as np
import tensorflow as tf
from tqdm import trange

class Node(object):
    __slots__ = ["_id", "_type", "parent", "data", "gmodel","grad", "childs", "level", "numb_clients", "in_clients"]

    def __init__(self, _id=None, _type="Group", parent=None, data=None, gmodel=None, grad= None, childs=None, level=None, numb_clients= None, in_clients=None ):

        self._type = _type
        self._id = _id
        self.data = data or []
        self.gmodel = gmodel or []
        self.grad= grad or []
        self.parent = parent or "Empty"
        self.childs = childs or []
        self.level = level or 0
        self.numb_clients = numb_clients or 1
        self.in_clients = in_clients or []

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
        # print("Checking at id:", self._id)
        # print("Parent",self.parent)
        if (self._type.upper() == "CLIENT"):
            return self.parent.get_hierrachical_info()
        else:
            if(self.parent != "Empty"):
                parent_md = self.parent.get_hierrachical_info()
                # print(parent_md)
                return (self.gmodel[0]/self.numb_clients + parent_md[0], self.gmodel[1]/self.numb_clients + parent_md[1])
            elif(self.parent == "Empty"):  #root node
                return (self.gmodel[0]/self.numb_clients, self.gmodel[1]/self.numb_clients)

    def get_hierrachical_info1(self): # with defactor later for normalize the sum
        # print("Checking at id:", self._id)
        # print("Parent",self.parent)
        if (self._type.upper() == "CLIENT"):
            return self.parent.get_hierrachical_info1()
        else:
            if(self.parent != "Empty"):
                parent_md, parent_normalize_term = self.parent.get_hierrachical_info1()
                normalize_term = 1./self.numb_clients + parent_normalize_term
                # print(parent_md)
                return ((self.gmodel[0]/self.numb_clients + parent_md[0], self.gmodel[1]/self.numb_clients + parent_md[1]), normalize_term)
            elif(self.parent == "Empty"):  #root node
                return ((self.gmodel[0]/self.numb_clients, self.gmodel[1]/self.numb_clients), 1./self.numb_clients)


    def count_clients(self):
        if self._type.upper()=="CLIENT":
            return 1
        elif self.level ==1:
            return len(self.childs)
        else:
            counts = 0
            for c in self.childs:
                counts += c.count_clients()
            return counts

    def collect_clients(self):
        # print("Node:",self._id)
        if self._type.upper() == "CLIENT":
            return None
        elif self.level==1:
            return self.childs
        else:
            rs = []
            for c in self.childs:
                tmp = c.collect_clients()
                if(tmp):  #Not None => subgroup
                    rs += tmp
                else:  #None =>c is a leave
                    rs.append(c)
            return rs

    def print_structure(self):
        if(self._type.upper() !="CLIENT"):
            print(self)
        if self.childs:
            for c in self.childs:
                c.print_structure()

    def __repr__(self):
        return "id: %s, lv: %s, type: %s, parent: %s;" % (self._id, self.level, self._type, self.parent)


# class Tree(object):
#     __slots__ = ["elements", "nums", "levels"]
#
#     def __init__(self, nums=None, elements=None, levels=None):
#         self.nums = nums
#         self.elements = elements
#         self.levels = levels
#
#     def __getitem__(self, item):
#         return getattr(self, item, 0)
#
#     def get_node(self, _id=None):
#         i = 0
#         while i < self.nums and self.elements[i]["_id"] != _id:
#             i = i + 1
#         if i < self.nums:
#             return self.elements[i]
#         else:
#             return None
#
#
# class Level(object):
#     __slots__ = ["num_level", "members"]
#
#     def __init__(self, num_level=None, members=None):
#         self.num_level = num_level  # number of levels
#         self.members = members  # number of clients


def t_generalized_update(node,mode="hard"):
        model_shape = (1,1)
        gamma=1.0
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            node.numb_clients = node.count_clients()
            # print(self.Weight_dimension)
            rs_w = np.zeros(model_shape[0])
            rs_b = np.zeros(model_shape[1])
            for child in childs:
                gmd = t_generalized_update(child,mode)
                # print("shape=",gmd.shape)
                rs_w += gmd[0] * child.numb_clients #weight
                rs_b += gmd[1] * child.numb_clients #bias
            avg_w = 1.0 * rs_w /node.numb_clients
            avg_b = 1.0 * rs_b /node.numb_clients
            if(mode=="hard"):
                node.gmodel = (avg_w,avg_b)
            else:
                node.gmodel = ((1-gamma)*node.gmodel[0] + gamma * avg_w,
                               (1 - gamma) * node.gmodel[1] + gamma * avg_b )# (weight,bias)
            return node.gmodel
        else: #Client
            md = node.gmodel
            # print(md[0].shape,"--",md[1].shape)
            # return np.concatenate( (md[0].flatten(),md[1]), axis=0 )
            return md

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
    Root = Node(_id=0, parent="Empty",gmodel=(5,5), numb_clients= 4 , _type="Group" )
    Group1 = Node(_id=1, parent=Root,gmodel=(2,2), numb_clients=2, _type="Group")
    Group2 = Node(_id=2, parent=Root, gmodel=(3, 3), numb_clients=2, _type="Group")
    Client1 =  Node(_id=3, parent=Group1,gmodel=(3.0,3.0), _type="Client")
    Client2 = Node(_id=4, parent=Group1, gmodel=(6, 6), _type="Client")
    Client3 = Node(_id=5, parent=Group2, gmodel=(3, 3), _type="Client")
    Client4 = Node(_id=6, parent=Group2, gmodel=(4, 4), _type="Client")
    Group1.childs = [Client1,Client2]
    Group2.childs = [Client3,Client4 ]
    # Root.childs = [Group1, Client2]
    # print(Group2.get_hierrachical_info())
    # print(Group1.get_hierrachical_info())
    # print(Client1.get_hierrachical_info())
    # print(Client2.get_hierrachical_info())
    # test_generalized_update(Root)
    # print(Root.gmodel)
    # print(Group1.gmodel)
    Root.childs = [Group1, Group2]
    print(Group1.get_hierrachical_info1())
    print(Group2.get_hierrachical_info1())
    print("Client")
    c1 = Client1.get_hierrachical_info1()
    print(c1 , (c1 [0][0]/c1[1], c1 [0][1]/c1[1]))
    c2 = Client2.get_hierrachical_info1()
    print(c2, (c2[0][0] / c2[1], c2[0][1] / c2[1]))
    c3 = Client3.get_hierrachical_info1()
    print(c3, (c3[0][0] / c3[1], c3[0][1] / c3[1]))
    c4 = Client4.get_hierrachical_info1()
    print(c4, (c4[0][0] / c4[1], c4[0][1] / c4[1]))


