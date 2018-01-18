import pandas as pd
import math

class Node:
    def __init__(self, attr_name=None, label_val=None):
        # atribute that node is testing
        self.attr_name = attr_name

        # only leaf nodes have this value set
        self.label_val = label_val 

        # key = test result, value = pointer to node
        self.children = {}

class SimpleTree:
    def __init__(self, root=None) :
        self.root = root

    def fit(self, df, label='label'):
        self.label = 'label'
        self.root = id3(df, label)

    def test_single(self, sample):
        def test(node):
            if node.label_val != None:
                return node.label_val
            else:
                # value of the testing attribute
                part_res = sample[node.attr_name]
                # get the next node (according to the part_res)
                node = node.children[part_res]
                # recurse
                return test(node)
        return test(self.root)


# pandas only
def compute_entropy(df, label):
    nr_entries = len(df.index)

    # number of appearances
    classes_info = df[label].value_counts()
    nr_classes = len(classes_info)
    entropy = 0
    for c in classes_info:
        pc = float(c / nr_entries) 
        if nr_classes == 1 or pc == 1: continue
        entropy += -pc * math.log(pc, nr_classes)

    return entropy

def compute_attr_fitness(df, entropy, attr_name, label):
    nr_entries = len(df.index)

    # number of appearances
    attr_info = df[attr_name].value_counts()
    # values that go with attr_info list
    attr_info_values = attr_info.index.tolist()

    sum = 0
    for ai in attr_info_values:
        if ai == label: continue
        subseti = df.where(df[attr_name] == ai)
        sum +=  float(attr_info[ai] / nr_entries) * compute_entropy(subseti, label)

    return entropy - sum

def id3(df, label):
    #df = df.reset_index(drop=False)
    node = Node() # this is the return value
    entropy = compute_entropy(df, label)
    if entropy == 0:
    # set node as leaf and return it
        #node.label_val = df.get_value(0, label)
        node.label_val = df.iloc[0][label]
        return node
    else:
        # create split cond and children nodes
        # select split atribute
        attr_list = list(df)
        fits = {}
        for a in attr_list:
            if a == label: continue
            fits[a] = compute_attr_fitness(df, entropy, a, label)

        fittest = max(fits, key=fits.get)

        attr_values = df[fittest].value_counts().index.tolist()
        node.attr_name = fittest

        for v in attr_values:
            node.children[v] = id3(df.where(df[fittest] == v) \
                                     .drop([fittest], axis=1) \
                                     .dropna(),\
                                   label)

        return node




if __name__ == '__main__':
    #  d = {'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5]}
    #  df = pd.DataFrame(data=d)
    #  print(compute_entropy(df, 'col1'))

    df = pd.read_csv('bas.txt')
    tree = SimpleTree()
    tree.fit(df, 'klasa')

    sample = {'izgled_vremena':'sunčano', 'temperatura':'toplo', 'vetar':'slab', 'vlažnost': 'visoka'}
    print(tree.test_single(sample))