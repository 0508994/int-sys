import pandas as pd

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

    def fit(self, label='label'):
        self.label = 'label'
        return

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


def compute_entropy(y_train):


def id3(X_train, y_train):




def main():
    # testing only
    n1 = Node('weather')

    nt1 = Node(label_val='cant')
    n2 = Node('temperature')
    n1.children['nice'] = n2
    n1.children['not_nice'] = nt1

    n2t = Node(label_val='cant')
    n3 = Node('wind')
    n2.children['cold'] = n2t
    n2.children['worm'] = n3

    n4 = Node(label_val='can')
    n3.children['windy'] = n4

    sample1 = {'weather': 'nice', 'temperature': 'worm', 'wind':'windy'}
    sample2 = {'weather': 'not_nice'}

    tree = SimpleTree(root=n1)
    print(tree.test_single(sample2))
    print(tree.root.attr_name)


if __name__ == '__main__':
    main()