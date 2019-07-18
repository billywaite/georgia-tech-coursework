from util import entropy, information_gain, partition_classes
import numpy as np 
import ast


class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass
#
    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        #
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        # If only 1 y label, set that as label
        if len(set(y)) == 1:
            self.tree['label'] = y[0]
            return
        elif len(set(y)) == 0:
            self.tree['label'] = 0
            return

        # Create variables with initial values
        max_info_gain = 0
        
        # Initialize variables for loop
        x_len = len(X)
        row_len = len(X[0])
        
        # Find maximum information gain by looping through every possible partition
        for i in range(x_len):
            for j in range(row_len):
                
                test_split_val = X[i][j]
                xL, xR, yL, yR = partition_classes(X, y, j, test_split_val)
                info_gain = information_gain(y, [yL, yR])
                
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_attr = j
                    split_val = test_split_val
                    x_left, y_left, x_right, y_right = xL, yL, xR, yR
        
        # Create left and right trees
        self.tree['left'] = DecisionTree()
        self.tree['right'] = DecisionTree()
        
        # Train left and right trees
        self.tree['left'].learn(x_left, y_left)
        self.tree['right'].learn(x_right, y_right)
        
        # Store split attribute and split value in tree
        self.tree['split_attribute'] = split_attr
        self.tree['split_value'] = split_val
        pass

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        root_tree = self.tree
        
        while 'split_value' in root_tree:
            
            split_attribute = root_tree['split_attribute']
            split_value = root_tree['split_value']
            
            if type(split_value) == int:
                if record[split_attribute] <= split_value:
                    root_tree = root_tree['left'].tree
                else:
                    root_tree = root_tree['right'].tree
            else:
                if record[split_attribute] == split_value:
                    root_tree = root_tree['left'].tree
                else:
                    root_tree = root_tree['right'].tree
        return root_tree['label']
        pass
