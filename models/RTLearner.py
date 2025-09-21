import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model = None

    def calc_root_node(self, data):
        """
        Randomly selects a feature
        """
        return np.random.randint(0, data.shape[1] - 1, 1)[0]

    def build_tree(self, data):
        """
        Trains the DT.
        """
        # If there is only one row, create a leaf node
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, data[0, -1], -1, -1]])
        # If all the y values are the same, create a leaf node
        if np.std(data[:, -1]) == 0:
            return np.array([[-1, data[0, -1], -1, -1]])
        else:
            i = self.calc_root_node(data)
            split_val = np.median(data[:, i])
            if len(data[data[:, i] > split_val]) == 0 or len(data[data[:, i] <= split_val]) == 0:
                left, right = np.array_split(data, 2)
                left_tree = self.build_tree(left)
                right_tree = self.build_tree(right)
            else:
                left_tree = self.build_tree(data[data[:, i] <= split_val])
                right_tree = self.build_tree(data[data[:, i] > split_val])
            root = np.array([[int(i), split_val, 1, int(left_tree.shape[0] + 1)]], dtype=object)
            return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # Combine X and Y data
        data = np.column_stack((data_x, data_y))
        self.model = self.build_tree(data)

    def traverse(self, data):
        """
        Takes a row of data as input and calculates Y using the DT model
        """
        row_index = 0

        while True:
            feature_index = int(self.model[row_index, 0])
            split_val = self.model[row_index, 1]
            left_tree = int(self.model[row_index, 2])
            right_tree = int(self.model[row_index, 3])

            # If at a leaf node, return the prediction
            if feature_index == -1 and left_tree == -1 and right_tree == -1:
                return split_val

            # Traverse left or right based on split val
            if data[feature_index] <= split_val:
                row_index += left_tree
            else:
                row_index += right_tree

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        return np.apply_along_axis(self.traverse, axis=1, arr=points)
