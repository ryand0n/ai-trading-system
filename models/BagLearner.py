import numpy as np
from .RTLearner import RTLearner

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """
        Constructor method
        """
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.column_stack((data_x, data_y))
        for learner in self.learners:
            random_indices = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
            bag = data[random_indices]
            learner.add_evidence(bag[:, :-1], bag[:, -1])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.column_stack([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=1)
