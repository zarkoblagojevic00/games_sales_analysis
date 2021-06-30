import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from sklearn.model_selection import LeaveOneOut


class AbstractTrainTestSplitter(ABC):
    def __init__(self, x, y):
        self.x, self.y = x, y

    @abstractmethod
    # has to return a list of train test splits, so tester can iterate over it
    def train_test_split(self):
        pass


class BasicTrainTestSplitter(AbstractTrainTestSplitter):

    def __init__(self, x, y):
        super().__init__(x, y)

    def train_test_split(self):
        return [np.array(train_test_split(self.x, self.y, test_size=.3, random_state=0), dtype='object')]


class LeaveOneOutTrainTestSplitter(AbstractTrainTestSplitter):

    def __init__(self, x, y):
        super().__init__(x, y)

    def train_test_split(self):
        leave_one_out = LeaveOneOut()
        train_test_splits = []
        for train_index, test_index in leave_one_out.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            split = np.array((x_train, x_test, y_train, y_test), dtype='object')
            train_test_splits.append(split)

        return train_test_splits




