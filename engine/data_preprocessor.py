import pandas as pd
import numpy as np


class AbstractDataPreProcessor():

    def __init__(self, data_base_path, y_labels, test_ratio):
        self.test_ratio = test_ratio
        self._data_base = self.extract_database(data_base_path)
        self.y_labels = self.extract_y_labels(y_labels)
        self.x_labels = self.extract_x_labels(y_labels)

    def extract_database(self, data_base_path):
        raise NotImplementedError

    def extract_x_labels(self, *args):
        raise NotImplementedError

    def extract_y_labels(self, *args):
        raise NotImplementedError

class DataPreProcessor(AbstractDataPreProcessor):

    def __init__(self, data_base_path, y_labels,useless_labels, test_ratio):
        super(DataPreProcessor, self).__init__(data_base_path, y_labels, test_ratio)
        self.remove_useless_labels(self.x_labels, useless_labels)

        # we put target_labels to the end of dataframe
        self.data_set = pd.concat([self.x_labels, self.y_labels], axis=1)
        self.test_set, self.train_set = self.extract_test_train()
        self.input_size = len(self.data_set.columns) - len(y_labels)
        self.output_size = len(y_labels)

    def extract_database(self, data_base_path):
        return pd.read_csv(data_base_path)

    def extract_y_labels(self, y_labels):
        return self._data_base[y_labels]

    def extract_x_labels(self, y_labels):
        x_labels = [item for item in self._data_base.columns.values if item not in y_labels]
        return self._data_base[x_labels]

    def extract_test_train(self):
        suffeled_dataset = self.data_set.sample(frac=1)
        size_of_test_set = int(np.ceil(self.test_ratio * len(self.data_set)))
        test_set = suffeled_dataset[:size_of_test_set]
        train_set = suffeled_dataset[size_of_test_set:]
        return test_set, train_set

    def remove_useless_labels(self, dataset, useless_labels):
        if len(useless_labels) == 0:
            return
        for label in useless_labels:
            del dataset[label]
