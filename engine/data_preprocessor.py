import pandas as pd
import numpy as np
import glob
import logging

logger = logging.getLogger(__name__)

class AbstractDataPreProcessor():

    def __init__(self, data_base_path):
        self.files = [f for f in glob.glob(data_base_path + "**/*.csv", recursive=True)]
        self.data_base_list = [self.extract_database(data_base_path) for data_base_path in self.files]

    def extract_database(self, data_base_path):
        raise NotImplementedError


class DataPreProcessor(AbstractDataPreProcessor):

    def __init__(self, data_base_path, useless_labels):
        super(DataPreProcessor, self).__init__(data_base_path)
        self.remove_useless_labels(self.data_base_list,useless_labels)
        logger.info("dataset extraction is completed.")

    def extract_database(self, data_base_path):
        return pd.read_csv(data_base_path)

    def remove_useless_labels(self, dataset, useless_labels):
        if len(useless_labels) == 0:
            return
        for label in useless_labels:
            for dataset in self.data_base_list:
                del dataset[label]
