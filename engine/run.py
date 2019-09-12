import numpy as np


class Run():

    def __init__(self, database_frame,model):
        self.database_frame = database_frame
        self.input_size = database_frame.input_size
        self.output_size = database_frame.output_size
        self.train_set, self.test_set = self.get_train_test()
        self.model=model

    def train_model(self):


    def get_train_test(self):
        return self.database_frame.train_set.values, self.database_frame.test_set.values
