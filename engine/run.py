class Run():

    def __init__(self, database_frame):
        self.database_frame = database_frame
        self.
        self.train_set,self.test_set=self.get_train_test()

    def get_train_test(self):
        return self.database_frame.train_set.values, self.database_frame.test_set.values

