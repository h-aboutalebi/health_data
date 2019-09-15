import numpy as np

try:
    import cPickle as pkl
except:
    import pickle as pkl
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)
import time


# Assumption: first column is taget in our dataset
class Run():

    def __init__(self, database_frame, model, file_path_save, file_path_load):
        self.data_base_list = [np.array(data.values) for data in database_frame.data_base_list]
        self.files_name = [f.split("/")[-1] for f in database_frame.files]
        self.model = model
        self.file_pth_save_model = file_path_save + "/model.pkl"  # This is for saving model
        self.file_pth_load_model = file_path_load

    def train_model(self, load_mode):
        if (load_mode is False):
            logger.info("Building model ...")
            for i in range(len(self.data_base_list)):
                start_time = time.time()
                x_test = self.data_base_list[i][:, 1:]
                y_test = np.array(self.data_base_list[i][:, 1]).reshape(-1)
                train_list = self.get_train_list(i)
                file_names_train = self.get_file_names_train(i)
                x_train = train_list[:, 1:]  # first column is taget
                y_train = train_list[:, 1]
                self.model.fit(x_train, y_train)
                y_pred = np.array(self.model.predict(x_test))
                MSE = mean_squared_error(y_test, y_pred)
                logger.info("*" * 15)
                logger.info("Results: For Test Set: {} .With Train Set of {}:".format(self.files_name[i], file_names_train))
                logger.info("Mean Squared Error: {}| Process Time: {}".format(MSE, time.time() - start_time))
            self.save_model()
            logger.info("Model has been saved in following path:\n {}".format(self.file_pth_save_model))
        else:
            logger.info("Loading saved model ...")

            # Note: In the loading mode there is no training
            self.model = self.load_model()
            for i in range(len(self.data_base_list)):
                start_time = time.time()
                x_test = self.data_base_list[i][:, 1:]
                y_test = np.array(self.data_base_list[i][:, 1]).reshape(-1)
                y_pred = np.array(self.model.predict(x_test))
                MSE = mean_squared_error(y_test, y_pred)
                logger.info("*" * 15)
                logger.info("Results: For Test Set: {}:".format(self.files_name[i]))
                logger.info("Mean Squared Error: {}| Process Time: {}".format(MSE, time.time() - start_time))

    def save_model(self):
        with open(self.file_pth_save_model, "wb") as output_file:
            pkl.dump(self.model, output_file)

    def load_model(self):
        with open(self.file_pth_load_model, "rb") as input_file:
            return pkl.load(input_file)

    def get_train_list(self, i):
        if (i == 0):
            train_list = self.data_base_list[1:]
        elif (i < len(self.data_base_list) - 1):
            train_list = self.data_base_list[:i] + self.data_base_list[i + 1:]
        else:
            train_list = self.data_base_list[:len(self.data_base_list) - 1]
        return np.concatenate(train_list, axis=0)

    def get_file_names_train(self, i):
        if (i == 0):
            files_name = self.files_name[1:]
        elif (i < len(self.files_name) - 1):
            files_name = self.files_name[:i] + self.files_name[i + 1:]
        else:
            files_name = self.files_name[:len(self.files_name) - 1]
        return files_name
