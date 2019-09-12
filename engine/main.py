

import argparse

import numpy as np

import pandas as pd
import logging
import datetime


import os

from engine.data_preprocessor import DataPreProcessor
from engine.run import Run

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='regression model')

#here we are assuming that database path contains all the files necessary to process and it has already been divided
parser.add_argument('-p', '--database_path', type=str, default="data_set/13_record_diast.csv",
                    help='path of directory containting datasets')

parser.add_argument('--result_path', type=str, default=os.path.expanduser('~') + '/results',
                    help='path of directory containting logs for runs')

parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
#we use cross validation but initially hold out the test set
parser.add_argument('--test_ratio', type=int, default=0.2,
                    help='Test set ratio for final evaluation')

parser.add_argument('--target_col_name', type=int, default="target", metavar='N',
                    help='column name containing target values for prediction')

parser.add_argument('--model', type=str, default="XGBoost",
                    help='The model to use for regression. Current supported packages: XGBoost |'
                         'lightGBM | user_defined')

args = parser.parse_args()

file_path_results = args.result_path + "/" + str(datetime.datetime.now()).replace(" ", "_")
if not os.path.exists(args.result_path):
    os.mkdir(args.result_path)
os.mkdir(file_path_results)

logging.basicConfig(level=logging.DEBUG, filename=file_path_results + "/log.txt")
logging.getLogger().addHandler(logging.StreamHandler())

header = "===================== Experiment configuration ========================"
logger.info(header)
args_keys = list(vars(args).keys())
args_keys.sort()
max_k = len(max(args_keys, key=lambda x: len(x)))
for k in args_keys:
    s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
    logger.info(s + ' ' * max((len(header) - len(s), 0)))
logger.info("=" * len(header))

database_processor=DataPreProcessor(data_base_path=args.database_path, y_labels=args.target_col_name, test_ratio=args.test_ratio)
run_program=Run(database_processor)




