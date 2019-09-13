import argparse
import logging
import datetime
import random
import numpy as np

import os

from engine.data_preprocessor import DataPreProcessor
from engine.run import Run
from engine.utils import get_model_type

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='regression model')

# here we are assuming that database path contains all the files necessary to process and it has already been divided
parser.add_argument('-p', '--database_path', type=str, default="../data_set",
                    help='path of directory containting datasets')

parser.add_argument('--result_path', type=str, default=os.path.expanduser('~') + '/results',
                    help='path of directory containing logs for runs')

parser.add_argument('--path_model', type=str, default='/Users/hosseinaboutalebi/results/2019-09-12_23:44:54.939063/model.pkl',
                    help='path of file containing saved model')

parser.add_argument('--load_mode', action="store_true",
                    help='Whether we want to load a model from given path or train a model from scartch (Default=False)')

parser.add_argument('--seed', type=int, default=442, metavar='N',
                    help='random seed (default: 42)')

parser.add_argument('-u', '--useless_col_name', nargs='*', help='column name containing useless values for '
                                                                'prediction that need to be deleted before regression', default="Unnamed: 0")

parser.add_argument('--model', type=str, default="XGBoost",
                    help='The model to use for regression. Current supported packages: XGBoost |'
                         'lightGBM | user_defined')

args = parser.parse_args()

#fixing seed
random.seed(args.seed)
np.random.seed(args.seed)


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

database_frame = DataPreProcessor(data_base_path=args.database_path, useless_labels=args.useless_col_name.split(","))
model=get_model_type(args.model)
run_program = Run(database_frame,model=model,file_path_save=file_path_results,file_path_load=args.path_model)
run_program.train_model(load_mode=args.load_mode)
