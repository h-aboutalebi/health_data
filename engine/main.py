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
                    help='path of directory containting logs for runs')

parser.add_argument('--seed', type=int, default=42, metavar='N',
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
run_program = Run(database_frame,model=model)
run_program.train_model()
