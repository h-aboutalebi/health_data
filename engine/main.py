

import argparse

import numpy as np

import pandas as pd

parser = argparse.ArgumentParser(description='XGBoost regression model')

parser.add_argument('-p', '--database_path', type=str, default="data_base.csv",
                    help='Validation set ratio')