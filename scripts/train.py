# Run from root directory
# Example: python .\modules\interview_test_final.py --mode test --eval True --model_name RandomForestClassifier

import logging
import json
from collections import namedtuple
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from customer_classification.util.data_loaders import FileDataLoader
from customer_classification.predictors import CustomerValuePredictor
from customer_classification.util.arg_helpers import *
import argparse

def parse_args():
    # TODO: Improve the interface
    parser = argparse.ArgumentParser(description='CLI for validate.py')
    parser.add_argument('-n', '--model_name', help='Model name: RandomForestClassifier or NN', type = validate_model_name, required=True)
    args = parser.parse_args()
    return args

def load_config():
    run_configuration_file = './configs/project_configs.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration


if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name

    # Specify paths
    models_dir = './output/models/'
    X_train_path = './data/input/X_train.csv'
    y_train_path = './data/input/y_train.csv'

    # Initialize logging
    logging.basicConfig(format="%(asctime)s; %(levelname)s; %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    logging.info('Setting project environment')
    run_configuration = load_config()

    logging.info('Loading training data')
    X_train = FileDataLoader(X_train_path).load_data()  
    y_train = FileDataLoader(y_train_path).load_data()

    logging.info('Training Model')
    cvp = CustomerValuePredictor(model_name)
    cvp.train(X_train, y_train)

    logging.info('Saving Model')
    cvp.save_model(models_dir)

    logging.info('Completed program')