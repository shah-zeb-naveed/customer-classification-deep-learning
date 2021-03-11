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
    parser = argparse.ArgumentParser(description='CLI for interview_test_final.py')
    parser.add_argument('-m', '--mode', help='The running mode: train, run_experiment or test', type = validate_mode, required=True)
    parser.add_argument('-e', '--eval', help='Model evaluation: True or False (ignored while training or experimentation). For live predictions, it must be turned to False.', type = validate_eval, required=True)
    parser.add_argument('-n', '--model_name', help='Model name: RandomForestClassifier or NN', type = validate_model_name, required=True)
    args = parser.parse_args()
    return args

def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # change directory so we can use relative filepaths
    os.chdir(rundir + project_name)

def load_config():
    run_configuration_file = './resources/interview-test-final.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    mode = args.mode
    eval = args.eval

    # Specify paths
    models_dir = './output/models/'
    X_train_path = './data/X_train.csv'
    X_test_path = './data/X_test.csv'
    y_train_path = './data/y_train.csv'
    y_test_path = './data/y_test.csv'

    # Initialize logging
    logging.basicConfig(format="%(asctime)s; %(levelname)s; %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    logging.info('Setting project environment')
    #sort_file_paths(project_name='interview-test-final') # don't need this since relative paths can still work
    run_configuration = load_config()

    # TODO: Add re-training mode for warm-start

    if mode == 'train':
        logging.info('Loading training data')
        X_train = FileDataLoader(X_train_path).load_data()  
        y_train = FileDataLoader(y_train_path).load_data()

        logging.info('Training Model')
        cvp = CustomerValuePredictor(model_name)
        cvp.train(X_train, y_train)

        logging.info('Saving Model')
        cvp.save_model(models_dir)
    
    # Perform grid search on hypermaraters defined in the corresponding json files.
    if mode == 'run_experiment':
        logging.info('Loading training data')
        X_train = FileDataLoader(X_train_path).load_data()  
        y_train = FileDataLoader(y_train_path).load_data()

        logging.info('Running Experiment')
        cvp = CustomerValuePredictor(model_name)
        cvp.run_experiment(X_train, y_train)

    # Prediction mode. For live predictions, eval must be False (as there would be no reference data to test on).
    elif mode == 'test':
        logging.info('Loading Models')
        cvp = CustomerValuePredictor(model_name)
        cvp.load_model(models_dir)

        logging.info('Loading Test Data')
        X_test = FileDataLoader(X_test_path).load_data()

        logging.info('Preprocessing Data')
        X_test = cvp.preprocess_data(X_test, mode='test')

        logging.info('Starting Predictions')
        y_preds = cvp.predict(X_test)
        
        if eval == 'True':
            logging.info('Performing Evaluation')
            y_true = FileDataLoader(y_test_path).load_data()
            cvp.perform_evaluation(X_test, y_true, y_preds) 

    logging.info('Completed program')