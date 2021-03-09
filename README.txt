A. interview-test-final sub-folder has been renamed to modules (to avoid hyphens) so I can access its content from other folders of parent directory.

B. I assumed all the calls must be made in interview_test_final.py. I therefore, did not make separate scripts for different modes of running and instead, used command line arguments (see below for details).

C. The main script is ./modules/interview_test_final.py for which the following options must be specified:

-m or --mode: train for performing training, run_experiment for performing gridsearchcv and test for making predictions and optionally validating the model.
-e or --eval: True or False. Whether to perform model evaluation or not using the reference data. This is used only if mode = 'test'. While making predictions in live mode, it must be turned to False.
-n or --model_name: The name of the classifer to be used. Currently, RandomForestClassifier and NN are supported. SVC has not yet been tested/tuned properly.

D. To run the program:

1. Please install all dependencies via pip install requirements.txt
2. From root directory of the project, run the script with the commands.

Example: python .\modules\interview_test_final.py --mode test --eval True --model_name RandomForestClassifier

E. Unit Test has been implemented for check_file_exists() routine of the FileDataLoader class.
- train_test_split.py used to split the data.
- arg_helpers.py contains functions to validate the command-line arguments.
- preprocessors.py contains implementation of various preprocessing stages of the ML pipeline.