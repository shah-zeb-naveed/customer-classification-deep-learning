def validate_mode(mode):
    if mode == 'train' or mode == 'test' or mode == 'run_experiment':
        return mode 
    else:
        raise ValueError

def validate_eval(eval):
    if eval == 'True' or eval == 'False':
        return eval 
    else:
        raise ValueError

def validate_model_name(model_name):
    if model_name == 'RandomForestClassifier' or  model_name == 'NN':
        return model_name 
    else:
        raise ValueError