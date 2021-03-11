import logging
from abc import ABC, abstractmethod
from customer_classification.util.preprocessors import *
import pandas as pd, numpy as np 
import matplotlib.pyplot as plt, seaborn as sns 
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 8)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
import pickle
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

class Model(ABC):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')

    @abstractmethod
    def train(self):
        logging.info('Training model')

    @abstractmethod
    def predict(self):
        logging.info('Doing predictions')


class CustomerValuePredictor():
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        # Check for supported model types
        if self.model_name != "RandomForestClassifier" and self.model_name != "NN":
            raise Exception('Only RandomForestClassifier and NN are supported.')

    def preprocess_data(self, df, mode='train'):
        "Input: a df containg features, X_train or X_test"
        "Returns: processed data (imputation, scaling, one-hot encoding)."
        
        df = df.copy()
        df = apply_imputation(df, self.imputers_dict)
        df = apply_scaling(df, self.scalers_dict)

        # TODO: Remove categorical_features_names. No use now.
        if mode == 'train':
            df = apply_one_hot_encoding(df, self.one_hot_encoder, self.categorical_feature_names, mode)
        elif mode == 'test':
            df = apply_one_hot_encoding(df, self.one_hot_encoder, mode=mode)

        return df

    def train(self, X_train, y_train):
        "Input: Training data dfs."
        "Returns the trained model/s on the disk."

        # Fit transformations
        logging.info('Fitting transformations')
        self.imputers_dict = train_imputers(X_train)
        self.scalers_dict = train_scalers(X_train)
        self.one_hot_encoder = train_one_hot_encoder(X_train)

        # Save categorical feature names to label one-hot-encoder transformed output
        self.categorical_feature_names = X_train.select_dtypes(object).columns

        logging.info('Preprocessing Data')
        X_train = self.preprocess_data(X_train)

        logging.info('Upsampling Minority Class')
        X_train, y_train = up_sample_minority_class(X_train, y_train, "great_customer_class")

        logging.info('Fitting the model')

        if self.model_name == 'RandomForestClassifier':
            # Read params
            f = open("./configs/rf_params.json")
            params = json.load(f)
            f.close()

            # Fit the model
            rf_model = RandomForestClassifier(**params, random_state=0)
            rf_model.fit(X_train, y_train)
            self.model = rf_model

            # Plot feature importance
            logging.info('Plotting Feature Importance')
            features_importance_plot = sns.barplot(x = rf_model.feature_importances_, y=X_train.columns).set_title("Feature Importance in Random Forest - Training")
            features_importance_plot.get_figure().savefig("./output/graphs/rf_features_importance_plot_training.png")

        elif self.model_name == 'SVM':
            # Read params
            f = open("./configs/svm_params.json")
            params = json.load(f)
            f.close()

            # Fit Model
            svc_model = LinearSVC(**params, max_iter=10000, random_state=0)
            svc_model.fit(X_train, y_train)
            self.model = svc_model


        elif self.model_name == 'NN':
            # Read params
            f = open("./configs/nn_params.json")
            params = json.load(f)
            f.close()

            # Specify callbacks
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            mc = ModelCheckpoint('./output/models/nn_checkpoint', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            # TODO: Enable modelcheckpoints
            callbacks = [es]

            # Split into train and validation sets to be able to use callbacks based on validation loss
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

            # Get network architecture
            nn_model = self.get_neural_network_architecture(input_dim=X_train.shape[1])

            # Fit model
            model_fitting = nn_model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), callbacks=callbacks)

            # Plot training history
            plt.plot(model_fitting.history['accuracy'])
            plt.plot(model_fitting.history['loss'])
            plt.plot(model_fitting.history['val_accuracy'])
            plt.plot(model_fitting.history['val_loss'])
            plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'])
            plt.title("Model Fitting History - Training")
            plt.savefig("./output/graphs/nn_training_history.png")

            self.model = nn_model
            

    def run_experiment(self, X_train, y_train):
        "Input: Training dfs"
        "Performs grid search over the hyperparameter grid in the corresponding"
        "JSON files. Saves the best parameters on the disk."
        # TODO: Output more results of experimentation

        # Fit transformations
        logging.info('Fitting transformations')
        self.imputers_dict = train_imputers(X_train)
        self.scalers_dict = train_scalers(X_train)
        self.one_hot_encoder = train_one_hot_encoder(X_train)

        # Save categorical feature names to label one-hot-encoder transformed output
        self.categorical_feature_names = X_train.select_dtypes(object).columns

        logging.info('Preprocessing Data')
        X_train = self.preprocess_data(X_train)

        logging.info('Upsampling Minority Class')
        X_train, y_train = up_sample_minority_class(X_train, y_train, "great_customer_class")

        # Grid Search Model Params
        # TODO: Import from JSON
        logging.info('Performing GridSearchCV')

        if self.model_name == 'RandomForestClassifier':
            # Read params
            f = open("./configs/rf_gridsearch_params.json")
            params = json.load(f)
            f.close()

            #TODO: Add feature selection in gridsearch

            # Gridsearch
            grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=0), params, scoring='f1', verbose=1, cv=5)
            grid_search_cv.fit(X_train, np.array(y_train).ravel())

            with open('./configs/rf_params.json', 'w') as f:
                json.dump(grid_search_cv.best_params_, f)

        elif self.model_name == 'SVM':
            # Read params
            f = open("./configs/svm_gridsearch_params.json")
            params = json.load(f)
            f.close()

            #TODO: Add feature selection in gridsearch

            # Gridsearch
            grid_search_cv = GridSearchCV(LinearSVC(max_iter=10000, random_state=0), params, scoring='f1', verbose=1, cv=5)
            grid_search_cv.fit(X_train, np.array(y_train).ravel())

            with open('./configs/svm_params.json', 'w') as f:
                json.dump(grid_search_cv.best_params_, f)

        elif self.model_name == 'NN':

            # Read params
            f = open("./configs/nn_gridsearch_params.json")
            params = json.load(f)
            f.close()
            
            #TODO: Add callbacks while performing gridsearch
            #TODO: Add feature selection in gridsearch

            # Gridsearch
            nn_classifier = KerasClassifier(build_fn=self.get_neural_network_architecture, input_dim=X_train.shape[1], verbose=1)
            grid_search_cv = GridSearchCV(nn_classifier, params, scoring = 'f1', verbose=1, cv=5)
            grid_search_cv.fit(X_train, y_train)
            
            with open('./configs/nn_params.json', 'w') as f:
                json.dump(grid_search_cv.best_params_, f)

    def get_neural_network_architecture(self, input_dim):
        "Input: number of features"
        "Returns the compiled architecture of neural network."
        # TODO: Experiment with different architectures
        nn_model = Sequential()
        nn_model.add(Dense(12, input_dim=input_dim, activation='relu'))
        nn_model.add(Dense(8, activation='relu'))
        nn_model.add(Dense(1, activation='sigmoid'))
        nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return nn_model

    def save_model(self, models_dir):
        "Saves the model on disk."
        if self.model_name != 'NN':
            model_path = models_dir + self.model_name + ".pickle"
            pickle.dump(self.model, open(model_path, 'wb'))
        else:
            self.model.save(models_dir + 'nn')

        imputers_path = models_dir + "imputers" + ".pickle"
        pickle.dump(self.imputers_dict, open(imputers_path, 'wb'))

        scalers_path = models_dir + "scalers" + ".pickle"
        pickle.dump(self.scalers_dict, open(scalers_path, 'wb'))
    
        one_hot_encoder_path = models_dir + "one_hot_encoder" + ".pickle"
        pickle.dump(self.one_hot_encoder, open(one_hot_encoder_path, 'wb'))
    
    def load_model(self, models_dir):
        "Loads the model from disk."
        if self.model_name != 'NN':
            self.model = pickle.load(open(models_dir + self.model_name + ".pickle",'rb'))
        else:
            self.model = load_model(models_dir + 'nn')

        self.imputers_dict = pickle.load(open(models_dir + "imputers" + ".pickle",'rb'))
        self.scalers_dict = pickle.load(open(models_dir + "scalers" + ".pickle",'rb'))
        self.one_hot_encoder = pickle.load(open(models_dir + "one_hot_encoder" + ".pickle",'rb'))

    def predict(self, X_test):
        "Performs predictions."
        X_test = X_test.copy()

        if self.model_name != "NN":
            X_test['predictions'] = self.model.predict(X_test)
        else:
            X_test['predictions'] = (self.model.predict(X_test) > 0.5).astype("int32")

        X_test.to_csv("./output/predictions/" + self.model_name + "_predictions.csv", index=False)
        return X_test['predictions']

    def get_classification_report(self, y_true, y_preds):
        "Saves classification report on disk."
        cr = classification_report(y_true, y_preds, output_dict=True)
        cr_df = pd.DataFrame(cr).transpose().to_csv("./output/evaluation/" + self.model_name + "_classification_report.csv")
        return cr_df

    def get_confusion_matrix(self, y_true, y_preds):
        "Saves confusion matrix on disk."
        cm = confusion_matrix(y_true, y_preds)
        heat_map = sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title("Confusion Matrix - " + self.model_name, fontsize=18)
        heat_map.get_figure().savefig("./output/evaluation/" +  self.model_name + "_confusion_matrix.png")

    def get_roc_curve(self, X_test, y_true):
        "Saves ROC Curve plot on disk."
        roc = plot_roc_curve(self.model, X_test, y_true)
        plt.title("ROC Curve - " + self.model_name, fontsize=18)
        plt.savefig("./output/evaluation/" +  self.model_name + "_roc_curve.png")

    def perform_evaluation(self, X_test, y_true, y_preds):
        "Performs all steps of evaluation."
        self.get_classification_report(y_true, y_preds)
        self.get_confusion_matrix(y_true, y_preds)
        
        # TODO: Add support for NN to plot roc curves
        if self.model_name != "NN":
            self.get_roc_curve(X_test, y_true)