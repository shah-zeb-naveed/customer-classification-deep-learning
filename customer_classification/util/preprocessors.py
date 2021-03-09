import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# TODO: Convert to sklearn Pipeline Column object to make the code more concise

def train_one_hot_encoder(df):
    "Input: df containing features, X_train"
    "Returns one-hot enoder for all categorical columns of df as a dictionary."
    df = df.copy()
    df = df.dropna()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder.fit(df.select_dtypes(object))

    return one_hot_encoder

def apply_one_hot_encoding(df, one_hot_encoder, feature_names=None, mode='test'):
    "Input: df containing features, X_train or X_test"
    "Returns one-hot encoded df based on one-hot-encoder provided."
    if mode == 'train':
        categorical_df = pd.DataFrame(one_hot_encoder.transform(df.select_dtypes(object)).toarray(), columns=one_hot_encoder.get_feature_names(feature_names))
    elif mode == 'test':
        categorical_df = pd.DataFrame(one_hot_encoder.transform(df.select_dtypes(object)).toarray())
    df = pd.concat([df.select_dtypes(include=np.number), categorical_df], axis=1)
    return df

def train_scalers(df):
    "Input: df containing features, X_train"
    "Returns StandardScalers for each column of df as a dictionary."
    scalers_dict = dict()
    # TODO: No need to loop. StandardScaler() can handle multiple columns
    for col in df.select_dtypes(include=np.number):
        scaler = StandardScaler()
        scalers_dict[col] = scaler.fit(np.array(df[col]).reshape(-1, 1))

    return scalers_dict

def apply_scaling(df, scalers_dict):
    "Input: df containing features, X_train or X_test"
    "Returns StandardScaled df based on scalers dictionary provided."
    # TODO: No need to loop. StandardScaler() can handle multiple columns
    df = df.copy()
    for col in df.select_dtypes(include=np.number):
        df[col] = scalers_dict[col].transform(np.array(df[col]).reshape(-1, 1))

    return df

def train_imputers(df):
    "Input: df containing features, X_train"
    "Returns imputers for each column of df as a dictionary."
    "For numeric columns: impute strategy = mean"
    "For non-numeric columns: impute strategy = most_frequent" 
    # TODO: No need to loop. SimpleImputer() can handle multiple columns
    imputers_dict = dict()
    for col in df:
        if df[col].dtype == float or df[col].dtype == int:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
        imputers_dict[col] = imputer.fit(np.array(df[col]).reshape(-1, 1))

    return imputers_dict

def apply_imputation(df, imputer_dict):
    "Input: df containing features, X_train or X_test"
    "Returns imputed df based on imputer dictionary provided."
    df = df.copy()
    for col in df:
        df[col] = imputer_dict[col].transform(np.array(df[col]).reshape(-1, 1))

    return df


def up_sample_minority_class(X_train, y_train, target):
    "Input: features and target df."
    "Returns features and target df with minority class over-samples."
    # TODO: Replace this with Modified SMOTE

    df_train = pd.concat([X_train, y_train], axis=1)
    count_class_0, count_class_1 = df_train[target].value_counts()
    df_class_0 = df_train[df_train[target] == 0]
    df_class_1_over = df_train[df_train[target] == 1].sample(count_class_0, replace=True)
    df_train_balanced = pd.concat([df_class_0, df_class_1_over], axis=0)
    return df_train_balanced.drop(columns=[target]), df_train_balanced[target]