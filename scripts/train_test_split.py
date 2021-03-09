from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np 

if __name__ == "__main__":
    data_dir = "./data/"
    data_path = data_dir + "final_dataset.csv"

    # Read data
    df = pd.read_csv(data_path)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['user_id', 'great_customer_class']), df['great_customer_class'], test_size=0.2, random_state=0)
    
    # Save onto disk
    X_train.to_csv(data_dir + "X_train.csv", index=False)
    X_test.to_csv(data_dir + "X_test.csv", index=False)
    y_test.to_csv(data_dir + "y_test.csv", index=False)
    y_train.to_csv(data_dir + "y_train.csv", index=False)
