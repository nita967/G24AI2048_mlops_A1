# misc.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import joblib
import os

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
                     'DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def preprocess(df, target_col='MEDV', test_size=0.2, random_state=None, scale=True):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
    return X_train, X_test, y_train, y_test, scaler

def run_repeated_experiment(model, df, n_runs=5, test_size=0.2, scale=True):
    mses = []
    trained_models = []
    for seed in range(n_runs):
        X_train, X_test, y_train, y_test, _ = preprocess(
            df, test_size=test_size, random_state=seed, scale=scale
        )
        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mses.append(mse)
        trained_models.append(m)
    avg_mse = float(np.mean(mses))
    return avg_mse, mses, trained_models[-1]

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def save_results_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
