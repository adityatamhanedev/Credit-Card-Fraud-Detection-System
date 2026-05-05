import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.dropna(subset=['Class'])

    X = df.drop('Class', axis=1)
    y = df['Class']

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])

    return X, y