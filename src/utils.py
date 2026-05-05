import joblib

def save_model(model, path="models/fraud_model.pkl"):
    joblib.dump(model, path)

def load_model(path="models/fraud_model.pkl"):
    return joblib.load(path)