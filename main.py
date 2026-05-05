import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.utils import save_model

# Load data
df = pd.read_csv("data/creditcard.csv")

# Sample for faster execution
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=20000, random_state=42)
df = pd.concat([fraud, normal])

# Preprocess
X, y = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train
model = train_model(X_train_res, y_train_res)

# Predict
y_pred = model.predict(X_test)

# Evaluate
evaluate_model(y_test, y_pred)

# Save
save_model(model)

print("✅ Project executed successfully!")