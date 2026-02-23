import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("BASE_DIR =", BASE_DIR)

EMPLOYEE_CSV = os.path.join(BASE_DIR, "employees.csv")
PERFORMANCE_CSV = os.path.join(BASE_DIR, "performance.csv")

print("EMPLOYEE_CSV =", EMPLOYEE_CSV)
print("EMPLOYEE_CSV exists =", os.path.exists(EMPLOYEE_CSV))
print("PERFORMANCE_CSV exists =", os.path.exists(PERFORMANCE_CSV))

def train():
    print("Loading data...")
    try:
        emp = pd.read_csv(EMPLOYEE_CSV)
        perf = pd.read_csv(PERFORMANCE_CSV)
    except FileNotFoundError:
        print("Data files not found. Run setup_env.py first.")
        return

    # Merge
    data = pd.merge(emp, perf, on="EmployeeID")
    
    # Features
    X = data.drop(columns=['EmployeeID', 'Name', 'Attrition'])
    y = data['Attrition']
    
    # Preprocessing
    categorical_features = ['Department', 'Role', 'Gender']
    numeric_features = ['Age', 'Salary', 'JoiningYear', 'Rating', 'ProjectsCompleted', 'AvgDailyHours']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Model Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Model...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    score = clf.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2f}")
    print(classification_report(y_test, clf.predict(X_test)))
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, "attrition_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
