from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from  xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('cardio_train.csv', delimiter=';')
data['age_years'] = data['age']//365 #to convert days to age

# Select features and target

features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
target = 'cardio'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0, random_state=30)
print(X_test)
print(y_test)
# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=30),
    'Gradient Boosting': GradientBoostingClassifier(random_state=30),
    'XGBoost': XGBClassifier(random_state=30),
    'LightGBM': LGBMClassifier(random_state=30),
    'CatBoost': CatBoostClassifier(random_state=30, verbose=0),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=30),
    'SVM': SVC(random_state=30)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Model')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-' * 60)