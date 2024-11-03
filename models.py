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

data = pd.read_csv('cardio_train.csv', delimiter=';')
data['age_years'] = data['age']//365 


features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
target = 'cardio'


X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=30)


models = {
    'Random Forest': RandomForestClassifier(random_state=30),
    'Gradient Boosting': GradientBoostingClassifier(random_state=30),
    'XGBoost': XGBClassifier(random_state=30),
    'LightGBM': LGBMClassifier(random_state=30),
    'CatBoost': CatBoostClassifier(random_state=30, verbose=0),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=30),
    'SVM': SVC(random_state=30)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Model')
    print(y_pred)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    accuracies[name] = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print('-' * 60)


sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
model_names = [item[0] for item in sorted_accuracies]
accuracy_values = [item[1] for item in sorted_accuracies]

plt.figure(figsize=(12, 8))
sns.barplot(x=accuracy_values, y=model_names, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.show()

