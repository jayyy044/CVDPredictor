import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('cardio_train.csv', delimiter=';')
data['age_years'] = data['age']//365 #to convert days to age\

# creating a plot for age v cv 
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='age_years', hue='cardio',bins=20, multiple='stack')
plt.title('Cardiovascular Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# systolic bp v cv
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='ap_hi', hue='cardio', multiple = 'stack')
plt.title('Cardiovascular Disease Frequency for Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot cholesterol levels vs. cardiovascular disease
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='cholesterol', hue='cardio', multiple='stack',bins=3)
plt.title('Cholesterol Levels vs. Cardiovascular Disease')
plt.xlabel('Cholesterol Level')
plt.ylabel('Count')
plt.show()

# Example of feature engineering and model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Select features and target
features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
target = 'cardio'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
#what exactly are we doing here?
#we are splitting the data into training and testing data. The training data is 80% of the data and the testing data is 20% of the data.
#The random_state is the seed used by the random number generator. It can be any integer.
#
# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
print("Training Accuracy:", model.score(X_train, y_train))
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


