import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('cardio_train.csv', sep=';')

# Drop rows with any null values
data = data.dropna()

#Converting age from days to years for easier processing 
data['age'] = (data['age'] / 365).astype(int)

#Generally since Diastolic blood pressure needs to be less then Systolic blood pressure
#We need extreme values for blood pressure and get them into realistic ranges 
#We can check and remove outliers from the data set
rangedBP = data[(data['ap_hi'] > 90) & (data['ap_hi'] < 180) &
                (data['ap_lo'] > 60) & (data['ap_lo'] < 120)]

# Keep rows where diastolic is less than systolic
cleanedData = rangedBP[rangedBP['ap_lo'] < rangedBP['ap_hi']].copy()

#Transferring height into meters 
cleanedData.loc[:, 'height_m'] = cleanedData['height'] / 100.0

#Dropping rows that we don't need to predictions
cleanedData = cleanedData.drop(columns=['id', 'height'])

# Removin outliers/ unrealistic values from the specified column so we cna work with realistic data points
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Apply to each relevant column
height_outliers = detect_outliers(cleanedData, 'height_m')
weight_outliers = detect_outliers(cleanedData, 'weight')
systolic_outliers = detect_outliers(cleanedData, 'ap_hi')
diastolic_outliers = detect_outliers(cleanedData, 'ap_lo')

# Combine all outliers
all_outliers = pd.concat([height_outliers, weight_outliers, systolic_outliers, diastolic_outliers])
all_outliers = all_outliers.drop_duplicates()

# Filter out outliers
df_filtered = cleanedData[~cleanedData.index.isin(all_outliers.index)]

#adding BMI to the dataset to relate weight and height
df_filtered['BMI'] = df_filtered['weight'] / (df_filtered['height_m'] ** 2)

# Calculate Pulse Pressure: as its correlated to 
df_filtered['pulse_pressure'] = df_filtered['ap_hi'] - df_filtered['ap_lo']

# Lifestyle connections
df_filtered['lifestyle_score'] = (
    df_filtered['smoke'] + 
    df_filtered['alco'] + 
    (df_filtered['active'] == 0).astype(int)
)

# Age and blood pressure interaction
df_filtered['age_bp_interaction'] = df_filtered['age'] * (df_filtered['ap_lo'] + ((1/3)*(df_filtered['ap_hi']-df_filtered['ap_lo'])))

# Combine cholesterol and glucose into a metabolic score
df_filtered['metabolic_score'] = df_filtered['cholesterol'] + df_filtered['gluc']

# # # List of columns to plot
# columns = ['height_m', 'weight', 'ap_hi', 'ap_lo']

# # Print summary statistics
# for column in columns:
#     print(f"\n{column}:")
#     print("Original data:")
#     print(cleanedData[column].describe())
#     print("\nFiltered data:")
#     print(df_filtered[column].describe())

# # Print number of rows removed
# print(f"\nOriginal dataset size: {len(cleanedData)}")
# print(f"Filtered dataset size: {len(df_filtered)}")
# print(f"Number of rows removed: {len(cleanedData) - len(df_filtered)}")
# print(f"Percentage of data removed: {((len(cleanedData) - len(df_filtered)) / len(cleanedData)) * 100:.2f}%")
print("Target class distribution:")
print(df_filtered['cardio'].value_counts(normalize=True))

# Separate features and target variable
X = df_filtered.drop(columns=['cardio'])
y = df_filtered['cardio']

# Encode categorical features if any exist
categorical_features = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set up the LightGBM dataset
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)

# Define model parameters
params = {
    'objective': 'binary',
    'random_state': 42,
    'metric': 'binary_logloss',
}

# Train the model with early stopping using callbacks
model = lgb.train(
    params,
    d_train,
    valid_sets=[d_test],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),  # Early stopping callback
        lgb.log_evaluation(10)  # Logs evaluation metrics every 10 rounds
    ]
)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Convert probabilities to binary predictions (threshold 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.4f}")


lgb.plot_importance(model, importance_type='split', title='Feature Importance', xlabel='Importance Score', ylabel='Features')
plt.show()