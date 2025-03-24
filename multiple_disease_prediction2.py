import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
balanced_data = pd.read_csv('/Users/bahar/Documents/IT-Högskolan/assignment3/dataset/Blood_samples_dataset_balanced_2(f).csv')
test_data = pd.read_csv('/Users/bahar/Documents/IT-Högskolan/assignment3/dataset/blood_samples_dataset_test.csv')

# Check data information (already done in Step 1)
print(balanced_data.info())
print(test_data.info())

# Check for missing values
print(balanced_data.isnull().sum())
print(test_data.isnull().sum())

# Remove duplicate rows
balanced_data = balanced_data.drop_duplicates()

# Convert numeric columns to float (if any)
balanced_data.iloc[:, :-1] = balanced_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Calculate quantiles for outlier handling
Q1 = balanced_data.select_dtypes(include=['number']).quantile(0.25)
Q3 = balanced_data.select_dtypes(include=['number']).quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Apply the filtering column by column for outliers
for col in balanced_data.select_dtypes(include=['number']).columns:
    balanced_data = balanced_data[(balanced_data[col] >= lower_bound[col]) & (balanced_data[col] <= upper_bound[col])]

# Standardize numerical data
scaler = StandardScaler()
X = balanced_data.drop(columns=['Disease'])  # Features
y = balanced_data['Disease']  # Target

# Standardize numerical data
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models (Logistic Regression and Random Forest)
model_lr = LogisticRegression(max_iter=1000)  # Logistic Regression
model_rf = RandomForestClassifier(random_state=42)  # Random Forest

# Train Logistic Regression Model
model_lr.fit(X_train, y_train)

# Train Random Forest Model
model_rf.fit(X_train, y_train)

# Predict using Logistic Regression
y_pred_lr = model_lr.predict(X_test)

# Predict using Random Forest
y_pred_rf = model_rf.predict(X_test)

# Evaluate Logistic Regression Model
print("Logistic Regression Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_lr)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}')

# Evaluate Random Forest Model
print("\nRandom Forest Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_rf)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}')