import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Load datasets
balanced_data = pd.read_csv('dataset/Blood_samples_dataset_balanced_2(f).csv')
test_data = pd.read_csv('dataset/blood_samples_dataset_test.csv')

# Check data information
print(balanced_data.info())
print(test_data.info())

# Check for missing values
print(balanced_data.isnull().sum())
print(test_data.isnull().sum())

# Remove duplicate rows
balanced_data = balanced_data.drop_duplicates()

# Convert numeric columns to float
balanced_data.iloc[:, :-1] = balanced_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Handle outliers by applying IQR-based filtering
Q1 = balanced_data.select_dtypes(include=[float, int]).quantile(0.25)
Q3 = balanced_data.select_dtypes(include=[float, int]).quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in balanced_data.select_dtypes(include=[float, int]).columns:
    balanced_data = balanced_data[(balanced_data[col] >= lower_bound[col]) & (balanced_data[col] <= upper_bound[col])]

# Normalize numerical data
X = balanced_data.drop(columns=['Disease'])  # Features
y = balanced_data['Disease']  # Target
X_normalized = (X - X.mean()) / X.std()

# Manually split data into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize StratifiedKFold for balanced splits
stratified_cv = StratifiedKFold(n_splits=5)

# Initialize models (Logistic Regression and Random Forest)
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')  # Logistic Regression with balanced class weights
model_rf = RandomForestClassifier(random_state=42, class_weight='balanced')  # Random Forest with balanced class weights

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
print(f'Classification Report:\n{classification_report(y_test, y_pred_lr, zero_division=1)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}')

# Evaluate Random Forest Model
print("\nRandom Forest Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_rf, zero_division=1)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}')

# Use fewer splits in StratifiedKFold
stratified_cv = StratifiedKFold(n_splits=3)  # Reduce the number of splits to 3

# Logistic Regression hyperparameters with StratifiedKFold
param_grid_lr = {'C': [0.1, 1, 10]}
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), param_grid_lr, cv=stratified_cv)
grid_search_lr.fit(X_train, y_train)
print(f"Best parameters for Logistic Regression: {grid_search_lr.best_params_}")

# Random Forest hyperparameters with StratifiedKFold
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_rf, cv=stratified_cv)
grid_search_rf.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")