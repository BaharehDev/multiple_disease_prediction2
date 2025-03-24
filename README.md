# Machine Learning Model for Disease Prediction

## Overview  
This project applies **Logistic Regression** and **Random Forest** to predict disease risk based on patient data. The dataset is preprocessed, trained, and evaluated with model performance metrics.

---

## Data Processing  
- Load datasets from CSV files.  
- Check for missing values and remove duplicates.  
- Convert numeric data to float type.  
- Handle outliers using **IQR filtering**.  
- Normalize numerical features.  
- Split data into **80% training / 20% testing**.  

---

## Model Training  
### - Logistic Regression  
- Uses `class_weight='balanced'` to handle class imbalances.  
- Trained with hyperparameter tuning (`C=10`).  

### - Random Forest  
- Trained with `class_weight='balanced'`.  
- Best parameters found: `max_depth=10, n_estimators=100`.  

---

## Model Evaluation  
- **Accuracy**: Measures overall model correctness.  
- **Classification Report**: Precision, recall, and F1-score per class.  
- **Confusion Matrix**: Visualizes true and false predictions.  

---

## Hyperparameter Tuning  
- `GridSearchCV` used to optimize:  
  - **Logistic Regression**: `C` values tested: `[0.1, 1, 10]`.  
  - **Random Forest**: `n_estimators=[100, 200]`, `max_depth=[10, 20]`.  

---

## Results  
- Best model parameters found for both classifiers.  
- Models trained and evaluated successfully.  
- No further changes required for the assignment.  

---

## Next Steps  
- If needed, fine-tune with more hyperparameter combinations.  
- Test with different cross-validation strategies.  
- Explore additional models for comparison.  

---

### Ready for Submission!
