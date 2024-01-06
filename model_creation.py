# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

vaccine_data = pd.read_csv("RS_Session_254_AU_878_1.csv")
positivity_data = pd.read_csv("WeeklyDistrictPositivityRateGT10_10-16November2021.csv")

# Explore vaccine dataset
print("Vaccine Dataset:")
print(vaccine_data.head())

# Explore positivity dataset
print("\nPositivity Dataset:")
print(positivity_data.head())
# Visualize vaccine doses using bar charts
plt.figure(figsize=(12, 6))
sns.barplot(x="State/UT", y="1st Dose - 18-44 years", data=vaccine_data)
plt.title("1st Dose Administered - 18-44 years")
plt.xticks(rotation=90)
plt.show()

# Visualize positivity rates using line charts
plt.figure(figsize=(12, 6))
sns.lineplot(x="District", y="Positivity", data=positivity_data)
plt.title("Positivity Rates by District")
plt.xticks(rotation=90)
plt.show()

# Identify numeric and non-numeric columns
numeric_columns = vaccine_data.select_dtypes(include=[np.number]).columns
non_numeric_columns = vaccine_data.select_dtypes(exclude=[np.number]).columns

# Handle missing data for numeric columns
imputer_numeric = SimpleImputer(strategy="mean")
vaccine_data_imputed_numeric = pd.DataFrame(imputer_numeric.fit_transform(vaccine_data[numeric_columns]), columns=numeric_columns)

# Concatenate imputed numeric columns with non-numeric columns
vaccine_data_imputed = pd.concat([vaccine_data_imputed_numeric, vaccine_data[non_numeric_columns]], axis=1)

# Feature engineering
vaccine_data_imputed['Total Doses'] = vaccine_data_imputed['1st Dose - HCWs'] + vaccine_data_imputed['1st Dose - FLWs'] + vaccine_data_imputed['1st Dose - 45+ years'] + vaccine_data_imputed['1st Dose - 18-44 years']



# Prepare data for ML
X = vaccine_data_imputed[['1st Dose - HCWs', '1st Dose - FLWs', '1st Dose - 45+ years', 'Total Doses']]
y = vaccine_data_imputed['1st Dose - 18-44 years']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Advanced model - Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the trained model to a file
joblib.dump(best_rf_model, 'advanced_vaccine_dose_model.pkl')
# Visualize feature importance
feature_importance = best_rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.title('Residual Analysis')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()

from sklearn.model_selection import cross_val_score

# Cross-validation score
cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
mean_cv_rmse = np.mean(cv_rmse_scores)
print(f"Mean Cross-Validation RMSE: {mean_cv_rmse}")

from sklearn.pipeline import Pipeline

# Create a pipeline
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Fit the pipeline
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_pipeline = model_pipeline.predict(X_test)

# Evaluate the model
mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
r2_pipeline = r2_score(y_test, y_pred_pipeline)

print(f"Mean Squared Error (Pipeline): {mse_pipeline}")
print(f"R-squared (Pipeline): {r2_pipeline}")

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=100,
                                   cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train_scaled, y_train)
best_rf_model_random = random_search.best_estimator_
