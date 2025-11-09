"""
Train selected Random Forest model for wine quality prediction
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# Load and prepare data
print("Loading data...")
df = pd.read_csv('midterm-datasets/winequality-red.csv', sep=';')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.lower()

# Remove duplicates
print(f"Original dataset size: {len(df)}")
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(f"After removing duplicates: {len(df)}")

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Prepare target variable (predicting high quality wine: rating >= 6)
y_train = (df_train.quality >= 6).astype('int').values
y_val = (df_val.quality >= 6).astype('int').values
y_test = (df_test.quality >= 6).astype('int').values

# Remove target from features
df_train_features = df_train.drop('quality', axis=1)
df_val_features = df_val.drop('quality', axis=1)
df_test_features = df_test.drop('quality', axis=1)

print(f"\nTrain size: {len(df_train)}")
print(f"Validation size: {len(df_val)}")
print(f"Test size: {len(df_test)}")

# Convert to DictVectorizer format
print("\nPreparing features...")
train_dicts = df_train_features.fillna(0).to_dict(orient='records')
val_dicts = df_val_features.fillna(0).to_dict(orient='records')
test_dicts = df_test_features.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

# Train the best Random Forest model
print("\nTraining Random Forest model...")
print("Parameters: n_estimators=50, max_depth=10, min_samples_leaf=20")

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = rf_model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred_val)
print(f"\nValidation AUC: {auc_val:.4f}")

# Evaluate on test set
y_pred_test = rf_model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_test)
print(f"Test AUC: {auc_test:.4f}")

# Save the model and DictVectorizer
print("\nSaving model and vectorizer...")
with open('wine_quality_model.pkl', 'wb') as f_model:
    pickle.dump(rf_model, f_model)

with open('wine_quality_dv.pkl', 'wb') as f_dv:
    pickle.dump(dv, f_dv)

print("\nModel training complete!")
print("Files saved:")
print("  - wine_quality_model.pkl")
print("  - wine_quality_dv.pkl")

# Print feature names for reference
print("\nExpected input features:")
for i, feature in enumerate(dv.get_feature_names_out(), 1):
    print(f"  {i}. {feature}")
