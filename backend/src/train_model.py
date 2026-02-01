import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Load features
X = np.load("data/features/X.npy")
y = np.load("data/features/y.npy")

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

# Feature scaling for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Enhanced XGBoost model with optimized parameters for accuracy
model = XGBClassifier(
    n_estimators=300,  # More trees for better accuracy
    max_depth=8,  # Deeper trees
    learning_rate=0.03,  # Lower learning rate for precision
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0.1,  # Regularization
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle class imbalance
    random_state=42,
    tree_method='hist',  # Faster training
    max_bin=256
)

print("\nTraining model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Cross-validation for robust accuracy measurement
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Detailed metrics
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Feature importance
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-10:]
print("\nTop 10 Most Important Features:")
for idx in reversed(top_features):
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/behavior_xgb.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
print(f"   - models/behavior_xgb.pkl")
print(f"   - models/scaler.pkl")