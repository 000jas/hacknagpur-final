import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

# Load sequence-based features
X = np.load("data/features/X_sequence.npy")
y = np.load("data/features/y_sequence.npy")

print(f"📊 Sequence Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

# Remove any NaN or Inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Enhanced XGBoost model for sequence-based harassment detection
model = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.2,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42,
    tree_method='hist',
    max_bin=512,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0   # L2 regularization
)

print("\n🔄 Training harassment detection model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

print("\n" + "="*60)
print("🎯 HARASSMENT DETECTION MODEL PERFORMANCE")
print("="*60)
print(f"\n🔍 Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"📊 AUC-ROC Score: {auc_score:.4f}")
print(f"✅ Cross-Validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Harassment/Abnormal']))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n   True Negatives: {cm[0,0]}")
print(f"   False Positives: {cm[0,1]} (Normal flagged as harassment)")
print(f"   False Negatives: {cm[1,0]} (Missed harassment)")
print(f"   True Positives: {cm[1,1]} (Correct harassment detection)")

# Calculate detection metrics
if cm[1,1] + cm[1,0] > 0:
    detection_rate = cm[1,1] / (cm[1,1] + cm[1,0])
    print(f"\n✅ Harassment Detection Rate: {detection_rate:.2%}")

if cm[0,0] + cm[0,1] > 0:
    false_alarm_rate = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"⚠️  False Alarm Rate: {false_alarm_rate:.2%}")

# Feature importance analysis
feature_importance = model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-15:]

print("\n🔝 Top 15 Most Important Features for Harassment Detection:")
feature_names = [
    # Temporal features person 1 (0-19)
    "P1_avg_speed", "P1_max_speed", "P1_speed_variance", "P1_direction_variance",
    "P1_max_direction_change", "P1_linearity", "P1_avg_acceleration", "P1_accel_variance",
    "P1_stop_ratio", "P1_stopped_frames", "P1_sudden_movements", "P1_posture_changes",
    "P1_X_displacement", "P1_Y_displacement", "P1_path_length", "P1_erratic_flag",
    "P1_loitering_flag", "P1_nonlinear_flag", "P1_fast_movement", "P1_direction_changes",
    
    # Temporal features person 2 (20-39)
    "P2_avg_speed", "P2_max_speed", "P2_speed_variance", "P2_direction_variance",
    "P2_max_direction_change", "P2_linearity", "P2_avg_acceleration", "P2_accel_variance",
    "P2_stop_ratio", "P2_stopped_frames", "P2_sudden_movements", "P2_posture_changes",
    "P2_X_displacement", "P2_Y_displacement", "P2_path_length", "P2_erratic_flag",
    "P2_loitering_flag", "P2_nonlinear_flag", "P2_fast_movement", "P2_direction_changes",
    
    # Interaction features (40-54)
    "distance", "relative_angle", "facing_alignment", "p1_facing_p2", "p2_facing_p1",
    "height_ratio", "intimate_zone", "personal_zone", "social_zone", "orientation_diff",
    "p1_hands_raised", "p2_hands_raised", "X_offset", "Y_offset", "approaching",
    
    # Harassment indicators (55-59)
    "distance_trend", "following_score", "invasion_count", "persistent_approach", "frequent_invasion"
]

for idx in reversed(top_features_idx):
    if idx < len(feature_names):
        print(f"   {feature_names[idx]}: {feature_importance[idx]:.4f}")
    else:
        print(f"   Feature {idx}: {feature_importance[idx]:.4f}")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/harassment_detector.pkl")
joblib.dump(scaler, "models/sequence_scaler.pkl")

print("\n" + "="*60)
print("✅ Harassment detection model saved successfully!")
print(f"   - models/harassment_detector.pkl")
print(f"   - models/sequence_scaler.pkl")
print("="*60)

print("\n💡 Model Insights:")
print("   - Uses temporal sequences to detect sustained patterns")
print("   - Tracks interpersonal interactions and proximity violations")
print("   - Identifies following behavior and approach patterns")
print("   - Privacy-preserving: No facial recognition, only pose analysis")
print("   - Sequence-level confidence for reliable harassment detection")
