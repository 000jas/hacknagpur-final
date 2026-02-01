"""
Contrastive Harassment Detection Model Training
================================================
Trains XGBoost classifier with:
- Class weighting for balanced learning
- Feature importance analysis
- Cross-validation for robustness
- Focus on distinguishing harassment from normal behavior
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import os

print("="*80)
print("CONTRASTIVE HARASSMENT DETECTION - Model Training")
print("="*80)

# Load harassment-focused dataset
print("\n📂 Loading harassment-focused temporal sequence dataset...")
X = np.load("data/features/X_harassment_sequence.npy")
y = np.load("data/features/y_harassment_sequence.npy")

print(f"✅ Dataset loaded: {X.shape[0]:,} sequences with {X.shape[1]} features")
print(f"\n📊 Class Distribution:")
normal_count = np.sum(y == 0)
abnormal_count = np.sum(y == 1)
print(f"   Normal Behavior:     {normal_count:,} sequences ({normal_count/len(y)*100:.1f}%)")
print(f"   Harassment/Abnormal: {abnormal_count:,} sequences ({abnormal_count/len(y)*100:.1f}%)")

# Calculate class weights for balanced learning
scale_pos_weight = normal_count / abnormal_count if abnormal_count > 0 else 1.0
print(f"\n⚖️  Class Weight (scale_pos_weight): {scale_pos_weight:.2f}")
print(f"   → Balances learning to prevent majority class bias")

# Split data
print("\n🔀 Splitting data: 80% train, 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train):,} sequences")
print(f"   Testing:  {len(X_test):,} sequences")

# Feature scaling
print("\n📏 Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features normalized (mean=0, std=1)")

# Train XGBoost model with contrastive learning optimization
print("\n" + "="*80)
print("🎯 Training XGBoost Classifier (Contrastive Learning Mode)")
print("="*80)
print("\n⚙️  Model Configuration:")
print(f"   - Estimators: 500 (deep ensemble for complex patterns)")
print(f"   - Max Depth: 12 (capture nuanced interaction patterns)")
print(f"   - Learning Rate: 0.015 (careful convergence)")
print(f"   - Class Weight: {scale_pos_weight:.2f} (balanced normal/abnormal learning)")
print(f"   - Regularization: L1=0.1, L2=1.5 (prevent overfitting)")
print(f"   - Subsample: 0.8 (robust generalization)")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.015,
    scale_pos_weight=scale_pos_weight,  # Critical for class balance
    reg_alpha=0.1,   # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

print("\n🔄 Training model...")
model.fit(X_train_scaled, y_train, verbose=False)
print("✅ Training complete!")

# Evaluate on test set
print("\n" + "="*80)
print("📈 MODEL PERFORMANCE EVALUATION")
print("="*80)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives (Normal correctly identified):  {cm[0,0]}")
print(f"   False Positives (Normal flagged as harassment): {cm[0,1]}")
print(f"   False Negatives (Harassment missed):           {cm[1,0]}")
print(f"   True Positives (Harassment detected):          {cm[1,1]}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n🎯 Classification Metrics:")
print(f"   Overall Accuracy:        {accuracy*100:.2f}%")
print(f"   Harassment Detection:    {recall*100:.2f}% (recall/sensitivity)")
print(f"   Normal Recognition:      {specificity*100:.2f}% (specificity)")
print(f"   Precision:               {precision*100:.2f}% (when flagged, how often correct)")
print(f"   F1-Score:                {f1*100:.2f}% (harmonic mean)")

# AUC-ROC
try:
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"   AUC-ROC Score:           {auc_score:.4f}")
except:
    auc_score = 0
    print(f"   AUC-ROC Score:           N/A")

# Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Harassment'],
                          digits=3))

# Cross-validation
print("\n🔄 Cross-Validation (5-fold stratified):")
cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='roc_auc')
print(f"   Cross-val AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# Feature Importance Analysis
print("\n" + "="*80)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*80)
print("\nTop 15 Features for Harassment Detection:\n")

feature_importance = model.feature_importances_
feature_names = [
    # Temporal features (0-39)
    'P1_speed_mean', 'P1_speed_std', 'P1_accel_mean', 'P1_accel_std',
    'P1_direction_change', 'P1_path_smoothness', 'P1_stopped_frames',
    'P1_rapid_movements', 'P1_bbox_area_change', 'P1_vertical_movement',
    'P2_speed_mean', 'P2_speed_std', 'P2_accel_mean', 'P2_accel_std',
    'P2_direction_change', 'P2_path_smoothness', 'P2_stopped_frames',
    'P2_rapid_movements', 'P2_bbox_area_change', 'P2_vertical_movement',
    'P1_trajectory_x_mean', 'P1_trajectory_y_mean', 'P1_trajectory_x_std', 'P1_trajectory_y_std',
    'P1_x_range', 'P1_y_range', 'P1_distance_traveled', 'P1_displacement',
    'P2_trajectory_x_mean', 'P2_trajectory_y_mean', 'P2_trajectory_x_std', 'P2_trajectory_y_std',
    'P2_x_range', 'P2_y_range', 'P2_distance_traveled', 'P2_displacement',
    'relative_speed', 'speed_differential', 'synchronized_movement', 'movement_correlation',
    # Interaction features (40-54)
    'avg_distance', 'min_distance', 'max_distance', 'distance_variance',
    'distance_decreasing', 'orientation_diff', 'facing_alignment',
    'p1_facing_p2', 'p2_facing_p1', 'mutual_facing',
    'proximity_zone', 'invasion_count', 'sustained_proximity',
    'approach_speed', 'approach_consistency',
    # Harassment indicators (55-59)
    'following_score', 'loitering_score', 'invasion_score',
    'approach_score', 'evasion_score'
]

# Ensure we don't exceed available features
feature_names = feature_names[:len(feature_importance)]

# Get top features
top_indices = np.argsort(feature_importance)[-15:][::-1]

for idx in top_indices:
    if idx < len(feature_names):
        print(f"   {feature_names[idx]:30s} {feature_importance[idx]:.4f}")

# Identify key harassment indicators
print("\n🚨 Key Harassment Indicators:")
harassment_features = {
    'following_score': feature_importance[55] if len(feature_importance) > 55 else 0,
    'loitering_score': feature_importance[56] if len(feature_importance) > 56 else 0,
    'invasion_score': feature_importance[57] if len(feature_importance) > 57 else 0,
    'approach_score': feature_importance[58] if len(feature_importance) > 58 else 0,
    'evasion_score': feature_importance[59] if len(feature_importance) > 59 else 0
}

for feat, imp in sorted(harassment_features.items(), key=lambda x: x[1], reverse=True):
    print(f"   {feat:20s} {imp:.4f}")

# Save model and scaler
print("\n" + "="*80)
print("💾 Saving Model")
print("="*80)

os.makedirs("models", exist_ok=True)
model_path = "models/harassment_detector_v2.pkl"
scaler_path = "models/harassment_scaler_v2.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Model saved: {model_path}")
print(f"✅ Scaler saved: {scaler_path}")

# Summary
print("\n" + "="*80)
print("🎉 TRAINING SUMMARY")
print("="*80)
print(f"\n✅ Contrastive harassment detection model trained successfully!")
print(f"   - Accuracy: {accuracy*100:.2f}%")
print(f"   - Harassment Detection Rate: {recall*100:.2f}%")
print(f"   - Normal Recognition Rate: {specificity*100:.2f}%")
print(f"   - AUC-ROC: {auc_score:.4f}")
print(f"   - Class Balance: 1:{scale_pos_weight:.2f} (Normal:Abnormal)")
print(f"\n📊 Model ready for deployment!")
print(f"{'='*80}\n")
