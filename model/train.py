# ===== CELLULE 1: IMPORTS AMÉLIORÉS =====
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

print("XGBoost version:", xgb.__version__)
print("LightGBM version:", lgb.__version__)

# ===== CELLULE 2: CHARGER LES DONNÉES =====
s3 = boto3.client('s3')
bucket = 'data-pipeline-1764670683'
key = 'preprocessed-data/train_preprocessed.csv'

print(f"Loading data from s3://{bucket}/{key}")
obj = s3.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(obj['Body'])

print(f"Data shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean():.2%}")

# ===== CELLULE 3: FEATURE ENGINEERING AVANCÉ =====
def advanced_feature_engineering(df):
    """Créer des features plus sophistiquées"""
    
    df_enhanced = df.copy()
    
    # 1. Features statistiques par groupe (si applicable)
    # Exemple: statistiques par card, addr, email, etc.
    
    # 2. Features temporelles (si TransactionDT existe)
    if 'TransactionDT' in df_enhanced.columns:
        df_enhanced['hour'] = (df_enhanced['TransactionDT'] // 3600) % 24
        df_enhanced['day'] = (df_enhanced['TransactionDT'] // (3600 * 24))
        df_enhanced['day_of_week'] = df_enhanced['day'] % 7
        df_enhanced['is_night'] = ((df_enhanced['hour'] >= 22) | (df_enhanced['hour'] <= 6)).astype(int)
        df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
    
    # 3. Features sur les montants (si TransactionAmt existe)
    if 'TransactionAmt' in df_enhanced.columns:
        df_enhanced['amt_log'] = np.log1p(df_enhanced['TransactionAmt'])
        df_enhanced['amt_decimal'] = df_enhanced['TransactionAmt'] - df_enhanced['TransactionAmt'].astype(int)
        df_enhanced['is_round_amt'] = (df_enhanced['amt_decimal'] < 0.01).astype(int)
        df_enhanced['amt_sqrt'] = np.sqrt(df_enhanced['TransactionAmt'])
        
        # Statistiques par quantile
        df_enhanced['amt_quantile'] = pd.qcut(df_enhanced['TransactionAmt'], q=10, labels=False, duplicates='drop')
    
    # 4. Interactions entre features (exemples)
    if 'C1' in df_enhanced.columns and 'C2' in df_enhanced.columns:
        df_enhanced['C1_C2_ratio'] = df_enhanced['C1'] / (df_enhanced['C2'] + 1)
        df_enhanced['C1_C2_diff'] = df_enhanced['C1'] - df_enhanced['C2']
        df_enhanced['C1_C2_prod'] = df_enhanced['C1'] * df_enhanced['C2']
    
    if 'D1' in df_enhanced.columns and 'D2' in df_enhanced.columns:
        df_enhanced['D1_D2_ratio'] = df_enhanced['D1'] / (df_enhanced['D2'] + 1)
        df_enhanced['D1_D2_diff'] = df_enhanced['D2'] - df_enhanced['D1']
    
    # 5. Features d'agrégation (velocity features)
    # Par exemple: nombre de transactions par carte dans une fenêtre temporelle
    
    print(f"Features créées: {df_enhanced.shape[1] - df.shape[1]} nouvelles features")
    
    return df_enhanced

# Appliquer le feature engineering
df = advanced_feature_engineering(df)

# ===== CELLULE 4: PRÉPARER X ET Y =====
y = df['isFraud']
X = df.drop('isFraud', axis=1)

# Gérer les valeurs infinies
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(-999)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Fraud distribution: {y.value_counts().to_dict()}")
print(f"Fraud rate: {y.mean():.2%}")

# ===== CELLULE 5: SPLIT AVEC VALIDATION SET =====
# Split en 3: Train (60%), Validation (20%), Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape} - Fraud rate: {y_train.mean():.2%}")
print(f"Validation: {X_val.shape} - Fraud rate: {y_val.mean():.2%}")
print(f"Test: {X_test.shape} - Fraud rate: {y_test.mean():.2%}")

# ===== CELLULE 6: AMÉLIORATION 1 - SMOTE (OPTIONNEL) =====
# Utiliser SMOTE pour équilibrer les classes (optionnel, à tester)
USE_SMOTE = False  # Mettre True pour essayer SMOTE

if USE_SMOTE:
    print("\nApplying SMOTE...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_balanced.shape}")
    print(f"New fraud rate: {y_train_balanced.mean():.2%}")
else:
    X_train_balanced = X_train
    y_train_balanced = y_train

# ===== CELLULE 7: XGBOOST OPTIMISÉ =====
print("\n" + "="*60)
print("TRAINING OPTIMIZED XGBOOST")
print("="*60)

# Paramètres optimisés
scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,                    # Augmenté pour plus de capacité
    'learning_rate': 0.05,             # Réduit pour meilleur apprentissage
    'n_estimators': 500,               # Augmenté
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,             # Ajouté
    'gamma': 0.1,                      # Ajouté pour régularisation
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.0,                 # L2 regularization
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0,
    'n_jobs': -1
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# Early stopping sur validation set
xgb_model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=[(X_train_balanced, y_train_balanced), (X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# Prédictions XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
ap_xgb = average_precision_score(y_test, y_pred_proba_xgb)

print(f"\n✅ XGBoost Results:")
print(f"   AUC-ROC: {auc_xgb:.4f}")
print(f"   Average Precision: {ap_xgb:.4f}")

# ===== CELLULE 8: LIGHTGBM (ALTERNATIVE/ENSEMBLE) =====
print("\n" + "="*60)
print("TRAINING LIGHTGBM")
print("="*60)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

# Créer datasets LightGBM
lgb_train = lgb.Dataset(X_train_balanced, y_train_balanced)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Entraîner
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Prédictions LightGBM
y_pred_proba_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)
ap_lgb = average_precision_score(y_test, y_pred_proba_lgb)

print(f"\n✅ LightGBM Results:")
print(f"   AUC-ROC: {auc_lgb:.4f}")
print(f"   Average Precision: {ap_lgb:.4f}")

# ===== CELLULE 9: ENSEMBLE MODEL =====
print("\n" + "="*60)
print("ENSEMBLE MODEL (XGBoost + LightGBM)")
print("="*60)

# Moyenne pondérée des prédictions
y_pred_proba_ensemble = (0.5 * y_pred_proba_xgb + 0.5 * y_pred_proba_lgb)

auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
ap_ensemble = average_precision_score(y_test, y_pred_proba_ensemble)

print(f"\n🏆 ENSEMBLE Results:")
print(f"   AUC-ROC: {auc_ensemble:.4f}")
print(f"   Average Precision: {ap_ensemble:.4f}")

# ===== CELLULE 10: TROUVER LE SEUIL OPTIMAL =====
def find_optimal_threshold(y_true, y_pred_proba):
    """Trouve le seuil optimal basé sur le F1-score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold, f1_scores[optimal_idx]

optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba_ensemble)

print(f"\n📊 Optimal Threshold: {optimal_threshold:.3f}")
print(f"   F1-Score at optimal threshold: {optimal_f1:.4f}")

# Prédictions avec seuil optimal
y_pred_optimal = (y_pred_proba_ensemble >= optimal_threshold).astype(int)

# ===== CELLULE 11: ÉVALUATION COMPLÈTE =====
print("\n" + "="*60)
print("FINAL EVALUATION (Ensemble + Optimal Threshold)")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal, 
                          target_names=['Normal', 'Fraud'],
                          digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_optimal)
print(cm)

# Métriques détaillées
tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Metrics:")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives: {tp:,}")
print(f"\nFraud Detection Rate: {tp/(tp+fn)*100:.2f}%")
print(f"False Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"Precision: {tp/(tp+fp)*100:.2f}%")

# ===== CELLULE 12: VISUALISATIONS =====
# 1. Feature Importance (XGBoost)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_ensemble)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, label=f'Ensemble (AUC = {auc_ensemble:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Improved Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba_ensemble)

plt.figure(figsize=(10, 8))
plt.plot(recalls, precisions, linewidth=2, label=f'Ensemble (AP = {ap_ensemble:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Improved Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curve_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== CELLULE 13: SAUVEGARDER LES MODÈLES =====
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

# Sauvegarder XGBoost
joblib.dump(xgb_model, 'fraud_xgboost_improved.joblib')
s3.upload_file(
    'fraud_xgboost_improved.joblib',
    bucket,
    'model-artifacts/fraud_xgboost_improved.joblib'
)
print("✅ XGBoost saved")

# Sauvegarder LightGBM
lgb_model.save_model('fraud_lightgbm_improved.txt')
s3.upload_file(
    'fraud_lightgbm_improved.txt',
    bucket,
    'model-artifacts/fraud_lightgbm_improved.txt'
)
print("✅ LightGBM saved")

# Sauvegarder les feature names et le threshold optimal
import json

metadata = {
    'feature_names': X.columns.tolist(),
    'optimal_threshold': float(optimal_threshold),
    'auc_xgb': float(auc_xgb),
    'auc_lgb': float(auc_lgb),
    'auc_ensemble': float(auc_ensemble),
    'ap_ensemble': float(ap_ensemble),
    'n_features': X.shape[1]
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

s3.upload_file(
    'model_metadata.json',
    bucket,
    'model-artifacts/model_metadata.json'
)
print("✅ Metadata saved")

# ===== CELLULE 14: COMPARAISON DES PERFORMANCES =====
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

results_df = pd.DataFrame({
    'Model': ['Original XGBoost', 'Improved XGBoost', 'LightGBM', 'Ensemble'],
    'AUC-ROC': [0.9182, auc_xgb, auc_lgb, auc_ensemble],
    'Avg Precision': ['-', ap_xgb, ap_lgb, ap_ensemble]
})

print(results_df.to_string(index=False))
print(f"\n🎯 Improvement: {(auc_ensemble - 0.9182)*100:.2f} percentage points")

print("\n" + "="*60)
print("✅ ALL DONE! Models saved to S3.")
print("="*60)