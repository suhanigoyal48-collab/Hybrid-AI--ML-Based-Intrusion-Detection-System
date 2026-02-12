print("\nBuilding and training Stacking Ensemble model with TF-IDF features...")

# Import StackingClassifier and LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier # Ensure imported if not in current cell scope
from lightgbm import LGBMClassifier # Ensure imported if not in current cell scope

# Best parameters for XGBoost (from previous re-tuning with TF-IDF)
xgb_best_params_tfidf = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300}

# Best parameters for LightGBM (from previous re-tuning with TF-IDF)
lgbm_best_params_tfidf = {'learning_rate': 0.05, 'max_depth': 15, 'n_estimators': 200, 'num_leaves': 70}

# Define base estimators
estimators = [
    ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **xgb_best_params_tfidf)),
    ('lgbm', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **lgbm_best_params_tfidf))
]

# Define final estimator
final_estimator = LogisticRegression(random_state=42, solver='liblinear')

# Instantiate StackingClassifier
stacking_model_tfidf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5, # Cross-validation folds for stacking
    n_jobs=-1
)

# Fit the StackingClassifier to the X_resampled and y_resampled data
print("Fitting StackingClassifier...")
stacking_model_tfidf.fit(X_resampled, y_resampled)
print("StackingClassifier fitted successfully.")

# Predict on the X_test_scaled_df data
y_pred_ensemble_tfidf = stacking_model_tfidf.predict(X_test_scaled_df)
y_prob_ensemble_tfidf = stacking_model_tfidf.predict_proba(X_test_scaled_df)[:, 1]

# Evaluate the ensemble model's performance
print("\n===== Stacking Ensemble (with TF-IDF) MODEL PERFORMANCE ====")

accuracy_ensemble_tfidf = accuracy_score(y_test, y_pred_ensemble_tfidf)
print("Accuracy:", round(accuracy_ensemble_tfidf, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble_tfidf))

# Confusion Matrix
cm_ensemble_tfidf = confusion_matrix(y_test, y_pred_ensemble_tfidf)
plt.figure(figsize=(5,4))
sns.heatmap(cm_ensemble_tfidf, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Stacking Ensemble with TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.close() # Close the plot

# ROC Curve
auc_score_ensemble_tfidf = roc_auc_score(y_test, y_prob_ensemble_tfidf)
fpr_ensemble_tfidf, tpr_ensemble_tfidf, _ = roc_curve(y_test, y_prob_ensemble_tfidf)

plt.figure()
plt.plot(fpr_ensemble_tfidf, tpr_ensemble_tfidf)
plt.title(f"ROC Curve (Stacking Ensemble with TF-IDF) (AUC = {round(auc_score_ensemble_tfidf,4)})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
plt.close() # Close the plot

print("ROC-AUC Score (Stacking Ensemble with TF-IDF):", round(auc_score_ensemble_tfidf, 4))

# Save Ensemble Model
joblib.dump(stacking_model_tfidf, "IDS_stacking_ensemble_model_tfidf.pkl")
print("\nStacking Ensemble model (with TF-IDF) saved successfully.")