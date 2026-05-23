import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier, BaggingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import random
import warnings
import os
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_models(data_path):
    set_seed(42)
    print(f"Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target'].astype(int).values
    
    print(f"Dataset shape: {X.shape} features, {len(y)} samples.")
    print(f"Class distribution: {np.bincount(y)}")

    # Compute imbalance ratio for XGBoost
    class_counts = np.bincount(y)
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f} — applying weights where supported")

    # Define models
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

    estimators = [('rf', rf), ('gb', gb)]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=lr)
    voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('gb', gb)], voting='soft')

    models = {
        'Logistic Regression': lr,
        'LDA':                 LinearDiscriminantAnalysis(),
        'Decision Tree':       DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest':       rf,
        'Gradient Boosting':   gb,
        'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
        'LightGBM':            LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
        'CatBoost':            CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'SVM Linear':          SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced'),
        'SVM RBF':             SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced'),
        'KNN':                 KNeighborsClassifier(),
        'Gaussian NB':         GaussianNB(),
        'MLP Neural Network':  MLPClassifier(random_state=42, max_iter=1000),
        'Bagging':             BaggingClassifier(random_state=42),
        'Voting Ensemble':     voting_clf,
        'Stacking Ensemble':   stacking_clf,
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    fold_results = []
    detailed_metrics = []
    
    # Structures to hold fold-by-fold probabilities and metrics for ROC curve plotting
    model_fold_probabilities = {name: [] for name in models.keys()}
    model_fold_targets = {name: [] for name in models.keys()}
    cv_auc_scores = {name: [] for name in models.keys()}
    
    oof_predictions = {name: np.zeros(len(y)) for name in models.keys()}
    oof_probabilities = {name: np.zeros(len(y)) for name in models.keys()}
    cv_recall_raw = {}

    print("\nEvaluating Models using 5-Fold Cross Validation...")
    for name, model in models.items():
        print(f"--- {name} ---")
        
        fold_acc, fold_prec, fold_rec, fold_f1, fold_auc, fold_mcc = [], [], [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            sc = StandardScaler()
            X_train = sc.fit_transform(X.iloc[train_idx].values)
            X_test  = sc.transform(X.iloc[test_idx].values)
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                
            oof_predictions[name][test_idx] = y_pred
            oof_probabilities[name][test_idx] = y_prob
            
            # Save raw fold data for the individual ROC lines
            model_fold_probabilities[name].append(y_prob)
            model_fold_targets[name].append(y_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_val = roc_auc_score(y_test, y_prob)
            mcc = matthews_corrcoef(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fold_acc.append(acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            fold_f1.append(f1)
            fold_auc.append(auc_val)
            fold_mcc.append(mcc)
            
            cv_auc_scores[name].append(auc_val)
            
            fold_results.append({
                'Model': name, 'Fold': fold_idx + 1, 'Accuracy': acc, 'Precision': prec,
                'Recall': rec, 'F1': f1, 'ROC-AUC': auc_val, 'MCC': mcc, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
            })
            
            detailed_metrics.append({'Model': name, 'Metric': 'Precision', 'Score': prec})
            detailed_metrics.append({'Model': name, 'Metric': 'Recall', 'Score': rec})
            detailed_metrics.append({'Model': name, 'Metric': 'ROC-AUC', 'Score': auc_val})
            detailed_metrics.append({'Model': name, 'Metric': 'MCC', 'Score': mcc})
        
        mean_recall = np.mean(fold_rec)
        cv_recall_raw[name] = mean_recall
        
        metrics = {
            'Model': name,
            'Accuracy': f"{np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}",
            'Precision': f"{np.mean(fold_prec):.3f} ± {np.std(fold_prec):.3f}",
            'Recall': f"{mean_recall:.3f} ± {np.std(fold_rec):.3f}",
            'F1': f"{np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}",
            'ROC-AUC': f"{np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}",
            'MCC': f"{np.mean(fold_mcc):.3f} ± {np.std(fold_mcc):.3f}"
        }
        results.append(metrics)
        print(f"ROC-AUC: {metrics['ROC-AUC']}\nRecall:  {metrics['Recall']}\nMCC:     {metrics['MCC']}\n")

    # Output storage
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(fold_results).to_csv('results/fold_metrics.csv', index=False)
    summary_df = pd.DataFrame(results)
    summary_df.to_csv('results/summary_results.csv', index=False)

    print("\n===================== RESULTS SUMMARY =====================")
    print(summary_df.to_string(index=False))

    # Identify Best Model Based On Recall
    best_model_name = max(cv_recall_raw, key=cv_recall_raw.get)
    print(f"\n🥇 Top Model Selected by Cross-Validated Recall: '{best_model_name}'")
    best_estimator = models[best_model_name]

    # ── 1. Comprehensive Performance Chart (With SD Error Bars & Clean Spines) ──
    print("\nGenerating model benchmark performance comparison plot...")
    plot_df = pd.DataFrame(detailed_metrics)
    
    plt.figure(figsize=(15, 7))
    sns.set_theme(style="ticks")
    
    ax = sns.barplot(
        data=plot_df, x="Model", y="Score", hue="Metric", 
        palette="muted", errorbar="sd", capsize=0.06, err_kws={'linewidth': 1.2}
    )
    
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.xlabel("Classifier Model", fontsize=12, fontweight='bold')
    plt.ylim(-0.1, 1.05)
    plt.legend(loc="upper right", frameon=True)
    
    sns.despine(top=True, right=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11, fontweight='bold')
    plt.yticks(fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/model_metrics_comparison.png', dpi=300)
    plt.show()

    # ── 2. Top 5 Models Combined ROC Curve Plot ─────────────────────────────────
    print("\nGenerating ROC Curves for Top 5 Models based on Mean ROC-AUC...")
    
    # Calculate mean AUC and sort to identify the top 5 models
    mean_auc_scores = {name: np.mean(scores) for name, scores in cv_auc_scores.items()}
    top_5_models = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)[:5]
    
    plt.figure(figsize=(7, 6))
    
    for name in top_5_models:
        # Concatenate out-of-fold targets and probabilities to generate global curve path cleanly
        all_y_test = np.concatenate(model_fold_targets[name])
        all_y_prob = np.concatenate(model_fold_probabilities[name])
        
        fpr, tpr, _ = roc_curve(all_y_test, all_y_prob)
        
        mean_auc = mean_auc_scores[name]
        std_auc = np.std(cv_auc_scores[name])
        
        # Build legend string specifying metric mean and fold variation boundaries
        legend_label = f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
        plt.plot(fpr, tpr, lw=2, label=legend_label)
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    
    plt.legend(loc="lower right", frameon=True, prop={'weight': 'bold', 'size': 10})
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig('results/top_5_models_roc_curve.png', dpi=300)
    plt.show()

    # ── 3. Aggregated Out-Of-Fold Confusion Matrix Plot ─────────────────────────
    print(f"Generating Confusion Matrix for '{best_model_name}'...")
    best_cm = confusion_matrix(y, oof_predictions[best_model_name])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    sns.despine(top=True, right=True, left=False, bottom=False)
    
    plt.tight_layout()
    plt.savefig('results/best_model_confusion_matrix.png', dpi=300)
    plt.show()

    # ── 4. SHAP Summary Interpretation Workflow ─────────────────────────────────
    print(f"\nComputing SHAP values for global interpretation of '{best_model_name}'...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    best_estimator.fit(X_scaled_df, y)

    fig_shap = plt.figure(figsize=(10, 6))
    
    try:
        if "Stacking" in best_model_name or "Voting" in best_model_name:
            explainer = shap.Explainer(best_estimator.predict, X_scaled_df)
            shap_values = explainer(X_scaled_df)
        elif hasattr(best_estimator, "predict_proba"):
            try:
                explainer = shap.Explainer(best_estimator, X_scaled_df)
                shap_values = explainer(X_scaled_df)
            except Exception as inner_e:
                if "additivity check failed" in str(inner_e).lower():
                    print("⚠️ Native additivity check failed. Retrying with check_additivity=False...")
                    explainer = shap.Explainer(best_estimator, X_scaled_df)
                    shap_values = explainer(X_scaled_df, check_additivity=False)
                else:
                    raise inner_e
            
            if isinstance(shap_values, list) or (len(shap_values.shape) == 3 and shap_values.shape[-1] == 2):
                shap_values = shap_values[..., 1] if hasattr(shap_values, "shape") else shap_values[1]
        else:
            explainer = shap.Explainer(best_estimator.predict, X_scaled_df)
            shap_values = explainer(X_scaled_df)
            
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        
        sns.despine(top=True, right=True)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/best_model_shap_summary.png', dpi=300)
        print("🎉 Successfully saved SHAP global analysis plot to results/best_model_shap_summary.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ SHAP execution completely bypassed due to unresolvable error: {e}")

if __name__ == '__main__':
    evaluate_models('data/processed_data.csv')