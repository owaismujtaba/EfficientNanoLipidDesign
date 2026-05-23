# Efficient Nanoprticle Design using Machine Learning

A robust, end-to-end Python pipeline designed to evaluate, rank, and interpret a diverse collection of machine learning classifiers (including tree ensembles, linear models, neural networks, and stacking/voting variants). 

The framework handles class imbalance dynamically, evaluates models using **5-Fold Stratified Cross-Validation**, and generates publication-quality performance visualizations along with global feature importance analysis via SHAP.

---

## 🚀 Key Features

* **Comprehensive Model Zoo:** Automated evaluation of 16 classifiers including Logistic Regression, LightGBM, XGBoost, CatBoost, SVMs, and Ensemble methods (Stacking & Voting).
* **Robust Evaluation Workflows:** Implements `StratifiedKFold` cross-validation to ensure stable, reliable performance estimates on imbalanced datasets.
* **Auto-Imbalance Handling:** Computes minority/majority sample ratios on the fly and dynamically injects target weights (`scale_pos_weight`, `class_weight='balanced'`) into matching estimators.
* **Metric-Driven Selection:** Ranks architectures automatically based on mean cross-validated **Recall** to safeguard performance on critical minority targets.
* **Publication-Ready Graphics:** Automatic generation of clean, dual-axis optimized plots matching strict academic submission rules (font weights, layout constraints, and axis despinning).

---

## 📊 Automated Output Visualizations

The script processes data and outputs four analytical assets to the `results/` directory:

1. **`model_metrics_comparison.png`**: A grouped bar chart tracking Precision, Recall, MCC, and ROC-AUC across all models, fully equipped with fold-based standard deviation error bars ($sd$).
2. **`top_5_models_roc_curve.png`**: Multi-model ROC curves displaying performance trajectories for the top 5 models sorted by mean AUC, displaying mean and standard deviation directly inside the legend labels (`Mean ± SD`).
3. **`best_model_confusion_matrix.png`**: An aggregated, clear out-of-fold confusion matrix heatmap tracking absolute validation classification footprints for the top-performing model.
4. **`best_model_shap_summary.png`**: A SHAP beeswarm plot illustrating global feature importance distributions and showing how feature values drive individual risk/prediction probabilities.

---

## 🛠️ Project Structure

```text
├── data/
│   └── processed_data.csv       # Preprocessed input data containing a 'target' column
├── results/                     # Automatically generated on runtime
│   ├── fold_metrics.csv         # Raw metrics broken down slice-by-slice per fold
│   ├── summary_results.csv      # Formatted string report showcasing metric means and SDs
│   ├── model_metrics_comparison.png
│   ├── top_5_models_roc_curve.png
│   ├── best_model_confusion_matrix.png
│   └── best_model_shap_summary.png
├── preprocess.py        # Core execution script
├── train.py # train the models
└── README.md                    # Project documentation