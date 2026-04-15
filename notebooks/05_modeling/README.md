# 05 Modeling Notebooks — Simple Explanation

This file explains the three notebooks in `notebooks/05_modeling`: `01_baseline_model.ipynb`, `02_experiments.ipynb`, and `03_final_model_evaluation.ipynb`.
The explanations are written in easy terms, with clear reasons for each step, each graph, and why the team chose XGBoost.

## Why this modeling stage exists

The modeling notebooks are the part of the project where we take the cleaned and engineered data and turn it into a churn prediction model.
The goal is to build a model that predicts whether a customer will churn within 4 weeks after renewal.

The modeling stage has three main steps:
1. Build a first baseline model.
2. Try different settings to find a better model.
3. Finalize the best model, explain its behavior, and create the final prediction file.

---

## Why XGBoost was chosen

XGBoost is the model used across all three notebooks. We chose it because:
- It works very well with tabular data (structured data with columns).
- It handles class imbalance well. In this project, only about 9-10% of customers churn, so the model needs help not to ignore churners.
- It gives useful feature importance information.
- It is fast enough and easy to save/load.
- It handles missing or noisy features more easily than many simple models.

In simple words: XGBoost is a strong default model for this type of churn prediction problem.

---

## Notebook 05.01 — Baseline Model

### What this notebook does

This notebook builds the first version of the churn model using default settings.
The baseline model helps us understand what a simple XGBoost model can do before we try to improve it.

### Main steps

1. Load data and clean it.
   - It loads raw data from `src/data/loader.py` and cleans it with `src/data/cleaner.py`.
   - It also loads pre-made feature tables from `data/features/train_2024.parquet` and `data/features/test_2025.parquet`.

2. Prepare training and test datasets.
   - `train_2024.parquet` is used for model training.
   - `test_2025.parquet` is used for final evaluation.
   - Only rows with a known churn label are kept in the test set.

3. Check class imbalance.
   - The notebook prints how many customers churned and how many stayed.
   - It also prints the ratio used by `scale_pos_weight`.
   - This is important because churn is much less common than retention.

4. Run 5-fold cross-validation.
   - Cross-validation splits the training data into 5 parts.
   - The model trains on 4 parts and tests on the 5th part, five times.
   - This gives an idea of how the model performs on unseen data.

5. Train the final baseline model.
   - The model is trained on the full 2024 training data.
   - The 2025 test set is used as a validation set for early stopping.
   - The trained model is saved to `models_saved/churn_model_v1_baseline.json`.

6. Evaluate on the test set.
   - The notebook calculates precision, recall, F1, PR-AUC, ROC-AUC.
   - It prints a clear score summary.

7. Show feature importance.
   - It plots the top features used by the model.
   - This shows which information the model thinks is most useful.

8. Analyze thresholds.
   - It plots how precision, recall, and F1 change when the decision threshold changes.
   - This helps the business choose how many customers to flag.

9. Check calibration.
   - It draws a plot that compares predicted probabilities to actual churn rates.
   - This shows whether the model is overconfident or underconfident.

### Why each graph is shown

- **Feature importance plot**: Shows which features are most important. This is helpful to understand what the model uses.
- **Threshold analysis plot**: Shows how score changes if we make the model more or less strict. This helps decide the best threshold for business use.
- **Calibration plot**: Shows whether predicted risk scores match real churn rates. Good calibration means the probabilities are trustworthy.

### What is the baseline output

- A saved model file: `models_saved/churn_model_v1_baseline.json`
- A saved parameter file: `models_saved/churn_model_v1_baseline_params.json`
- Plots saved in `reports/figures/`
- Printed metrics for PR-AUC, ROC-AUC, precision, recall, and F1

---

## Notebook 05.02 — Model Experiments

### What this notebook does

This notebook tries to improve the baseline model.
It tests different XGBoost hyperparameters and checks if removing weak features helps.

### Main steps

1. Load the same training and test data as the baseline notebook.
2. Run experiments in this order:
   - Experiment 1: Try different `scale_pos_weight` values.
   - Experiment 2: Try different tree depths (`max_depth`).
   - Experiment 3: Try different learning rates and number of trees.
   - Experiment 4: Try dropping weak features.
   - Experiment 5: Combine the best settings and train the final tuned model.

### Why these experiments are important

- `scale_pos_weight`: Because churn is rare, this parameter tells XGBoost to pay more attention to churn examples.
- `max_depth`: Controls the complexity of each tree. Too low and the model is too simple; too high and it overfits.
- `learning_rate` and `n_estimators`: These control how fast the model learns and how many trees it builds.
- Feature selection: Removing weak features can make the model simpler and sometimes more stable.

### What the notebook prints and saves

- Results for each `scale_pos_weight` value.
- The best `scale_pos_weight` and the matching PR-AUC score.
- The best tree depth.
- The best learning rate and number of trees.
- Comparison of full features vs reduced feature set.
- The tuned final model saved as `models_saved/churn_model_v2_tuned.json`.
- The tuned model evaluation plot saved in `reports/figures/`.
- Lists of final features and final parameters saved to `data/features/final_feature_cols.json` and `models_saved/best_params.json`.

### Why the experiment graph is shown

- **PR-AUC vs `scale_pos_weight` plot**: Shows which imbalance weight gives the best performance. It helps choose the best balance between catching churners and not overfitting.

### What is the experiment output

- A better tuned XGBoost model.
- The tuned model evaluation metrics.
- A finalized feature list.
- Saved model and parameter files.

---

## Notebook 05.03 — Final Model Evaluation

### What this notebook does

This notebook creates the final business-ready output.
It takes the tuned model, evaluates it fully, explains predictions with SHAP, and scores every 2025 customer.

### Main steps

1. Load the best feature list and best parameters from the experiment notebook.
2. Load the same 2024 training and 2025 test data.
3. Retrain the final model on the full training set using the best settings.
4. Evaluate the final model on the test set and save plots.
5. Check calibration again with the final model.
6. Plot final feature importance.
7. Use SHAP to explain model predictions.
8. Score all 2025 customers and assign a risk tier.
9. Save the final prediction CSV.

### Why SHAP is used

- SHAP explains individual predictions.
- It tells us why a particular customer has a high churn score.
- This is useful for the customer success team because they need reasons, not just a number.
- SHAP gives both global importance and customer-level explanations.

### Why this final output is important

- The final CSV `reports/churn_predictions_2025.csv` is the real deliverable.
- It includes:
  - `Co_Ref` (customer reference)
  - `churn_probability` (0 to 1)
  - `risk_tier` (Critical / High / Medium / Low)
  - `churn_flag` (whether the customer is likely churned)
  - `top_reasons` (top reasons from SHAP)
- This file is meant to help the business outreach team decide who to contact.

### What graphs and plots are shown

- **Final evaluation plots**: Show performance again for the final model.
- **Calibration plot**: Ensures the final model is still well calibrated.
- **Feature importance plot**: Shows which features the final model uses most.
- **SHAP summary bar and beeswarm plots**: Show which features matter most across all customers.
- **SHAP waterfall plots for top customers**: Show the exact reasons one customer is scored as high risk.

### What is the final output

- `models_saved/churn_model_final.json`
- `reports/churn_predictions_2025.csv`
- SHAP plots in `reports/figures/`
- Final evaluation metrics printed in the notebook

---

## Simple explanation script for all modeling notebooks

This section is a short script-like summary of what all three notebooks do.
It is written as plain steps so anyone can follow the logic.

```python
# Simple script-like explanation for 05 modeling notebooks

def explain_modeling_workflow():
    print('1. Load cleaned data and prebuilt feature tables.')
    print('2. Build a first baseline XGBoost model with default settings.')
    print('3. Evaluate the baseline model on held-out 2025 data.')
    print('4. Try better settings in experiments:')
    print('   - Adjust class imbalance weight')
    print('   - Adjust tree depth')
    print('   - Adjust learning rate and number of trees')
    print('   - Remove weak features if that helps')
    print('5. Save the best tuned model and best feature list.')
    print('6. Retrain the final model with the best settings.')
    print('7. Evaluate again and make sure final results are stable.')
    print('8. Use SHAP to explain why the model makes each prediction.')
    print('9. Score every 2025 customer and save the prediction file.')
    print('10. The final file is used by the business team.')

if __name__ == '__main__':
    explain_modeling_workflow()
```

---

## Final summary in simple terms

- `05.01` builds the first model and shows what a starting performance looks like.
- `05.02` makes the model better by testing different settings and removing weak features.
- `05.03` creates the final business output with explanations and customer scores.

All three notebooks use XGBoost because it is strong for this kind of customer churn data, especially when churn is rare.

If you want, this README can also be copied into the main project README so the whole project has a modeling explanation section.