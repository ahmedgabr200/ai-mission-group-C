# Loan Default Risk: Causal Inference, Borrower Segmentation, and Supervised Prediction

**Course:** Data and AI in Economics
**University:** TU Dortmund University — Centre of Finance, Risk & Resource Management, Department of Business and Economics
**Supervision:** Prof. Dr. Peter N. Posch, Kira Schönhütte
**Authors:** Ahmed, Mahmoud, Mohamed
**Session:** Credit Risk, Consumer Behavior & Sports Analytics

---

## 1. Research Question

**Does a higher debt-to-income (DTI) ratio causally increase the probability of loan default?**

This project goes beyond a standard predictive classification task by anchoring the analysis in an explicit economic mechanism and testing it through a combination of causal inference, unsupervised borrower segmentation, and supervised prediction.

### Hypotheses

- **H1:** Borrowers with a high DTI ratio have a higher probability of default.
- **H2:** The causal effect of high DTI differs across borrower segments.
- **H3:** Borrower segments identified through clustering improve the economic interpretability of loan-default risk.

## 2. Economic Motivation

A high debt-to-income ratio means that a larger share of a borrower's income is already committed to debt repayment. This creates liquidity pressure: when a large share of income is already allocated to repayment, negative shocks such as job loss, rising interest rates, or unexpected expenses are more likely to push the borrower into default. Understanding whether this relationship is causal — not merely correlational — is relevant for:

- Banks designing responsible lending rules,
- Regulators concerned with borrower over-indebtedness,
- Credit-risk teams seeking to distinguish causal default drivers from spurious correlations,
- Transparency requirements in automated lending decisions.

## 3. Dataset

| Field | Value |
|---|---|
| Dataset name | Loan Default Prediction Dataset |
| Author / Publisher | NIKHIL |
| Year | 2023 |
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/nikhil1e9/loan-default/data |
| License | Creative Commons CC0 1.0 Universal Public Domain Dedication |
| Rows | 255,347 |
| Variables | 18 |
| Target variable | `Default` |
| Treatment variable | `DTIRatio` → `HighDTI` (top quartile indicator) |

### Dataset Citation

> NIKHIL (2023). *Loan Default Prediction Dataset.* Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default/data. License: CC0 1.0 Universal Public Domain Dedication.

### Data Quality

The notebook documents each identified data quality issue together with its concrete mitigation: missing values (median / most-frequent imputation), class imbalance (stratified split, class weighting, optional SMOTE), outliers (IQR-based clipping for clustering features), categorical encoding (one-hot encoding inside the preprocessing pipeline), scale differences (standardization prior to logistic regression, PCA, and K-Means), and data leakage (the `LoanID` identifier is dropped before modeling).

## 4. Methodology

The project integrates three method blocks that inform one another rather than running in isolation:

### 4.1 Causal Inference

- **Treatment:** `HighDTI` (1 if `DTIRatio` is at or above the 75th percentile)
- **Outcome:** `Default`
- **Identification strategy:** Backdoor adjustment over a directed acyclic graph (DAG) of observed borrower characteristics (income, age, credit score, employment duration, loan amount, interest rate, loan term, number of credit lines, education, employment type, marital status, mortgage status, dependents, loan purpose, co-signer status).
- **Estimation:** A logistic-regression-based backdoor adjustment (g-computation via `statsmodels`) is the primary estimator used to compute the Average Treatment Effect (ATE). A DoWhy specification of the same causal graph is also included in the notebook to formally represent the identification strategy.
- **Identifying assumptions:** Unconfoundedness, overlap, and SUTVA are stated explicitly, along with a discussion of their plausibility given the dataset.

### 4.2 Unsupervised Learning — Borrower Segmentation

- Numeric features are standardized and reduced via PCA (90% variance retained) prior to clustering.
- **K-Means** is used, with the choice justified against DBSCAN (less stable in high-dimensional standardized data) and hierarchical clustering (computationally expensive at this sample size).
- **k-selection:** Both the elbow method (inertia) and silhouette-score analysis are used to select the number of clusters.

### 4.3 Supervised Learning — Default Prediction

- **Train/test split:** 80/20 stratified split to preserve class proportions.
- **Cross-validation:** 5-fold stratified cross-validation.
- **Class imbalance handling:** `class_weight='balanced'` for Logistic Regression and Random Forest, and an additional SMOTE-resampled Logistic Regression specification.
- **Models:** Logistic Regression (balanced, and with SMOTE) and Random Forest (balanced).
- **Metrics:** Recall, Precision, F1-score, ROC-AUC, and Average Precision (PR-AUC), since accuracy alone is not informative under class imbalance.

### 4.4 Methodological Synthesis

The three blocks are explicitly connected: default rates and borrower profiles are compared across clusters; the causal effect of `HighDTI` is re-estimated separately within each cluster to test for heterogeneity; and cluster membership is tested as an additional feature in the supervised prediction models.

## 5. Results

- **Causal effect:** The estimated ATE of `HighDTI` on default probability is **+0.88 percentage points**, after adjustment for observed borrower characteristics.
- **Borrower segments:** K-Means (k=2) identifies a financially constrained segment (Cluster 0: 16.43% default rate, lower income, high loan-to-income ratio) and a financially stable segment (Cluster 1: 9.08% default rate, higher income, lower leverage).
- **Heterogeneous effects:** The HighDTI effect is positive in both segments (Cluster 0: 0.0092; Cluster 1: 0.0085).
- **Best predictive model:** Logistic Regression (balanced) achieves the strongest overall performance — ROC-AUC of 0.7615 and recall of 69.8% — outperforming the SMOTE variant and Random Forest on these dimensions, while Random Forest achieves higher precision and F1-score.
- **Role of clustering in prediction:** Adding cluster membership as a predictive feature does not change model metrics numerically; its contribution is to economic interpretability rather than predictive accuracy.

Full results, tables, and figures are available in the notebook (Section 11) and the accompanying presentation.

## 6. Repository Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── Loan_default.csv
├── LICENSE                    # MIT License (code) — see note on dataset license
├── .gitignore                 # Python / Jupyter ignore rules
├── Final_Notebook.ipynb       # Final, fully executed Jupyter Notebook (proposal + submission)
├── Presentation.pdf           # Final presentation slides (PDF, oral-exam format)

```

## 7. Installation

```bash
git clone https://github.com/<org-or-user>/dai-mission-group-A.git
cd dai-mission-group-A
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Required Packages

See `requirements.txt`. Core dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`, `imbalanced-learn`, `networkx`, `dowhy`, `jupyter`.

## 9. How to Run the Notebook

1. Ensure `Loan_default.csv` is available in the `data/` folder (see Data Handling note below).
2. Activate the virtual environment and install dependencies as above.
3. Launch Jupyter: `jupyter notebook Final_Notebook.ipynb`
4. Run all cells in order (`Kernel → Restart & Run All`) to reproduce all outputs, tables, and figures.

**Data handling note:** Per the course's GitHub size limits, if `Loan_default.csv` is under 100 MB it is included directly in `data/`. If the raw file exceeds 100 MB, it is not pushed to GitHub; instead, the notebook downloads it automatically via script, or a secure Sciebo link is provided here: `[INSERT SCIEBO LINK IF APPLICABLE]`.

## 10. References

1. NIKHIL (2023). *Loan Default Prediction Dataset.* Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default/data. License: CC0 1.0 Universal Public Domain Dedication.
2. Sharma, A., & Kiciman, E. (2020). *DoWhy: An End-to-End Library for Causal Inference.* arXiv preprint.
3. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
5. Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.

---

*Submitted as part of the Data and AI in Economics course, TU Dortmund University.*
