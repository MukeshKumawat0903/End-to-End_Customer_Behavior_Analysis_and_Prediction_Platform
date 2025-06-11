# 🛒 End-to-End Customer Behavior Analysis & Prediction Platform

**A Modular E-commerce Machine Learning Platform for Customer Targeting, Explainable AI, and Business Impact**

---

## 🚀 Overview

This project implements a **comprehensive, production-ready machine learning pipeline** to analyze customer behavior and predict purchase intent in an e-commerce setting. It combines advanced feature engineering, robust model evaluation, and explainable AI to deliver actionable insights for customer segmentation, personalized targeting, and business strategy.

The platform is designed for clarity, reproducibility, and business relevance—integrating data science best practices and industry-standard reporting.

---

## 📊 Key Features

* **End-to-End ML Pipeline:**
  Covers data ingestion, preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, and visualization.
* **Advanced Evaluation Suite:**
  OOF/Test ROC, PR, confusion matrices, calibration curves, lift/gain charts, and more.
* **Explainable AI:**
  Global (bar, beeswarm) and local (force) SHAP explanations for trust, compliance, and business alignment.
* **Interactive Dashboard:**
  Visual insights for technical and business users (Streamlit).
* **Business Focus:**
  Metrics and charts mapped to ROI, customer value, and strategic targeting.
* **Modular & Reusable:**
  Clean code, reusable classes (`ModelEvaluator`, `PlotHelper`, `MetricsHelper`), and ready for production or further research.

---

## 📂 Project Structure

```
project_root/
├── app/
│   └── streamlit/                # Dashboards & reports
├── data/                         # Raw & processed data
├── models/                       # Trained model artifacts
├── notebooks/                    # EDA, experiments, deep dives
├── src/
│   ├── data/             # Feature engineering, pipelines
│   ├── models/           # Training scripts & wrapper 
│   └── visualization/    # Plotting & SHAP utilities
├── README.md
└── requirements.txt
```

---

## 🖼️ Visual Highlights

![Dashboards Overview](assets/Dashboards_Overview.png)

| Chart/Section          | Example Screenshot                                              |
| ---------------------- | ---------------------------------------------------             |
| Correlation Matrix     | ![Correlation Matrix](assets/Feature%20Correlation%20Matrix.png) |
| Confusion Matrix       | ![Confusion Matrix](assets/Test%20Confusion%20Matrix.png)       |
| ROC & PR Curves        | ![ROC/PR](assets/Test%20ROC%20and%20PR%20Curve.png)             |
| Calibration Curve      | ![Calibration](assets/Calibration%20for%20OOF%20%26%20Test.png) |
| Lift/Gain Chart        | ![Lift/Gain](assets/Test%20Lift%20and%20Gain%20Charts.png)      |
| Global SHAP Importance | ![SHAP Bar](assets/Global%20Feature%20Importance.png)           |
| SHAP Beeswarm          | ![SHAP Beeswarm](assets/Feature%20Impact%20Distribution.png)    |
| SHAP Force Plot        | ![SHAP Force](assets/SHAP%20force%20plot.png)                   |
| Threshold vs Metrics   | ![Threshold](assets/OOF%20Threshold%20vs%20Matrix.png)          |

---
**Note:** Additional plots are available in the `/assets` folder of the GitHub repository.

## 📈 Example Results & Interpretation

| Metric            | OOF            | Test           | Business Insight                   |
| ----------------- | -------------- | -------------- | ---------------------------------- |
| **AUC-ROC**       | \~0.95         | \~0.95         | Excellent customer distinction     |
| **F1-opt Thresh** | 0.45           | 0.45           | Balanced precision-recall          |
| **Brier Score**   | 0.086          | 0.087          | Probabilities are well-calibrated  |
| **Initial Lift**  | 2.4x           | 2.4x           | Top scores 2.4× better than random |
| **Key Feature**   | purchase\_freq | purchase\_freq | Highest SHAP impact                |

**Interpretability:**

* **SHAP force plots** show exactly why the model labels a customer as a target/non-target.
* **Threshold analysis** visualizes precision/recall/lift tradeoffs at every probability cutoff.
* **Calibration curves** demonstrate that probability outputs can be trusted as real risk/propensity.

---

## 💡 What Makes This Project Stand Out

* **Thorough, real-world evaluation:**

  * OOF and test set metrics closely match, showing generalization (no overfitting).
  * Rich suite of diagnostics: ROC/PR, calibration, confusion, lift/gain, threshold tuning.
  * High AUC-ROC and PR curves demonstrate strong discriminative power, even under class imbalance.
  * Calibration curves and Brier scores confirm model probability outputs are reliable.
  *  Business metrics (lift, gain, cumulative capture) are highlighted.
* **Explainable AI (XAI):**
  
  * SHAP bar/beeswarm/force plots for global & per-customer explanation.
  * It enable transparent model decisions—crucial for business trust and actionable feedback.
  * Easy to audit, debug, and communicate model logic.
* **Production-minded, modular design:**

  * Clean separation of preprocessing, modeling, and evaluation.
  * Ready for dashboarding or integration in larger MLOps workflows.
* **Business alignment:**

  * Focused on ROI, targeting, customer segmentation, and clear decision support.
  * Clear trade-offs between precision/recall, actionable targeting based on model lift.
  * Data-driven feature importance and individual explanations for customer actions.

---

## 🧑‍💻 How To Run

1. **Install dependencies:**
   `pip install -r requirements.txt`

2. **Launch dashboard:**
   `streamlit run app/streamlit/dashboard.py`

---

## 📊 Core Evaluation and Visualization

* **Confusion Matrix (OOF & Test):**

  * Shows high recall and strong overall accuracy, with transparent tradeoffs between false positives/negatives.
* **ROC and PR Curves:**

  * Both OOF and Test curves show robust discrimination, minimal overfitting, and stable performance under class imbalance.
* **Calibration Curves & Brier Scores:**

  * Model probabilities are reliable and interpretable for decision making.
* **Lift & Gain Charts:**

  * The model enables business to target the top-scoring customers, achieving up to 2.4x gain over random outreach.
* **Threshold Tuning Curve:**

  * Optimal cutoff selection is visualized for F1, J-statistic, precision, recall.
* **SHAP Explainability:**

  * **Global:** Key drivers of predictions identified for feature selection and business strategy.
  * **Local:** Each individual decision is interpretable (force plot), supporting transparency and trust.

---

## 🏆 Why This Project is Valuable (for Employers & Teams)

* **Demonstrates full-stack ML skillset:** Data wrangling, modeling, tuning, deployment, and explanation.
* **Showcases advanced evaluation and reporting**—far beyond simple accuracy.
* **Direct business impact:** Model outputs support ROI, targeted campaigns, and customer understanding.
* **Built for production:** Modular codebase, reusable components, and real-time dashboard compatibility.
* **Best practices in explainable AI:**

  * Supports regulatory requirements and business transparency.

---

## 📌 Next Steps & Extensions

* Add user-level personalization and A/B testing
* Deploy dashboard to Streamlit Cloud, AWS, or similar
* Integrate with real-time pipelines for dynamic scoring
* Expand feature engineering and experiment tracking

---

## 🙌 Acknowledgements

Built using industry best practices in data science, MLOps, and open-source analytics.
Inspired by open-source e-commerce analytics and the broader ML community.

---
For more details, or to see the code/notebooks, reach out or visit my GitHub portfolio!

---