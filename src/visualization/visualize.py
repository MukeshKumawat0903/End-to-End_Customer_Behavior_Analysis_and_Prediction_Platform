# 🔧 Built-in libraries
import logging
from typing import Any, Dict, List, Optional, Tuple

# 📊 Data and computation
import numpy as np
import pandas as pd

# 🧠 Scikit-learn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
    log_loss, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve

# 📈 Visualization
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# 🌐 Streamlit
import streamlit as st
import plotly.express as px

# 📁 Internal modules
from src.models.evaluate import MetricsHelper


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:
    # Console handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # File handler (log output to a file)
    file_handler = logging.FileHandler("app.log")
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# Type aliases for better readability
DataType = Any

class EDAHelper:
    @staticmethod
    def plot_event_type_distribution(df):
        fig = px.histogram(df, x='event_type', title='Event Type Distribution')
        fig.update_layout(bargap=0.2)
        return fig
    
    @staticmethod
    def plot_top_main_categories(df):
        top_categories = df['main_category'].value_counts().nlargest(10)

        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            title='Top 10 Product Categories',
            labels={'x': 'Count', 'y': 'Main Category'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig
    
    @staticmethod
    def plot_top_subcategories(df):
        top_subcategories = df['category_code'].value_counts().nlargest(10)
        fig = px.bar(
            x=top_subcategories.values,
            y=top_subcategories.index,
            orientation='h',
            title='Top 10 Subcategories',
            labels={'x': 'Count', 'y': 'Subcategory'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig
    
    @staticmethod
    def plot_hourly_user_activity(df):
        fig = px.histogram(df, x='hour', nbins=24, title='Hourly User Activity')
        fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Activity Count')
        return fig

    @staticmethod
    def plot_daily_unique_visitors(df):
        df['event_date'] = pd.to_datetime(df['event_time']).dt.date
        visitor_by_date = df.groupby('event_date')['user_id'].nunique().sort_index()
        
        fig = px.line(
            x=visitor_by_date.index,
            y=visitor_by_date.values,
            labels={'x': 'Date', 'y': 'Number of Unique Visitors'},
            title='Daily Unique Visitors'
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    @staticmethod

    def get_correlation_heatmap_fig(df, title="Feature Correlation Heatmap"):
        """
        Generate a Plotly correlation heatmap figure from a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing numeric features.
        - title (str): Title to display on the heatmap.

        Returns:
        - fig (plotly.graph_objs._figure.Figure): Plotly heatmap figure.
        """
        # Compute correlation matrix
        corr_matrix = df.corr(numeric_only=True)

        if corr_matrix.empty:
            return None  # Or raise an exception if preferred

        # Create heatmap figure
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title=title
        )
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        return fig
    

class PlotHelper:
    @staticmethod
    def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):
        """
        Returns a Plotly heatmap for a confusion matrix.
        """
        heatmap = go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        )

        fig = go.Figure(data=heatmap)
        fig.update_layout(
            title_text=title,
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            width=500,
            height=400
        )
        return fig

    @staticmethod
    def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
        """
        Returns a Plotly figure for ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray"))
        )

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=400
        )
        return fig

    @staticmethod
    def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve"):
        """
        Returns a Plotly figure for precision-recall curve.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))

        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=500,
            height=400
        )
        return fig

    @staticmethod
    def plot_calibration_curve(y_true, y_pred_proba, model_name="Model", data_type="Data", n_bins=10):
        """
        Returns a Plotly calibration curve with Brier score annotation.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_pred_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode="lines+markers",
            name="Calibration Curve",
            line=dict(color="teal"),
            marker=dict(symbol='circle', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Perfect Calibration", line=dict(color="gray", dash="dash")
        ))

        fig.update_layout(
            title=f"Calibration Curve - {data_type} ({model_name})<br>Brier Score: {brier:.4f}",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            width=600,
            height=450,
            legend=dict(x=0.02, y=0.98),
        )
        return fig

    @staticmethod
    def plot_threshold_metrics(
        y_true,
        y_pred_proba,
        model_name='Model',
        data_type='Data',
        beta=1.0,
        optimal_metric='F1'  # Options: 'F1', 'J-stat', 'Precision', 'Recall'
    ):
        """
        Plots threshold vs. various metrics (F1, Precision, Recall, J-statistic) using Plotly,
        and highlights the optimal threshold based on the selected metric.
        """

        thresholds = np.linspace(0.01, 0.99, 99)
        precision, recall, f1, j_stat = [], [], [], []

        for thresh in thresholds:
            preds = (y_pred_proba >= thresh).astype(int)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            f = f1_score(y_true, preds, zero_division=0) if beta == 1.0 else \
                (1 + beta**2) * (p * r) / ((beta**2 * p) + r + 1e-10)
            j = r + p - 1

            precision.append(p)
            recall.append(r)
            f1.append(f)
            j_stat.append(j)

        df = pd.DataFrame({
            'Threshold': thresholds,
            'Precision': precision,
            'Recall': recall,
            f'F{beta:.1f}-Score': f1,
            'J-Statistic': j_stat
        })

        # Determine the optimal threshold
        metric_map = {
            'F1': f'F{beta:.1f}-Score',
            'J-stat': 'J-Statistic',
            'Precision': 'Precision',
            'Recall': 'Recall'
        }
        metric_column = metric_map.get(optimal_metric, f'F{beta:.1f}-Score')
        best_idx = df[metric_column].idxmax()
        best_threshold = df.loc[best_idx, 'Threshold']
        best_score = df.loc[best_idx, metric_column]

        # Plot all metrics
        fig = go.Figure()

        for col in df.columns[1:]:
            fig.add_trace(go.Scatter(x=df['Threshold'], y=df[col], mode='lines', name=col))

        # Add the optimal point
        fig.add_trace(go.Scatter(
            x=[best_threshold],
            y=[best_score],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            text=[f"Best {optimal_metric}: {best_threshold:.2f}"],
            textposition='top center',
            name=f'Optimal {optimal_metric}'
        ))

        fig.update_layout(
            title=f'Threshold vs Metrics - {data_type} ({model_name})',
            xaxis_title='Threshold',
            yaxis_title='Score',
            width=800,
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        return fig

    @staticmethod
    def plot_lift_gain_chart(y_true, y_pred_proba, model_name='Model', data_type='Validation'):
        """
        Plot Lift and Gain charts using Plotly.
        """

        # Create dataframe
        df = pd.DataFrame({
            'y_true': y_true,
            'y_proba': y_pred_proba
        })

        # Sort by predicted probability descending
        df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
        df['cum_total'] = np.arange(1, len(df) + 1)
        df['cum_positive'] = df['y_true'].cumsum()
        total_positives = df['y_true'].sum()

        # Gain = % of total positive captured at each percentile
        df['gain'] = df['cum_positive'] / total_positives

        # Lift = Gain / random expectation
        df['lift'] = df['gain'] / (df['cum_total'] / len(df))

        # Percentiles (x-axis)
        df['percentile'] = df['cum_total'] / len(df)

        # Plot Gain Chart
        gain_fig = go.Figure()
        gain_fig.add_trace(go.Scatter(
            x=df['percentile'], y=df['gain'],
            mode='lines', name='Gain', 
            line=dict(color='blue')
        ))
        gain_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Baseline', line=dict(color='gray', dash='dash')
        ))
        gain_fig.update_layout(
            title=f'Gain Chart - {data_type} ({model_name})',
            xaxis_title='Proportion of Data (percentile)',
            yaxis_title='Cumulative Gain',
            width=800,
            height=500
        )

        # Plot Lift Chart
        lift_fig = go.Figure()
        lift_fig.add_trace(go.Scatter(
            x=df['percentile'], y=df['lift'],
            mode='lines', name='Lift', 
            line=dict(color='green'),
        ))
        lift_fig.add_trace(go.Scatter(
            x=[0, 1], y=[1, 1],
            mode='lines', name='Baseline', line=dict(color='gray', dash='dash')
        ))
        lift_fig.update_layout(
            title=f'Lift Chart - {data_type} ({model_name})',
            xaxis_title='Proportion of Data (percentile)',
            yaxis_title='Lift',
            width=800,
            height=500
        )
        return gain_fig, lift_fig


class ModelEvaluator:
    def __init__(self, model: Any, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.oof_predictions = None
        self.y_train_oof = None
        self.trained_models = []

    def evaluate_oof(self, threshold: float = 0.5, class_labels: List[str] = ["0", "1"]):
        y_pred = (self.oof_predictions >= threshold).astype(int)
        cm = confusion_matrix(self.y_train_oof, y_pred)
        metrics = MetricsHelper.compute_metrics(self.y_train_oof, y_pred, self.oof_predictions)
        print("\nOOF Evaluation Metrics:", metrics)
        
        fig1 = PlotHelper.plot_confusion_matrix(cm, class_labels, "OOF Confusion Matrix")
        fig2 = PlotHelper.plot_roc_curve(self.y_train_oof, self.oof_predictions)
        fig3 = PlotHelper.plot_pr_curve(self.y_train_oof, self.oof_predictions)

        return fig1, fig2, fig3

    def evaluate_test_set(self, y_test, y_proba, y_pred, class_labels: List[str] = ["0", "1"]):

        cm = confusion_matrix(y_test, y_pred)

        try:
            fig1 = PlotHelper.plot_confusion_matrix(cm, class_labels, "Test Confusion Matrix")
            fig2 = PlotHelper.plot_roc_curve(y_test, y_proba)
            fig3 = PlotHelper.plot_pr_curve(y_test, y_proba)

        except Exception as e:
            print(f"Error in confusion matrix plotting: {e}")

        return fig1, fig2, fig3
    
    def plot_threshold_metrics(self, y_true, y_pred_proba, data_type="OOF Data", beta=1.0, optimal_metric='F1'):
        """
        Plots threshold vs. various metrics (F1, Precision, Recall, J-statistic) using Plotly,
        and highlights the optimal threshold based on the selected metric.
        """
        fig = PlotHelper.plot_threshold_metrics(
            y_true=y_true, y_pred_proba=y_pred_proba, 
            model_name=self.model_name, data_type=data_type,
            beta=beta, optimal_metric=optimal_metric  # You can change to 'F1', 'J-stat', 'Precision', etc.
        )
        return fig

    def plot_calibration_curve(self, y_train_oof, oof_predictions, y_true, y_prob, data_type=["OOF Data", "Test Data"]):
       """Plots calibration curve for OOF and test data."""
       fig1 =  PlotHelper.plot_calibration_curve(y_train_oof, oof_predictions, data_type[0])
       fig2 =  PlotHelper.plot_calibration_curve(y_true, y_prob, data_type[1])
       return fig1, fig2

    def plot_lift_gain_chart(self, y_true, y_scores, data_type="Data"):
        """Plots Lift and Gain charts for the model's predicted probabilities."""
        gain_fig, lift_fig = PlotHelper.plot_lift_gain_chart(y_true, y_scores, data_type)

        return gain_fig, lift_fig

def plot_rfm_segments(rfm: pd.DataFrame):
    """
    Given an RFM DataFrame with columns ['recency', 'frequency', 'monetary', 'rfm_segment', ...],
    returns a list of Plotly figures for dashboard display.
    """
    rfm_scaled = rfm.copy()
    # Only scale if not already scaled
    scaler = StandardScaler()
    rfm_scaled[['recency', 'frequency', 'monetary']] = scaler.fit_transform(
        rfm_scaled[['recency', 'frequency', 'monetary']]
    )

    # Plot 1: Frequency vs. Monetary
    fig1 = px.scatter(
        rfm_scaled, x='frequency', y='monetary', color='rfm_segment',
        title='Customer Segmentation (Frequency vs. Monetary, colored by Segment)',
        hover_data=['recency', 'main_category']  # choose columns that exist
    )

    # Plot 2: Recency vs. Frequency
    fig2 = px.scatter(
        rfm_scaled, x='recency', y='frequency', color='rfm_segment',
        title='Customer Segmentation (Recency vs. Frequency, colored by Segment)',
        hover_data=['monetary', 'main_category']
    )

    # Plot 3: Segment Size Bar Plot
    seg_counts = rfm['rfm_segment'].value_counts().sort_index()
    fig3 = px.bar(
        x=seg_counts.index, y=seg_counts.values,
        labels={'x': 'Segment', 'y': 'Number of Users'},
        title='Customer Count by Segment'
    )

    return [fig1, fig2, fig3]

def run_eda_visualization(df: pd.DataFrame):
    """
    Runs EDA visualizations on the provided DataFrame.

    Args:
        df: DataFrame containing the data to visualize.

    Returns:
        List of Plotly figures for EDA visualizations.
    """
    logger.info("Running EDA visualizations...")

    eda_helper = EDAHelper()
    figures = [
        eda_helper.plot_event_type_distribution(df),
        eda_helper.plot_top_main_categories(df),
        eda_helper.plot_top_subcategories(df),
        eda_helper.plot_hourly_user_activity(df),
        eda_helper.plot_daily_unique_visitors(df),
        eda_helper.get_correlation_heatmap_fig(df)
    ]

    logger.info("EDA visualizations completed.")
    return figures

def evaluate_oof_predictions(
    trained_model,
    oof_predictions,
    y_train_oof,
    optimal_threshold=0.5,
    class_labels: List[str] = ["0", "1"]
):
    """
    Evaluates OOF predictions and generates evaluation plots.

    Args:
        trained_model: The trained model to evaluate.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Test feature matrix.
        y_test: Test target vector.
        oof_predictions: Out-of-Fold predictions from cross-validation.
        y_train_oof: True labels for OOF predictions.
        optimal_threshold: Threshold for classification.
        class_labels: List of class labels for confusion matrix and plots.

    Returns:
        None
    """
    logger.info("Evaluating OOF predictions...")

    # Create evaluator instance
    evaluator = ModelEvaluator(
        model=trained_model,
        model_name=trained_model.__class__.__name__
        )
    
    # Set OOF predictions and training labels
    evaluator.oof_predictions = oof_predictions
    evaluator.y_train_oof = y_train_oof

    # Evaluate OOF predictions
    fig1, fig2, fig3 = evaluator.evaluate_oof(threshold=optimal_threshold, class_labels=class_labels)
    logger.info("Confusion matrix, ROC curve, and PR curve plotted for OOF data.")
    return fig1, fig2, fig3

def evaluate_test_set_predictions(
    trained_model,
    y_test,
    y_proba,
    y_pred,
    class_labels: List[str] = ["0", "1"]
):
    """
    Evaluates test set predictions and generates evaluation plots.

    Args:
        trained_model: The trained model to evaluate.
        X_test: Test feature matrix.
        y_test: Test target vector.
        optimal_threshold: Threshold for classification.
        class_labels: List of class labels for confusion matrix and plots.

    Returns:
        None
    """
    logger.info("Evaluating test set predictions...")

    # Create evaluator instance
    evaluator = ModelEvaluator(
        model=clone(trained_model),
        model_name=trained_model.__class__.__name__
    )
    
    # Evaluate test set predictions
    fig1, fig2, fig3 = evaluator.evaluate_test_set(y_test, y_proba, y_pred, class_labels=class_labels)
    logger.info("Confusion matrix, ROC curve, and PR curve plotted for test data.")
    return fig1, fig2, fig3

def run_plot_threshold_metrics(
    trained_model,
    y_true,
    y_pred_proba,
    data_type="OOF Data",
    beta=1.0,
    optimal_metric='F1'
):
    """
    Plots threshold vs. various metrics (F1, Precision, Recall, J-statistic) using Plotly,
    and highlights the optimal threshold based on the selected metric.

    Args:
        trained_model: The trained model to evaluate.
        y_true: True labels for the data.
        y_pred_proba: Predicted probabilities from the model.
        data_type: Type of data (e.g., "OOF Data", "Test Data").
        beta: Beta value for F-beta score.
        optimal_metric: Metric to optimize for threshold selection.

    Returns:
        Plotly figure with threshold metrics.
    """
    evaluator = ModelEvaluator(
        model=trained_model,
        model_name=trained_model.__class__.__name__
    )
    
    fig = evaluator.plot_threshold_metrics(y_true, y_pred_proba, data_type=data_type, beta=beta, optimal_metric=optimal_metric)
    logger.info(f"Threshold metrics plotted for {data_type} data with optimal metric: {optimal_metric}.")
    return fig

def run_plot_calibration_curve(
    trained_model,
    X_test,
    y_true,
    test_proba,
    y_train_oof,
    oof_predictions,
    data_type=["OOF Data", "Test Data"]
):
    """
    Plots calibration curve for the model's predicted probabilities.

    Args:
        trained_model: The trained model to evaluate.
        y_true: True labels for the data.
        y_pred_proba: Predicted probabilities from the model.
        data_type: Type of data (e.g., "OOF Data", "Test Data").

    Returns:
        Plotly figure with calibration curve.
    """
    evaluator = ModelEvaluator(
        model=trained_model,
        model_name=trained_model.__class__.__name__
    )
    
    # test_proba = evaluator.model.predict_proba(X_test)[:, 1]  ## some issue here, need to check later
    fig1, fig2 = evaluator.plot_calibration_curve(y_train_oof, oof_predictions, y_true, test_proba, data_type=data_type)
    logger.info("Calibration curves plotted for OOF and test data.")
    return fig1, fig2

def run_plot_lift_gain_chart(
    trained_model,
    X_test,
    y_test,
    test_proba,
    y_train_oof,
    oof_predictions,
    data_type="OOF Data"
):
    """
    Plots Lift and Gain charts for the model's predicted probabilities.

    Args:
        trained_model: The trained model to evaluate.
        y_true: True labels for the data.
        y_pred_proba: Predicted probabilities from the model.
        data_type: Type of data (e.g., "OOF Data", "Test Data").

    Returns:
        None
    """
    evaluator = ModelEvaluator(
            model=trained_model,
            model_name=trained_model.__class__.__name__
    )
    
    # Plot lift and gain charts for OOF and test data
    gain_fig1, lift_fig1 = evaluator.plot_lift_gain_chart(
            y_train_oof, oof_predictions, data_type="OOF Data"
        )
    
    gain_fig2, lift_fig2 = evaluator.plot_lift_gain_chart(y_test, test_proba, data_type="Test Data")
    logger.info("Lift and Gain charts plotted for OOF and test data.")
    return gain_fig1, lift_fig1, gain_fig2, lift_fig2

