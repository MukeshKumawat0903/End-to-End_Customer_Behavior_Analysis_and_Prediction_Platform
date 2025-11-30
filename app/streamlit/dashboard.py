# app/streamlit/dashboard.py
import os
import sys
import pickle
import tempfile

import shap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Add project root to sys.path for imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../')
    )
)

# Project-specific imports
from src.data.load import load_ecommerce_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.models.evaluate import run_full_evaluation
from src.schemas import PredictionRequest
from src.visualization.visualize import (
    evaluate_oof_predictions,
    evaluate_test_set_predictions,
    run_plot_threshold_metrics,
    run_plot_calibration_curve,
    run_plot_lift_gain_chart,
    plot_rfm_segments
)
from src.models.interpret.shap_analysis import calculate_shap_values
from src.visualization.explain_plots import plot_shap_global

# Streamlit and matplotlib configuration
st.set_page_config(page_title="Customer Conversion Analytics", layout="wide")
plt.style.use('ggplot')

# Feature columns
numerical_features = [
    'price', 'hour', 'is_weekend', 'total_clicks',
    'avg_session_time', 'purchase_freq', 'products_viewed',
    'price_range', 'session_duration', 'recency', 'frequency', 'monetary'
]
categorical_features = ['main_category', 'rfm_segment']
features = numerical_features + categorical_features

# Set joblib temp folder
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()

def main():
    # st.title("🛍 Customer Conversion Prediction Dashboard")
    st.markdown("## 🛍Customer Conversion Prediction Dashboard")

    # --- Centralize features in session state
    if "features" not in st.session_state:
        st.session_state["features"] = features
    if "numerical_features" not in st.session_state:
        st.session_state["numerical_features"] = numerical_features
    if "categorical_features" not in st.session_state:
        st.session_state["categorical_features"] = categorical_features

    # --- Sidebar for Configuration only
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        with st.expander("🔧 Model Parameters", expanded=False):
            st.markdown("**Used in:** 🧹 Data Loading")
            sample_size_option = st.selectbox(
                "📊 Sample Size Option",
                options=["Custom", "None"],
                index=0
            )
            if sample_size_option == "Custom":
                sample_size = st.number_input(
                    "Enter custom sample size",
                    min_value=50000,
                    max_value=10000000,
                    value=100000,
                    step=50000
                )
            else:
                sample_size = None

            st.markdown("### 📁 Data Split")
            test_size = st.slider(
                "Test Size (%)", min_value=10, max_value=40, value=20, step=1
            )
            val_size = st.slider(
                "Validation Size (%)", min_value=10, max_value=30, value=25, step=1
            )

            st.markdown("### 🎯 Threshold")
            use_optimal_thr = st.checkbox("Use optimal threshold from cross-validation?", value=True)
            threshold = st.slider(
                "Classification Threshold", min_value=0.3, max_value=0.7, value=0.5, step=0.01,
                disabled=use_optimal_thr
            )

    # --- Tab Structure ---
    tab_titles = ["EDA", "Feature Engineering", "Training", "Evaluation", "SHAP", "Single Prediction", "Download"]
    tabs = st.tabs(tab_titles)

    # --- EDA Tab ---
    with tabs[0]:
        st.markdown("### 🔍 Exploratory Data Analysis")
        load_data_btn = st.button("📥 Load Data", key="eda_load")
        if load_data_btn:
            with st.spinner("Loading and preprocessing data..."):
                try:
                    st.session_state["raw_df"] = load_data(sample_size)
                    st.success("✅ Data loaded and preprocessed.")
                except Exception as e:
                    st.error(f"❌ Failed to load data: {e}")

        # Only show EDA if data is loaded
        if "raw_df" in st.session_state:
            eda_btn = st.button("📂 Run EDA", key="eda_run")
            if eda_btn:
                with st.spinner("Preparing data for EDA..."):
                    try:
                        processed_df = preprocess_and_display(st.session_state["raw_df"])
                        st.session_state["processed_df"] = processed_df
                        st.success("✅ Data preprocessed for EDA.")
                    except Exception as e:
                        st.error(f"❌ EDA preparation failed: {e}")
            if "processed_df" in st.session_state:
                if st.session_state["processed_df"].empty:
                    st.warning("📌 No data available for EDA. Please load and preprocess the data first.")
        else:
            st.info("Please load the data to begin EDA.")

    # --- Feature Engineering Tab ---
    with tabs[1]:
        st.header("🔧 Feature Engineering")
        feature_engg_btn = st.button(" 🔧 Run Feature Engineering", key="feature_engg_btn")
        if feature_engg_btn:
            if "processed_df" in st.session_state and st.session_state["processed_df"] is not None:
                with st.spinner("Running feature engineering and RFM segmentation..."):
                    try:
                        feature_engineering_and_display()
                        st.success("✅ Feature engineering and RFM segmentation completed!")
                    except Exception as e:
                        st.error(f"❌ Feature engineering failed: {e}")
            else:
                st.warning("⚠️ Please run EDA and preprocessing before feature engineering.")


    # --- Training Tab ---
    with tabs[2]:
        st.markdown("### 🔧 Model Training")
        train_model_btn = st.button("🔧 Train Model", key="train_btn")
        def check_processed_data():
            if "processed_df" not in st.session_state or st.session_state["processed_df"] is None:
                st.error("❌ Please run EDA or preprocessing before training/evaluating the model.")
                return False
            return True

        if train_model_btn:
            if not check_processed_data():
                st.stop()
            final_threshold = st.session_state.optimal_threshold if use_optimal_thr and st.session_state.get("optimal_threshold") else threshold
            st.info(f"Using {'optimal' if use_optimal_thr else 'manual'} threshold: {final_threshold}")
            with st.spinner("Training model... please wait..."):
                try:
                    train_and_display_model(
                        st.session_state["processed_df"],
                        test_size,
                        val_size,
                        final_threshold
                    )
                    st.success("✅ Model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")

    # --- Evaluation Tab ---
    with tabs[3]:
        st.markdown("### 📈 Model Evaluation")
        evaluation_btn = st.button("📈 Evaluate Model", key="eval_btn")
        if evaluation_btn:
            if st.session_state.get("model", None) is not None:
                for key, val in {
                    "show_oof": False,
                    "show_test": False,
                    "show_threshold_matrix": False,
                    "calibration_curve": False,
                    "lift_gain_charts": False,
                    "evaluation_started": True
                }.items():
                    st.session_state.setdefault(key, val)
            else:
                st.warning("⚠️ Please train the model first.")

        if st.session_state.get("evaluation_started", False):
            display_evaluation_plots(threshold)

    # --- SHAP Tab ---
    with tabs[4]:
        st.markdown("### 🧠 Generate SHAP Explanations")
        generate_shap_btn = st.button("🧠 Generate SHAP", key="shap_btn")
        if generate_shap_btn:
            if st.session_state.get("model", None) is not None:
                selected_sample = st.selectbox("Select Sample for SHAP", options=range(0, 50, 5), index=2)
                st.session_state.selected_sample = selected_sample
                with st.spinner("Generating SHAP values..."):
                    try:
                        generate_and_display_shap(st.session_state["processed_df"])
                        st.success("✅ SHAP explanation generated!")
                    except Exception as e:
                        st.error(f"❌ SHAP generation failed: {e}")
            else:
                st.warning("⚠️ Please train the model first to use SHAP.")

    # --- Single Prediction Tab ---
    with tabs[5]:
        st.markdown("### 🎯 Single Customer Prediction")
        single_prediction_ui()

    # --- Download Tab ---
    with tabs[6]:
        st.markdown("### ⬇️ Download Model and Artifacts")
        download_btn = st.button("⬇️ Download Model", key="download_btn")
        if download_btn:
            if st.session_state.get("model", None) is not None:
                display_download_options()
            else:
                st.warning("⚠️ Please train the model before downloading.")

def load_data(sample_size):
    """Load and display raw data"""
    with st.spinner("Loading data..."):
        file_path = r"D:\Learnings\My_Projects\Datasets\eCommerce behavior data 2019-Nov\eCommerce behavior data 2019-Nov.parquet"
        raw_df = load_ecommerce_data(file_path, sample_size)

        st.write(f"Raw Data Shape: {raw_df.shape}")
        
        st.subheader("Raw Data Sample")
        st.dataframe(raw_df.head(), use_container_width=True)
            
        st.divider()
        st.subheader("Data Summary")
        st.write(f"Total Rows: {raw_df.shape[0]:,}")
        st.write(f"Columns: {raw_df.shape[1]}")
        st.write("Event Type Distribution:")
        st.bar_chart(raw_df['event_type'].value_counts())
            
        return raw_df

def preprocess_and_display(raw_df):
    """Preprocess data and display results with enhanced interaction"""
    with st.spinner("🔄 Preprocessing data..."):
        processed_df, results, rmf_results = preprocess_data(raw_df)
        st.session_state.rmf_results = rmf_results

    # Basic stats section
    st.markdown("### 📊 Data Overview")
    st.success("✅ Data Preprocessed Successfully")
    st.write(f"**Total Rows:** {processed_df.shape[0]:,}")
    st.write(f"**Total Columns:** {processed_df.shape[1]}")
    st.divider()

    # Expandable section for data inspection
    with st.expander("🔍 **Explore Processed Data**", expanded=False):
        st.subheader("📈 Summary Statistics")
        st.dataframe(processed_df.describe(), use_container_width=True)

        st.divider()
        st.subheader("⏱️ Session Duration Distribution")
        fig, ax = plt.subplots()
        processed_df['session_duration'].hist(ax=ax, bins=30, color='skyblue', edgecolor='black')
        ax.set_title("Session Duration Histogram")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.divider()

    # EDA Visualizations
    st.subheader("📉 Exploratory Data Analysis (EDA)")
    render_evaluation_section(
        section_title="📊 EDA Visualizations",
        results=results,
        plot_titles=[
            "Event Type Distribution",
            "Top 10 Product Categories",
            "Top 10 Subcategories",
            "Hourly User Activity",
            "Daily Unique Visitors",
            "Feature Correlation Heatmap"

        ],
        error_message="⚠️ Failed to generate EDA visualizations"
    )
    # selected_eda_cols = ['event_type', 'category_code', 'event_time', 'user_id']
    # processed_df.drop(columns=selected_eda_cols, inplace=True)

    return processed_df

def feature_engineering_and_display():
    """Feature engineering and display results"""
    rmf_results = st.session_state.get("rmf_results")
    if rmf_results is not None:
        plot_titles = [
            "Frequency vs. Monetary by Segment",
            "Recency vs. Frequency by Segment",
            "Customer Count by Segment"
        ]
        render_evaluation_section(
            section_title="RFM Segmentation Visualizations",
            results=rmf_results,
            plot_titles=plot_titles,
            error_message="Error generating RFM segmentation plots"
        )
    else:
        st.info("No RFM data available. Please ensure feature engineering is complete.")
 
def train_and_display_model(model_df, test_size, val_size, threshold):
    # """Handle model training flow""")
    features = st.session_state["features"]
    numerical_features = st.session_state["numerical_features"]
    categorical_features = st.session_state["categorical_features"]
    
    X = model_df[features]
    y = model_df['target']
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size/100, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/100, stratify=y_train_val, random_state=42
    )
    
    # Map original DF to NumPy for all datasets
    # X_train = X_train[features].to_numpy()
    # X_val = X_val[features].to_numpy()
    # y_train = y_train.to_numpy()
    # y_val = y_val.to_numpy()
    # X_test = X_test[features].to_numpy()
    # y_test = y_test.to_numpy()

    X_train = X_train[features]
    X_val = X_val[features]
    X_test = X_test[features]


    # Train model
    model, y_proba, y_pred = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        threshold = threshold,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    
    # Display model info
    st.subheader("🎯 Model Architecture")

    class_names = ['Non-Target Customer', 'Target Customer']

    oof_predictions, y_train_oof, optimal_threshold, cv_results = run_full_evaluation(
        trained_model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_labels=class_names,
        n_splits=5,
        early_stopping_rounds=20
    )

    st.write("### Model Evaluation Results")

    st.subheader("📈 Cross-Validation Summary")
    st.write(f"**Mean ROC AUC:** {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    st.write(f"**Mean PR AUC:** {cv_results['mean_pr_auc']:.4f} ± {cv_results['std_pr_auc']:.4f}")
    st.write(f"**Mean F1 (0.5 thr):** {cv_results['mean_f1_score_thresh05']:.4f} ± {cv_results['std_f1_score_thresh05']:.4f}")

    st.write(f"**Optimal Threshold:** {optimal_threshold:.4f}")
    st.session_state.oof_predictions = oof_predictions
    st.session_state.y_train_oof = y_train_oof
    st.session_state.optimal_threshold = optimal_threshold

    # Store in session state
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_proba = y_proba
    st.session_state.y_pred = y_pred

def render_evaluation_section(
    section_title: str,
    results,
    plot_titles: list[str],
    error_message: str
):
    """Render a section for evaluation plots"""
    st.divider()
    st.subheader(section_title)
    try:
        figs = list(results) if isinstance(results, (tuple, list)) else [results]

        if len(figs) != len(plot_titles):
            st.warning(f"⚠️ Number of plots ({len(figs)}) and titles ({len(plot_titles)}) differ. Titles may not match figures.")

        for i, fig in enumerate(figs):
            st.subheader(plot_titles[i] if i < len(plot_titles) else f"Plot {i+1}")
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=True)
            elif isinstance(fig, plt.Figure):
                st.pyplot(fig)
            else:
                st.warning(f"Figure {i+1} is not a valid Plotly or Matplotlib figure. Skipping.")
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")

def display_evaluation_plots(threshold):
    """Display evaluation plots based on session state"""
    st.markdown("### 🔍 Model Evaluation")

    required_keys = ["model", "oof_predictions", "y_train_oof", "X_test", "y_test"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning(f"Session state missing required key: {key}. Please train and evaluate the model first.")
            return

    # UI Toggles
    st.session_state.show_oof = st.checkbox("Show OOF Evaluation Plots", value=st.session_state.get("show_oof", False))
    st.session_state.show_test = st.checkbox("Show Test Evaluation Plots", value=st.session_state.get("show_test", False))
    st.session_state.show_threshold_matrix = st.checkbox("Show Threshold Confusion Matrix Plots ", value=st.session_state.get("show_threshold_matrix", False))
    st.session_state.calibration_curve = st.checkbox("Show Calibration Curve for Test Data", value=st.session_state.get("calibration_curve", False))
    st.session_state.lift_gain_charts = st.checkbox("Show Lift and Gain Charts", value=st.session_state.get("lift_gain_charts", False))

    if st.session_state.show_oof:
        try:
            results = evaluate_oof_predictions(
                trained_model=st.session_state.model,
                oof_predictions=st.session_state.oof_predictions,
                y_train_oof=st.session_state.y_train_oof,
                class_labels=['Non-Target Customer', 'Target Customer'],
                optimal_threshold=threshold
            )
            render_evaluation_section(
                section_title="OOF Evaluation Plots",
                results=results,
                plot_titles=["OOF Confusion Matrix", "OOF ROC Curve", "OOF Precision-Recall (PR) Curve"],
                error_message="Failed to generate OOF evaluation plots"
            )
        except Exception as e:
            st.error(f"Failed to generate OOF evaluation plots: {str(e)}")

    if st.session_state.show_test:
        try:
            results = evaluate_test_set_predictions(
                trained_model=st.session_state.model,
                y_test=st.session_state.y_test,
                y_proba=st.session_state.y_proba,
                y_pred=st.session_state.y_pred,
                class_labels=['Non-Target Customer', 'Target Customer'],
            )
            render_evaluation_section(
                section_title="Test Evaluation Plots",
                results=results,
                plot_titles=["Test Confusion Matrix", "Test ROC Curve", "Test Precision-Recall (PR) Curve"],
                error_message="Failed to generate test evaluation plots"
            )
        except Exception as e:
            st.error(f"Failed to generate test evaluation plots: {str(e)}")

    if st.session_state.show_threshold_matrix:
        try:
            results = run_plot_threshold_metrics(
                trained_model=st.session_state.model,
                y_true=st.session_state.y_train_oof,
                y_pred_proba=st.session_state.oof_predictions,
                data_type="OOF Data",
                beta=1.0,
                optimal_metric='F1'
            )
            render_evaluation_section(
                section_title="Threshold Confusion Matrix Plots",
                results=results,
                plot_titles=["OOF Threshold Confusion Matrix"],
                error_message="Failed to generate threshold confusion matrix plots"
            )
        except Exception as e:
            st.error(f"Failed to generate threshold confusion matrix plots: {str(e)}")

    if st.session_state.calibration_curve:
        try:
            results = run_plot_calibration_curve(
                trained_model=st.session_state.model,
                X_test=st.session_state.X_test,
                y_true=st.session_state.y_test,
                test_proba=st.session_state.y_proba,
                y_train_oof=st.session_state.y_train_oof,
                oof_predictions=st.session_state.oof_predictions,
                data_type=["OOF Data", "Test Data"]
            )
            render_evaluation_section(
                section_title="Calibration Curve for OOF and Test Data",
                results=results,
                plot_titles=["OOF Calibration Curve", "Test Calibration Curve"],
                error_message="Failed to generate calibration curve"
            )
        except Exception as e:
            st.error(f"Failed to generate calibration curve: {str(e)}")

    if st.session_state.lift_gain_charts:
        try:
            results = run_plot_lift_gain_chart(
                trained_model=st.session_state.model,
                X_test=st.session_state.X_test,
                y_test=st.session_state.y_test,
                test_proba=st.session_state.y_proba,
                y_train_oof=st.session_state.y_train_oof,
                oof_predictions=st.session_state.oof_predictions,
                data_type="Test Data"
            )
            render_evaluation_section(
                section_title="Lift and Gain Charts",
                results=results,
                plot_titles=["OOF Lift Chart", "OOF Gain Chart", "Test Data Lift Chart", "Test Data Gain Chart"],
                error_message="Failed to generate lift and gain charts"
            )
        except Exception as e:
            st.error(f"Failed to generate lift and gain charts: {str(e)}")

def generate_and_display_shap(df):
    """Handle SHAP analysis"""
    model = st.session_state.model
    preprocessor = model.named_steps['preprocessor']
    xgb_model = model.named_steps['xgb']
    sample_idx = st.session_state.selected_sample
    
    # Calculate SHAP values
    shap_values, X_val_processed, feature_names = calculate_shap_values(
        model=xgb_model,
        preprocessor=preprocessor,
        data=df,
        model_type='xgb'
    )
    
    # Store in session state
    st.session_state.shap_values = shap_values
    st.session_state.feature_names = feature_names
    
    # Display plots
    st.markdown("### 📊 SHAP Analysis")
    
    st.subheader("Global Feature Importance")
    fig = plot_shap_global(shap_values, feature_names, plot_type='bar')
    st.pyplot(fig)
        
    st.subheader("Feature Impact Distribution")
    fig = plot_shap_global(shap_values, feature_names, plot_type='dot')
    st.pyplot(fig)
        
    # Individual sample explanation
    st.subheader("🧠 Individual Prediction Explanation")
    explain_sample(sample_idx, shap_values, X_val_processed, feature_names)

def explain_sample(sample_idx, shap_values, X_val, feature_names):
    """Display force plot for individual sample"""    
    # Generate force plot as HTML (JS-based)
    force_plot = shap.force_plot(
        base_value=shap_values.base_values[sample_idx],
        shap_values=shap_values.values[sample_idx],
        features=X_val[sample_idx],
        feature_names=feature_names,
        matplotlib=False  # <- Use JS-based for compatibility
    )

    # Render force plot in Streamlit using HTML
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=500)

    html_data = shap_force_plot_to_html(force_plot)
    st.download_button(
        label="Download Explanation (HTML)",
        data=html_data,
        file_name=f"shap_explanation_{sample_idx}.html",
        mime="text/html"
    )

def shap_force_plot_to_html(force_plot_obj):
    """
    Convert a SHAP force_plot object to HTML string using its `html()` method.
    """
    try:
        return force_plot_obj.html()
    except AttributeError:
        raise RuntimeError("Invalid SHAP force plot object. Ensure you are using shap.force_plot().")

def display_download_options():
    st.markdown("### 📤 Download Assets")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Model",
            data=pickle.dumps(st.session_state.model),
            file_name="conversion_model.pkl",
            mime="application/octet-stream"
        )
    with col2:
        if st.session_state.get("shap_values", None) is not None:
            csv = save_top_features_to_csv()
            st.download_button(
                label="Download Top Features",
                data=csv,
                file_name="top_features.csv",
                mime="text/csv"
            )
        else:
            st.info("Run SHAP analysis first to download top features.")

def save_top_features_to_csv():
    """Convert top features to CSV"""
    shap_df = pd.DataFrame(st.session_state.shap_values.values, 
                          columns=st.session_state.feature_names)
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(10).index
    return shap_df[top_features].to_csv(index=False).encode('utf-8')


def single_prediction_ui():
    """UI for single customer prediction with schema validation"""
    if st.session_state.get("model", None) is None:
        st.warning("⚠️ Please train the model first to make predictions.")
        return
    
    st.markdown("### 📝 Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price = st.number_input("💰 Price", min_value=0.1, max_value=100000.0, value=100.0, step=10.0)
        hour = st.slider("⏰ Hour of Day", min_value=0, max_value=23, value=14)
        is_weekend = st.checkbox("📅 Is Weekend")
        total_clicks = st.number_input("🖱️ Total Clicks", min_value=0, max_value=10000, value=10, step=1)
        avg_session_time = st.number_input("⏱️ Avg Session Time (seconds)", min_value=0.0, max_value=10000.0, value=120.5, step=10.0)
    
    with col2:
        purchase_freq = st.slider("🛒 Purchase Frequency", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        products_viewed = st.number_input("👁️ Products Viewed", min_value=0, max_value=1000, value=5, step=1)
        price_range = st.number_input("💵 Price Range", min_value=0.0, max_value=100000.0, value=50.0, step=10.0)
        session_duration = st.number_input("⏳ Session Duration (seconds)", min_value=0.0, max_value=10000.0, value=300.0, step=10.0)
    
    with col3:
        recency = st.number_input("📆 Recency (days)", min_value=0, max_value=1000, value=10, step=1)
        frequency = st.number_input("🔄 Frequency", min_value=0, max_value=1000, value=3, step=1)
        monetary = st.number_input("💸 Monetary Value", min_value=0.0, max_value=1000000.0, value=500.0, step=50.0)
        main_category = st.text_input("📦 Main Category", value="electronics")
        rfm_segment = st.slider("🎯 RFM Segment", min_value=0, max_value=3, value=2)
    
    if st.button("🔮 Predict", key="predict_single"):
        try:
            # Validate input using PredictionRequest schema
            from pydantic import ValidationError
            request_data = PredictionRequest(
                price=price,
                hour=hour,
                is_weekend=is_weekend,
                total_clicks=total_clicks,
                avg_session_time=avg_session_time,
                purchase_freq=purchase_freq,
                products_viewed=products_viewed,
                price_range=price_range,
                session_duration=session_duration,
                recency=recency,
                frequency=frequency,
                monetary=monetary,
                main_category=main_category,
                rfm_segment=rfm_segment
            )
            
            # Convert to DataFrame for prediction
            input_data = pd.DataFrame([request_data.model_dump()])
            
            # Get features from session state
            features = st.session_state["features"]
            input_data = input_data[features]
            
            # Make prediction
            model = st.session_state["model"]
            probability = float(model.predict_proba(input_data)[0, 1])  # Convert to Python float
            threshold = st.session_state.get("optimal_threshold", 0.5)
            prediction = int(probability >= threshold)
            
            # Calculate confidence
            if probability < 0.3 or probability > 0.7:
                confidence = "High"
                confidence_color = "green"
            elif probability < 0.4 or probability > 0.6:
                confidence = "Medium"
                confidence_color = "orange"
            else:
                confidence = "Low"
                confidence_color = "red"
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Prediction", "Target Customer" if prediction == 1 else "Non-Target")
            with col2:
                st.metric("📈 Probability", f"{probability:.2%}")
            with col3:
                st.metric("🎚️ Threshold", f"{threshold:.3f}")
            with col4:
                st.markdown(f"**Confidence:** ::{confidence_color}[{confidence}]")
            
            # Progress bar for probability
            st.markdown("#### Conversion Probability")
            st.progress(probability)
            
            if prediction == 1:
                st.success("✅ This customer is likely to convert! Consider targeted marketing.")
            else:
                st.info("ℹ️ This customer is less likely to convert. May need engagement strategies.")
                
        except ValidationError as ve:
            st.error("❌ Input validation failed:")
            for error in ve.errors():
                st.error(f"  • {error['loc'][0]}: {error['msg']}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()