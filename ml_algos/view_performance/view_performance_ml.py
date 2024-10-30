import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import lime
import lime.lime_tabular
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os

def setup_logging():
    """Setup logging configuration"""
    log_filename = f'view_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def prepare_features(df):
    """Prepare features for modeling"""
    features = pd.DataFrame()
    
    # Calculate video age
    df['Collection Date'] = pd.to_datetime(df['Collection Date'])
    df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
    df['Video Age (Days)'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
    
    # Numeric features
    features['duration'] = df['Duration (Seconds)']
    features['subscriber_count'] = df['Channel Subscriber Count']
    features['channel_video_count'] = df['Channel Video Count']
    features['channel_view_count'] = df['Channel View Count']
    features['video_age'] = df['Video Age (Days)']
    
    # Categorical features
    features['category_id'] = df['Category ID']
    features['is_licensed'] = df['Licensed Content'].astype(int)
    features['has_caption'] = df['Caption'].astype(int)
    
    # Time-based features
    features['hour_published'] = pd.to_datetime(df['Published At']).dt.hour
    features['day_of_week'] = pd.to_datetime(df['Published At']).dt.dayofweek
    
    return features

def explain_with_shap(model, X_test):
    """Generate SHAP explanations"""
    logging.info("\nGenerating SHAP explanations...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For classification, use values for class 1 (above median)
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]  # Use positive class
        mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    else:
        # Reshape if needed
        n_features = X_test.shape[1]
        shap_values_plot = shap_values.reshape(-1, n_features)
        mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Ensure arrays are 1D
    feature_names_array = np.array(X_test.columns)
    mean_abs_shap_array = np.array(mean_abs_shap).flatten()
    
    # Create DataFrame with verified arrays
    feature_importance = pd.DataFrame({
        'feature': feature_names_array,
        'importance': mean_abs_shap_array
    }).sort_values('importance', ascending=False)
    
    logging.info("\nSHAP Feature Importance:")
    logging.info("\n" + feature_importance.to_string())
    
    return shap_values, explainer

def explain_with_lime(model, X_train, X_test, feature_names):
    """Generate LIME explanations"""
    logging.info("\nGenerating LIME explanations...")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['below_median', 'above_median'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        X_test.iloc[0].values, 
        model.predict_proba
    )
    
    plt.figure()
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()
    
    return exp

def main():
    # Change working directory to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting View Performance Classification Analysis")
    
    # Read data
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    # Prepare features and target
    X = prepare_features(df)
    y = (df['View Count'] > df['View Count'].median()).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate and log model performance metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info("\nModel Performance Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))
    
    # Generate explanations
    shap_values, shap_explainer = explain_with_shap(model, X_test)
    lime_exp = explain_with_lime(model, X_train, X_test, X.columns)
    
    logging.info(f"\nAnalysis complete. Log file saved to: {log_file}")
    logging.info("Visualizations saved in current directory")

if __name__ == "__main__":
    main()
