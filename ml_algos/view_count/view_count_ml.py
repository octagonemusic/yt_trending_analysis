import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import lime
import lime.lime_tabular
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os

def setup_logging():
    """Setup logging configuration"""
    # Save log in view_count directory
    log_filename = 'view_count_{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # Saves in view_count directory 
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
    
    # Save plot in view_count directory
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')  # Saves in view_count directory
    plt.close()
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(shap_values).mean(0)
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
        class_names=['views'],
        mode='regression'
    )
    
    exp = explainer.explain_instance(
        X_test.iloc[0].values, 
        model.predict
    )
    
    # Save plot in view_count directory
    plt.figure()
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')  # Saves in view_count directory
    plt.close()
    
    return exp

def main():
    # Change working directory to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting View Count Prediction Analysis")
    
    # Read data using absolute path
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    # Prepare features and target
    X = prepare_features(df)
    y = np.log1p(df['View Count'])  # Log transform for better regression
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate and log model performance metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logging.info("\nModel Performance Metrics:")
    logging.info(f"R2 Score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    
    # Generate explanations
    shap_values, shap_explainer = explain_with_shap(model, X_test)
    lime_exp = explain_with_lime(model, X_train, X_test, X.columns)
    
    logging.info(f"\nAnalysis complete. Log file saved to: {log_file}")
    logging.info("Visualizations saved in current directory")

if __name__ == "__main__":
    main()
