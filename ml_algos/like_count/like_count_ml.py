import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import lime
import lime.lime_tabular
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def setup_logging():
    """Setup logging configuration"""
    log_filename = 'like_count_{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
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
    
    # Basic features
    features['duration'] = df['Duration (Seconds)']
    features['subscriber_count'] = df['Channel Subscriber Count']
    features['channel_video_count'] = df['Channel Video Count']
    features['channel_view_count'] = df['Channel View Count']
    features['video_age'] = df['Video Age (Days)']
    features['view_count'] = df['View Count']  # Adding view count as a feature
    
    # Categorical features
    features['category_id'] = df['Category ID']
    features['is_licensed'] = df['Licensed Content'].astype(int)
    features['has_caption'] = df['Caption'].astype(int)
    
    # Time-based features
    features['hour_published'] = pd.to_datetime(df['Published At']).dt.hour
    features['day_of_week'] = pd.to_datetime(df['Published At']).dt.dayofweek
    
    # Add new features
    
    # View-based features
    features['views_per_sub'] = df['View Count'] / df['Channel Subscriber Count']
    features['views_per_day'] = df['View Count'] / features['video_age']
    
    # Channel engagement features
    features['channel_avg_views'] = df['Channel View Count'] / df['Channel Video Count']
    features['sub_to_view_ratio'] = df['Channel Subscriber Count'] / df['Channel View Count']
    
    # Time-based features
    features['is_weekend'] = features['day_of_week'].isin([5,6]).astype(int)
    features['is_peak_hours'] = features['hour_published'].between(17, 23).astype(int)
    
    # Title features (if available)
    features['title_length'] = df['Title'].str.len()
    features['title_word_count'] = df['Title'].str.split().str.len()
    
    # Interaction features
    features['view_sub_interaction'] = features['views_per_sub'] * features['sub_to_view_ratio']
    features['time_view_interaction'] = features['views_per_day'] * features['video_age']
    
    return features

def explain_with_shap(model, X_test):
    """Generate SHAP explanations"""
    logging.info("\nGenerating SHAP explanations...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Calculate feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_abs_shap
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
        class_names=['likes'],
        mode='regression'
    )
    
    exp = explainer.explain_instance(
        X_test.iloc[0].values, 
        model.predict
    )
    
    plt.figure()
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()
    
    return exp

def create_preprocessing_pipeline():
    # Define complete lists of numeric and categorical features
    numeric_features = [
        'duration', 'subscriber_count', 'channel_video_count', 'channel_view_count',
        'video_age', 'view_count', 'views_per_sub', 'views_per_day', 
        'channel_avg_views', 'sub_to_view_ratio', 'title_length', 
        'title_word_count', 'view_sub_interaction', 'time_view_interaction'
    ]
    
    categorical_features = [
        'category_id', 'is_licensed', 'has_caption', 'is_weekend', 
        'is_peak_hours', 'hour_published', 'day_of_week'
    ]
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            # Note: We're not applying any transformation to categorical features
            # but including them in the output
            ('cat', 'passthrough', categorical_features)
        ]
    )
    
    return preprocessor

def train_model(X_train, y_train):
    # Create pipeline
    preprocessor = create_preprocessing_pipeline()
    model = XGBRegressor(random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Parameter grid for XGBoost
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 4, 5],
        'regressor__learning_rate': [0.01, 0.1, 0.3],
        'regressor__subsample': [0.8, 1.0],        # XGBoost specific parameter
        'regressor__colsample_bytree': [0.8, 1.0]  # XGBoost specific parameter
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

def analyze_errors(y_true, y_pred, X_test):
    errors = pd.DataFrame({
        'true': y_true,
        'predicted': y_pred,
        'error': np.abs(y_true - y_pred)
    })
    
    # Join with features
    errors = pd.concat([errors, X_test], axis=1)
    
    # Analyze where model performs worst
    worst_predictions = errors.nlargest(5, 'error')
    logging.info("\nWorst Predictions Analysis:")
    logging.info(worst_predictions)
    
    return errors

def main():
    # Change working directory to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Like Count Prediction Analysis")
    
    # Read data
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    # Log initial data stats
    logging.info(f"\nInitial data shape: {df.shape}")
    logging.info(f"Missing Like Count values: {df['Like Count'].isna().sum()}")
    
    # Remove rows with NaN in Like Count
    df = df.dropna(subset=['Like Count'])
    logging.info(f"Shape after removing NaN: {df.shape}")
    
    # Prepare features and target
    X = prepare_features(df)
    y = np.log1p(df['Like Count'])
    
    # Log feature stats
    logging.info("\nFeature info:")
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Any NaN in features: {X.isnull().any().any()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    grid_search = train_model(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_.named_steps['regressor']
    
    # Generate and log model performance metrics
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)   
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logging.info("\nModel Performance Metrics:")
    logging.info(f"R2 Score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    
    # Generate explanations using the best model
    shap_values, shap_explainer = explain_with_shap(best_model, X_test)
    lime_exp = explain_with_lime(best_model, X_train, X_test, X.columns)
    
    # Analyze errors
    errors = analyze_errors(y_test, y_pred, X_test)
    
    logging.info(f"\nAnalysis complete. Log file saved to: {log_file}")
    logging.info("Visualizations saved in current directory")

if __name__ == "__main__":
    main()
