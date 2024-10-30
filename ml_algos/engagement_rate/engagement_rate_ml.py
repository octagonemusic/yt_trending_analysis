import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import shap
import logging
from datetime import datetime
import os
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

class EngagementRatePredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Change to script's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup logging in current directory
        log_filename = f'engagement_rate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    def prepare_features(self, df):
        features = pd.DataFrame()
        
        # Basic metrics (raw values)
        features['view_count'] = df['View Count']
        features['subscriber_count'] = df['Channel Subscriber Count']
        features['channel_video_count'] = df['Channel Video Count']
        features['channel_view_count'] = df['Channel View Count']
        features['duration'] = df['Duration (Seconds)']
        features['like_count'] = df['Like Count'].fillna(0)
        features['comment_count'] = df['Comment Count'].fillna(0)
        
        # Calculate video age
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        features['video_age'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
        
        # Engagement metrics
        features['likes_per_view'] = features['like_count'] / (features['view_count'] + 1)
        features['comments_per_view'] = features['comment_count'] / (features['view_count'] + 1)
        features['views_per_sub'] = features['view_count'] / (features['subscriber_count'] + 1)
        
        # Channel performance metrics
        features['channel_engagement'] = features['channel_view_count'] / (features['channel_video_count'] + 1)
        features['sub_to_view_ratio'] = features['subscriber_count'] / (features['channel_view_count'] + 1)
        
        # Time metrics
        features['hour'] = pd.to_datetime(df['Published At']).dt.hour
        features['day'] = pd.to_datetime(df['Published At']).dt.dayofweek
        features['month'] = pd.to_datetime(df['Published At']).dt.month
        
        # Binary features
        features['is_licensed'] = df['Licensed Content'].astype(int)
        features['has_caption'] = df['Caption'].astype(int)
        features['is_weekend'] = features['day'].isin([5,6]).astype(int)
        features['is_evening'] = features['hour'].between(17, 23).astype(int)
        
        # Category
        features['category_id'] = df['Category ID']
        
        # Handle missing values
        features = features.fillna(0)
        
        # Log transform numeric columns
        numeric_cols = ['view_count', 'subscriber_count', 'channel_video_count', 
                       'channel_view_count', 'duration', 'video_age']
        
        for col in numeric_cols:
            features[f'log_{col}'] = np.log1p(features[col])
        
        return features

    def prepare_target(self, df):
        """Calculate engagement rate focusing on like/comment relationship"""
        likes = df['Like Count'].fillna(0)
        comments = df['Comment Count'].fillna(0)
        views = df['View Count']
        
        # Calculate base metrics
        like_rate = likes / views
        comment_rate = comments / views
        
        # Calculate engagement score
        engagement_score = np.sqrt(like_rate * comment_rate)  # Geometric mean
        
        # Handle infinite/missing values
        engagement_score = engagement_score.replace([np.inf, -np.inf], np.nan)
        engagement_score = engagement_score.fillna(engagement_score.mean())
        
        # Log transform to handle skewness
        engagement_score = np.log1p(engagement_score)
        
        return engagement_score

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """Train model with different parameters"""
        self.model = XGBRegressor(
            n_estimators=2000,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=1,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            early_stopping_rounds=50
        )
        
        logging.info("Starting model training...")
        
        # Train model
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        logging.info(f"Model Performance Metrics:")
        logging.info(f"R2 Score: {r2:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
            
        return r2, rmse

    def plot_feature_importance(self):
        # SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
        plt.title('Feature Importance (SHAP values)')
        plt.tight_layout()
        plt.savefig('engagement_rate_importance.png')
        plt.close()
        
        # XGBoost feature importance
        plt.figure(figsize=(10, 8))
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig('engagement_rate_xgb_importance.png')
        plt.close()

    def explain_with_shap(self, sample_size=100):
        """Generate SHAP explanations"""
        logging.info("Generating SHAP explanations...")
        
        # Sample data for explanation
        if len(self.X_test) > sample_size:
            X_sample = self.X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.X_test
            
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Plot 1: Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('Feature Importance (SHAP values)')
        plt.tight_layout()
        plt.savefig('engagement_rate_shap_summary.png')
        plt.close()
        
        # Plot 2: Detailed SHAP values
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('Feature Impact on Engagement Rate')
        plt.tight_layout()
        plt.savefig('engagement_rate_shap_detailed.png')
        plt.close()
        
        # Plot 3: SHAP Dependence plots for top features
        feature_importance = np.abs(shap_values).mean(0)
        top_features = self.X_train.columns[np.argsort(feature_importance)[-3:]]
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature, 
                shap_values, 
                X_sample,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature}')
            plt.tight_layout()
            plt.savefig(f'engagement_rate_shap_dependence_{feature}.png')
            plt.close()
            
        logging.info("SHAP explanations generated and saved")
        return shap_values

    def explain_with_lime(self, num_features=10, num_samples=5):
        """Generate LIME explanations for random samples"""
        logging.info("Generating LIME explanations...")
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['Engagement Rate'],
            mode='regression'
        )
        
        # Generate explanations for random samples
        for i in range(num_samples):
            # Get random instance
            idx = np.random.randint(0, len(self.X_test))
            instance = self.X_test.iloc[idx]
            
            # Generate explanation
            exp = explainer.explain_instance(
                instance.values, 
                self.model.predict,
                num_features=num_features
            )
            
            # Plot explanation
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'LIME Explanation for Instance {i+1}')
            plt.tight_layout()
            plt.savefig(f'engagement_rate_lime_explanation_{i+1}.png')
            plt.close()
            
            # Log feature importance
            logging.info(f"\nLIME Explanation for Instance {i+1}:")
            for feature, importance in exp.as_list():
                logging.info(f"{feature}: {importance:.4f}")
                
        logging.info("LIME explanations generated and saved")

def main():
    # Read data
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    # Initialize predictor
    predictor = EngagementRatePredictor()
    
    # Prepare features and target
    X = predictor.prepare_features(df)
    y = predictor.prepare_target(df)
    
    # Split data
    predictor.split_data(X, y)
    
    # Train model
    predictor.train_model()
    
    # Evaluate model
    predictor.evaluate_model()
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Generate SHAP explanations
    shap_values = predictor.explain_with_shap()
    
    # Generate LIME explanations
    predictor.explain_with_lime()

if __name__ == "__main__":
    main()
