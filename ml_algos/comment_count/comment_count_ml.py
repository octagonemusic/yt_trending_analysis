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

class CommentCountPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Change to script's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup logging
        log_filename = f'comment_count_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        
        # Core metrics that directly influence comments
        features['view_count'] = df['View Count']
        features['like_count'] = df['Like Count'].fillna(0)
        features['subscriber_count'] = df['Channel Subscriber Count']
        
        # Calculate video age and velocity metrics
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        features['video_age'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
        
        # Velocity and acceleration metrics
        features['views_per_hour'] = features['view_count'] / (features['video_age'] * 24 + 1)
        features['likes_per_hour'] = features['like_count'] / (features['video_age'] * 24 + 1)
        
        # Engagement intensity
        features['like_view_ratio'] = features['like_count'] / (features['view_count'] + 1)
        features['view_sub_ratio'] = features['view_count'] / (features['subscriber_count'] + 1)
        
        # Time features (focusing on when people are most likely to comment)
        hour = pd.to_datetime(df['Published At']).dt.hour
        features['is_morning'] = hour.between(6, 11).astype(int)
        features['is_afternoon'] = hour.between(12, 17).astype(int)
        features['is_evening'] = hour.between(18, 23).astype(int)
        features['is_night'] = (~(features['is_morning'] | features['is_afternoon'] | features['is_evening'])).astype(int)
        
        # Day type features
        day = pd.to_datetime(df['Published At']).dt.dayofweek
        features['is_weekend'] = day.isin([5,6]).astype(int)
        features['is_monday'] = (day == 0).astype(int)
        features['is_friday'] = (day == 4).astype(int)
        
        # Content features
        features['has_caption'] = df['Caption'].astype(int)
        features['category_id'] = df['Category ID']
        features['duration'] = df['Duration (Seconds)']
        
        # Log transform core metrics
        for col in ['view_count', 'like_count', 'subscriber_count', 'duration']:
            features[f'log_{col}'] = np.log1p(features[col])
        
        # Handle any remaining missing values
        features = features.fillna(0)
        
        return features

    def prepare_target(self, df):
        """Prepare comment count target with better normalization"""
        comments = df['Comment Count'].fillna(0)
        
        # Calculate comments per view ratio first
        comments_per_view = comments / (df['View Count'] + 1)
        
        # Then log transform both metrics
        log_comments = np.log1p(comments)
        log_comments_per_view = np.log1p(comments_per_view)
        
        # Combine both metrics with weights
        target = 0.7 * log_comments + 0.3 * log_comments_per_view
        
        return target

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """Train model with focused parameters"""
        self.model = XGBRegressor(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.008,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_alpha=0.1,
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
        plt.savefig('comment_count_importance.png')
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
        plt.savefig('comment_count_xgb_importance.png')
        plt.close()

    def explain_with_lime(self, num_features=10, num_samples=5):
        """Generate LIME explanations for random samples"""
        logging.info("Generating LIME explanations...")
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['Comment Count'],
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
            plt.savefig(f'comment_count_lime_explanation_{i+1}.png')
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
    predictor = CommentCountPredictor()
    
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
    
    # Generate LIME
    predictor.explain_with_lime()

if __name__ == "__main__":
    main()
