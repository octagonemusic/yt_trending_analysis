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

class SubscriberGrowthPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Change to script's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup logging
        log_filename = f'subscriber_growth_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        
        # Basic metrics (most directly related to subscriber growth)
        features['view_count'] = df['View Count']
        features['like_count'] = df['Like Count'].fillna(0)
        features['comment_count'] = df['Comment Count'].fillna(0)
        features['subscriber_count'] = df['Channel Subscriber Count']
        
        # Video age
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        features['video_age'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
        
        # Simple engagement metrics
        features['likes_per_view'] = features['like_count'] / (features['view_count'] + 1)
        features['comments_per_view'] = features['comment_count'] / (features['view_count'] + 1)
        
        # Basic channel metrics
        features['channel_video_count'] = df['Channel Video Count']
        features['channel_view_count'] = df['Channel View Count']
        
        # Simple time features
        features['hour'] = pd.to_datetime(df['Published At']).dt.hour
        features['day'] = pd.to_datetime(df['Published At']).dt.dayofweek
        features['is_weekend'] = features['day'].isin([5,6]).astype(int)
        
        # Content features
        features['duration'] = df['Duration (Seconds)']
        features['has_caption'] = df['Caption'].astype(int)
        features['category_id'] = df['Category ID']
        
        # Log transform core metrics
        numeric_cols = ['view_count', 'like_count', 'comment_count', 'subscriber_count', 
                       'channel_video_count', 'channel_view_count', 'duration']
        
        for col in numeric_cols:
            features[f'log_{col}'] = np.log1p(features[col])
        
        # Handle missing values
        features = features.fillna(0)
        
        return features

    def prepare_target(self, df):
        """Simplified subscriber growth rate calculation"""
        # Basic subscriber growth rate
        views = df['View Count']
        subscribers = df['Channel Subscriber Count']
        video_age = (pd.to_datetime(df['Collection Date']) - 
                    pd.to_datetime(df['Published At']).dt.tz_localize(None)).dt.total_seconds() / 3600
        
        # Simple growth rate (subscribers per view)
        sub_growth_rate = subscribers / (views + 1)
        
        # Log transform
        log_growth_rate = np.log1p(sub_growth_rate)
        
        return log_growth_rate

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """Train model with simpler parameters"""
        self.model = XGBRegressor(
            n_estimators=1000,
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

    def explain_with_lime(self, num_features=10, num_samples=5):
        logging.info("Generating LIME explanations...")
        
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['Subscriber Growth Rate'],
            mode='regression'
        )
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.X_test))
            instance = self.X_test.iloc[idx]
            
            exp = explainer.explain_instance(
                instance.values, 
                self.model.predict,
                num_features=num_features
            )
            
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'LIME Explanation for Instance {i+1}')
            plt.tight_layout()
            plt.savefig(f'subscriber_growth_lime_explanation_{i+1}.png')
            plt.close()
            
            logging.info(f"\nLIME Explanation for Instance {i+1}:")
            for feature, importance in exp.as_list():
                logging.info(f"{feature}: {importance:.4f}")

    def analyze_with_shap(self, max_display=20):
        """Generate SHAP analysis and plots"""
        logging.info("Generating SHAP analysis...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for test set
        shap_values = explainer.shap_values(self.X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            self.X_test,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('subscriber_growth_shap_importance.png')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            self.X_test,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Impact')
        plt.tight_layout()
        plt.savefig('subscriber_growth_shap_impact.png')
        plt.close()
        
        # Log top features and their importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_dict = dict(zip(self.X_test.columns, feature_importance))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logging.info("\nTop features by SHAP importance:")
        for feature, importance in sorted_features[:max_display]:
            logging.info(f"{feature}: {importance:.4f}")

def main():
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    predictor = SubscriberGrowthPredictor()
    
    X = predictor.prepare_features(df)
    y = predictor.prepare_target(df)
    
    predictor.split_data(X, y)
    predictor.train_model()
    predictor.evaluate_model()
    predictor.analyze_with_shap()
    predictor.explain_with_lime()

if __name__ == "__main__":
    main()
    