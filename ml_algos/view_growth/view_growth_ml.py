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

class ViewGrowthPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Change to script's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup logging
        log_filename = f'view_growth_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        
        # Core metrics
        features['view_count'] = df['View Count']
        features['like_count'] = df['Like Count'].fillna(0)
        features['comment_count'] = df['Comment Count'].fillna(0)
        features['subscriber_count'] = df['Channel Subscriber Count']
        
        # Calculate video age and velocity metrics
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        features['video_age'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
        
        # Engagement metrics
        features['likes_per_view'] = features['like_count'] / (features['view_count'] + 1)
        features['comments_per_view'] = features['comment_count'] / (features['view_count'] + 1)
        features['views_per_sub'] = features['view_count'] / (features['subscriber_count'] + 1)
        
        # Channel performance
        features['channel_video_count'] = df['Channel Video Count']
        features['channel_view_count'] = df['Channel View Count']
        features['avg_channel_views'] = features['channel_view_count'] / (features['channel_video_count'] + 1)
        
        # Time features
        hour = pd.to_datetime(df['Published At']).dt.hour
        features['is_morning'] = hour.between(6, 11).astype(int)
        features['is_afternoon'] = hour.between(12, 17).astype(int)
        features['is_evening'] = hour.between(18, 23).astype(int)
        features['is_night'] = (~(features['is_morning'] | features['is_afternoon'] | features['is_evening'])).astype(int)
        
        day = pd.to_datetime(df['Published At']).dt.dayofweek
        features['is_weekend'] = day.isin([5,6]).astype(int)
        features['is_monday'] = (day == 0).astype(int)
        features['is_friday'] = (day == 4).astype(int)
        
        # Content features
        features['duration'] = df['Duration (Seconds)']
        features['has_caption'] = df['Caption'].astype(int)
        features['category_id'] = df['Category ID']
        
        # Log transform core metrics
        for col in ['view_count', 'like_count', 'comment_count', 'subscriber_count', 
                   'channel_video_count', 'channel_view_count', 'duration']:
            features[f'log_{col}'] = np.log1p(features[col])
        
        # Handle missing values
        features = features.fillna(0)
        
        return features

    def prepare_target(self, df):
        """Calculate view growth rate"""
        # Calculate views per hour
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        hours_since_published = (df['Collection Date'] - df['Published At']).dt.total_seconds() / 3600
        
        views_per_hour = df['View Count'] / (hours_since_published + 1)
        
        # Log transform to handle skewness
        log_growth_rate = np.log1p(views_per_hour)
        
        return log_growth_rate

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
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
        # Get feature importance from model
        importance = self.model.feature_importances_
        features = self.X_train.columns
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20))
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('view_growth_importance.png')
        plt.close()

    def explain_with_lime(self, num_features=10, num_samples=5):
        logging.info("Generating LIME explanations...")
        
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['View Growth Rate'],
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
            plt.savefig(f'view_growth_lime_explanation_{i+1}.png')
            plt.close()
            
            logging.info(f"\nLIME Explanation for Instance {i+1}:")
            for feature, importance in exp.as_list():
                logging.info(f"{feature}: {importance:.4f}")

def main():
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    predictor = ViewGrowthPredictor()
    
    X = predictor.prepare_features(df)
    y = predictor.prepare_target(df)
    
    predictor.split_data(X, y)
    predictor.train_model()
    predictor.evaluate_model()
    predictor.plot_feature_importance()
    predictor.explain_with_lime()

if __name__ == "__main__":
    main()