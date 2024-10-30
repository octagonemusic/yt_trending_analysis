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

class ChannelGrowthPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Setup directory and logging
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        log_filename = f'channel_growth_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        features['subscriber_count'] = df['Channel Subscriber Count']
        features['channel_video_count'] = df['Channel Video Count']
        features['channel_view_count'] = df['Channel View Count']
        features['view_count'] = df['View Count']
        features['like_count'] = df['Like Count'].fillna(0)
        features['comment_count'] = df['Comment Count'].fillna(0)
        
        # Channel performance metrics
        features['avg_views_per_video'] = features['channel_view_count'] / (features['channel_video_count'] + 1)
        features['subs_per_video'] = features['subscriber_count'] / (features['channel_video_count'] + 1)
        features['views_per_sub'] = features['channel_view_count'] / (features['subscriber_count'] + 1)
        
        # Video performance metrics
        features['likes_per_view'] = features['like_count'] / (features['view_count'] + 1)
        features['comments_per_view'] = features['comment_count'] / (features['view_count'] + 1)
        features['view_to_sub_ratio'] = features['view_count'] / (features['subscriber_count'] + 1)
        
        # Video age
        df['Collection Date'] = pd.to_datetime(df['Collection Date'])
        df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
        features['video_age'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
        
        # Time features
        hour = pd.to_datetime(df['Published At']).dt.hour
        features['is_peak_hours'] = hour.between(17, 23).astype(int)
        features['is_morning'] = hour.between(6, 11).astype(int)
        features['is_afternoon'] = hour.between(12, 16).astype(int)
        
        day = pd.to_datetime(df['Published At']).dt.dayofweek
        features['is_weekend'] = day.isin([5,6]).astype(int)
        
        # Content features
        features['duration'] = df['Duration (Seconds)']
        features['has_caption'] = df['Caption'].astype(int)
        features['category_id'] = df['Category ID']
        
        # Channel size categories
        features['is_small_channel'] = (features['subscriber_count'] < 100000).astype(int)
        features['is_medium_channel'] = ((features['subscriber_count'] >= 100000) & 
                                       (features['subscriber_count'] < 1000000)).astype(int)
        features['is_large_channel'] = (features['subscriber_count'] >= 1000000).astype(int)
        
        # Log transforms
        for col in ['subscriber_count', 'channel_video_count', 'channel_view_count', 
                   'view_count', 'like_count', 'comment_count', 'duration']:
            features[f'log_{col}'] = np.log1p(features[col])
        
        return features

    def prepare_target(self, df):
        """Calculate channel growth rate"""
        # Core metrics
        views = df['View Count']
        subscribers = df['Channel Subscriber Count']
        video_age = (pd.to_datetime(df['Collection Date']) - 
                    pd.to_datetime(df['Published At']).dt.tz_localize(None)).dt.total_seconds() / (24*60*60)
        
        # Calculate engagement factors
        likes = df['Like Count'].fillna(0)
        comments = df['Comment Count'].fillna(0)
        engagement_score = np.sqrt((likes + comments) / (views + 1))
        
        # Calculate growth rate (weighted by engagement)
        growth_rate = (subscribers * engagement_score) / (video_age + 1)
        
        # Handle outliers
        q1 = growth_rate.quantile(0.25)
        q3 = growth_rate.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # Cap extreme values
        growth_rate = np.minimum(growth_rate, upper_bound)
        
        # Log transform
        log_growth_rate = np.log1p(growth_rate)
        
        return log_growth_rate

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
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

    def analyze_with_shap(self, max_display=20):
        logging.info("Generating SHAP analysis...")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
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
        plt.savefig('channel_growth_shap_importance.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            self.X_test,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Impact')
        plt.tight_layout()
        plt.savefig('channel_growth_shap_impact.png')
        plt.close()
        
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_dict = dict(zip(self.X_test.columns, feature_importance))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logging.info("\nTop features by SHAP importance:")
        for feature, importance in sorted_features[:max_display]:
            logging.info(f"{feature}: {importance:.4f}")

    def explain_with_lime(self, num_features=10, num_samples=5):
        logging.info("Generating LIME explanations...")
        
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['Channel Growth Rate'],
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
            plt.savefig(f'channel_growth_lime_explanation_{i+1}.png')
            plt.close()
            
            logging.info(f"\nLIME Explanation for Instance {i+1}:")
            for feature, importance in exp.as_list():
                logging.info(f"{feature}: {importance:.4f}")

def main():
    df = pd.read_excel('/home/octagone/Documents/Coding Projects/yt_trending_analysis/youtube_trending_data_20241030_090258.xlsx')
    
    predictor = ChannelGrowthPredictor()
    
    X = predictor.prepare_features(df)
    y = predictor.prepare_target(df)
    
    predictor.split_data(X, y)
    predictor.train_model()
    predictor.evaluate_model()
    predictor.analyze_with_shap()
    predictor.explain_with_lime()

if __name__ == "__main__":
    main()
