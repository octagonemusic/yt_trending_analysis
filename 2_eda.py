import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import logging
from pathlib import Path
from io import StringIO

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()
sns.set_palette("husl")

# Read the data
df = pd.read_excel('youtube_trending_data_20241030_090258.xlsx')  # Replace with your file name

def clean_data(df):
    """Clean and prepare data for analysis"""
    # Convert columns to appropriate types
    numeric_cols = ['View Count', 'Like Count', 'Comment Count', 'Duration (Seconds)',
                   'Channel Subscriber Count', 'Channel Video Count', 'Channel View Count']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamps and make them timezone-naive
    df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
    df['Collection Date'] = pd.to_datetime(df['Collection Date'])
    
    # Calculate video age at collection time
    df['Video Age (Days)'] = (df['Collection Date'] - df['Published At']).dt.total_seconds() / (24*60*60)
    
    return df

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    log_filename = f'logs/eda_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will print to console as well
        ]
    )
    return log_filename

def basic_stats(df):
    """Print basic statistics about the dataset"""
    logging.info("\n=== Basic Statistics ===")
    logging.info(f"Total number of videos: {len(df)}")
    
    logging.info("\nNumerical columns summary:")
    logging.info("\n" + df.describe().to_string())
    
    logging.info("\nMissing values:")
    logging.info("\n" + df.isnull().sum().to_string())

def create_visualizations(df):
    """Create various visualizations for the dataset"""
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    logging.info("\n=== Creating Visualizations ===")
    
    # 1. View Count Distribution
    logging.info("Creating View Count Distribution plot...")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='View Count', bins=50)
    plt.title('Distribution of View Counts')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('plots/views_distribution.png')
    plt.close()

    # 2. Correlation Matrix
    numeric_cols = ['View Count', 'Like Count', 'Comment Count', 'Duration (Seconds)',
                   'Channel Subscriber Count', 'Video Age (Days)']
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

    # 3. Engagement Ratio (Likes/Views)
    df['Engagement Ratio'] = df['Like Count'] / df['View Count']
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='View Count', y='Engagement Ratio')
    plt.title('Engagement Ratio vs View Count')
    plt.xscale('log')
    plt.savefig('plots/engagement_ratio.png')
    plt.close()

    # 4. Video Duration Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Duration (Seconds)', bins=50)
    plt.title('Distribution of Video Durations')
    plt.savefig('plots/duration_distribution.png')
    plt.close()

    # 5. Top Channels
    top_channels = df['Channel Title'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    top_channels.plot(kind='bar')
    plt.title('Top 10 Channels in Trending Videos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top_channels.png')
    plt.close()

    # 6. Publishing Time Analysis
    df['Hour Published'] = df['Published At'].dt.hour
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Hour Published', bins=24)
    plt.title('Distribution of Publishing Hours')
    plt.savefig('plots/publishing_hours.png')
    plt.close()

def generate_insights(df):
    """Generate key insights from the data"""
    logging.info("\n=== Generating Insights ===")
    insights = []
    
    # View count insights
    avg_views = df['View Count'].mean()
    median_views = df['View Count'].median()
    insights.append(f"Average views: {avg_views:,.0f}")
    insights.append(f"Median views: {median_views:,.0f}")
    
    # Most popular channels
    top_channel = df['Channel Title'].mode().iloc[0]
    insights.append(f"Most frequent channel in trending: {top_channel}")
    
    # Video duration insights
    avg_duration = df['Duration (Seconds)'].mean() / 60
    insights.append(f"Average video duration: {avg_duration:.2f} minutes")
    
    # Engagement insights
    avg_engagement = (df['Like Count'] / df['View Count']).mean() * 100
    insights.append(f"Average engagement rate: {avg_engagement:.2f}%")
    
    # Log additional statistical information
    logging.info("\nDetailed Statistics:")
    logging.info(f"View Count Stats:\n{df['View Count'].describe().to_string()}")
    logging.info(f"\nLike Count Stats:\n{df['Like Count'].describe().to_string()}")
    logging.info(f"\nComment Count Stats:\n{df['Comment Count'].describe().to_string()}")
    
    # Log top 10 channels
    logging.info("\nTop 10 Channels:")
    logging.info("\n" + df['Channel Title'].value_counts().head(10).to_string())
    
    return insights

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting YouTube Trending Videos Analysis")
    
    # Read and clean data
    logging.info("Reading and cleaning data...")
    df = pd.read_excel('youtube_trending_data_20241030_090258.xlsx')
    df = clean_data(df)
    
    # Log initial data info
    logging.info("\nDataset Info:")
    buffer = StringIO()
    df.info(buf=buffer)
    logging.info(buffer.getvalue())
    
    # Perform analysis
    basic_stats(df)
    create_visualizations(df)
    
    # Generate insights
    insights = generate_insights(df)
    
    # Log insights
    logging.info("\n=== Key Insights ===")
    for insight in insights:
        logging.info(f"- {insight}")
    
    logging.info(f"\nAnalysis complete. Log file saved to: {log_file}")
    logging.info("Visualizations saved in 'plots' directory")

if __name__ == "__main__":
    main()
