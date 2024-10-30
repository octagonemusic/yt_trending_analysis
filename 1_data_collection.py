from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd
import time
import os
from dotenv import load_dotenv
from isodate import parse_duration

# Load environment variables
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_details(channel_id):
    """Get detailed channel statistics"""
    try:
        channel_response = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        ).execute()
        
        if channel_response['items']:
            return channel_response['items'][0]
        return None
    except Exception as e:
        print(f"Error getting channel details for {channel_id}: {str(e)}")
        return None

def convert_duration_to_seconds(duration_str):
    """Convert ISO 8601 duration to seconds"""
    try:
        duration = parse_duration(duration_str)
        return int(duration.total_seconds())
    except:
        return None

def get_video_stats(video_id):
    """Get detailed statistics for a specific video"""
    try:
        stats = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_id
        ).execute()
        
        if not stats['items']:
            return None
        
        video_info = stats['items'][0]
        snippet = video_info['snippet']
        statistics = video_info['statistics']
        content_details = video_info['contentDetails']
        
        # Get channel details
        channel_id = snippet['channelId']
        channel_details = get_channel_details(channel_id)
        channel_country = channel_details['snippet'].get('country') if channel_details else None
        
        # Convert duration to seconds
        duration = convert_duration_to_seconds(content_details['duration'])
        
        return {
            "Video ID": video_id,
            "Title": snippet['title'],
            "Published At": snippet['publishedAt'],
            "Channel ID": channel_id,
            "Channel Title": snippet['channelTitle'],
            "Category ID": snippet.get('categoryId'),
            "Tags": snippet.get('tags'),
            "View Count": statistics.get('viewCount'),
            "Like Count": statistics.get('likeCount'),
            "Favorite Count": statistics.get('favoriteCount'),
            "Comment Count": statistics.get('commentCount'),
            "Duration (Seconds)": duration,
            "Definition": content_details.get('definition'),
            "Caption": content_details.get('caption'),
            "Licensed Content": content_details.get('licensedContent'),
            "Region Restriction": content_details.get('regionRestriction'),
            "Channel Country": channel_country,
            "Channel Subscriber Count": channel_details['statistics'].get('subscriberCount') if channel_details else None,
            "Channel Video Count": channel_details['statistics'].get('videoCount') if channel_details else None,
            "Channel View Count": channel_details['statistics'].get('viewCount') if channel_details else None,
            "Collection Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error getting stats for video {video_id}: {str(e)}")
        return None

def get_trending_videos():
    """Get top 100 trending videos"""
    videos_data = []
    
    try:
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            chart='mostPopular',
            regionCode='US',
            maxResults=100
        )
        response = request.execute()
        
        for video in response['items']:
            video_id = video['id']
            video_stats = get_video_stats(video_id)
            
            if video_stats:
                videos_data.append(video_stats)
                print(f"Collected data for video: {video_stats['Title']}")
            
            # Respect API quota limits
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error fetching trending videos: {str(e)}")
    
    return videos_data

def main():
    # Collect trending videos data
    print("Starting data collection...")
    trending_videos = get_trending_videos()
    
    # Convert to DataFrame
    df = pd.DataFrame(trending_videos)
    
    # Save to Excel
    output_file = f'youtube_trending_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
