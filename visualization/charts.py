"""
Chart generation functions for Discord message analysis visualization
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (headless)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import re
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from wordcloud import WordCloud
from utils.helpers import get_temp_file_path, sanitize_filename, ensure_temp_dir_exists

# Configure matplotlib for headless operation
plt.ioff()  # Turn off interactive mode
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable warnings


def cleanup_matplotlib():
    """Force cleanup of matplotlib resources"""
    try:
        plt.close('all')  # Close all figures
        plt.clf()         # Clear current figure
        plt.cla()         # Clear current axes
        gc.collect()      # Force garbage collection
    except:
        pass


def create_activity_graph(data: pd.DataFrame) -> str:
    """Create activity over time graph"""
    plt.figure(figsize=(12, 6))
    
    # Plot message count
    plt.plot(data['date'], data['message_count'], label='Messages', linewidth=2)
    
    # Plot active users
    plt.plot(data['date'], data['active_users'], label='Active Users', linewidth=2)
    
    plt.title('Message Activity Over Time', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the graph
    graph_path = get_temp_file_path('activity_graph.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_channel_activity_pie(data: pd.DataFrame) -> str:
    """Create channel activity pie chart"""
    plt.figure(figsize=(10, 10))
    
    # Create pie chart
    plt.pie(data['message_count'], 
            labels=data['channel_name'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True)
    
    plt.title('Channel Activity Distribution', fontsize=14, pad=20)
    
    # Save the graph
    graph_path = get_temp_file_path('channel_activity.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_user_activity_bar(data: pd.DataFrame) -> str:
    """Create user activity bar chart"""
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(data['username'], data['message_count'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.title('Top Users by Message Count', fontsize=14, pad=20)
    plt.xlabel('User', fontsize=12)
    plt.ylabel('Message Count', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the graph
    graph_path = get_temp_file_path('user_activity.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_word_cloud(word_freq: Dict[str, int]) -> str:
    """Create word cloud from word frequencies"""
    plt.figure(figsize=(12, 8))
    
    # Generate word cloud
    wordcloud = WordCloud(width=1200, height=800,
                        background_color='white',
                        max_words=100,
                        contour_width=3,
                        contour_color='steelblue')
    
    wordcloud.generate_from_frequencies(word_freq)
    
    # Display the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words', fontsize=14, pad=20)
    
    # Save the graph
    graph_path = get_temp_file_path('word_cloud.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_user_activity_chart(daily_activity: List[Tuple], user_name: str) -> str:
    """Generate user activity chart for past 30 days"""
    chart_path = None
    
    if daily_activity and len(daily_activity) > 1:
        # Prepare data for plotting
        dates = []
        message_counts = []
        
        for date_str, count in daily_activity:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_obj)
                message_counts.append(count)
            except:
                continue
        
        if dates and message_counts:
            # Create the plot with explicit figure management
            fig, ax = plt.subplots(figsize=(12, 6))
            
            try:
                ax.bar(dates, message_counts, color='#5865F2', alpha=0.7, edgecolor='#4752C4', linewidth=1)
                
                # Clean user name for chart title (remove emojis and special chars)
                clean_user_name = re.sub(r'[^\w\s-]', '', user_name)
                
                # Formatting
                ax.set_title(f'Daily Message Activity - {clean_user_name}', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Number of Messages', fontsize=12)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                plt.setp(ax.get_xticklabels(), rotation=45)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add some statistics to the plot
                avg_messages = sum(message_counts) / len(message_counts)
                max_messages = max(message_counts)
                
                ax.axhline(y=avg_messages, color='red', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_messages:.1f} msg/day')
                
                ax.legend()
                plt.tight_layout()
                
                # Save the chart with sanitized filename
                safe_user_name = sanitize_filename(user_name)
                if not safe_user_name:
                    safe_user_name = "unknown_user"
                
                chart_filename = f"user_activity_{safe_user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                chart_path = os.path.join('temp', chart_filename)
                
                # Ensure temp directory exists
                ensure_temp_dir_exists()
                
                plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
                
            finally:
                # Always close the figure to free memory
                plt.close(fig)
                # Force aggressive cleanup
                cleanup_matplotlib()
    
    return chart_path


def create_channel_activity_chart(daily_activity: List[Tuple], channel_name: str) -> str:
    """Generate channel activity chart for past 30 days"""
    chart_path = None
    
    if daily_activity and len(daily_activity) > 1:
        # Prepare data for plotting
        dates = []
        message_counts = []
        
        for date_str, count in daily_activity:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_obj)
                message_counts.append(count)
            except:
                continue
        
        if dates and message_counts:
            # Create the plot with explicit figure management
            fig, ax = plt.subplots(figsize=(12, 6))
            
            try:
                ax.bar(dates, message_counts, color='#5865F2', alpha=0.7, edgecolor='#4752C4', linewidth=1)
                
                # Clean channel name for chart title (remove emojis)
                clean_channel_name = re.sub(r'[^\w\s-]', '', channel_name)
                
                # Formatting
                ax.set_title(f'Daily Message Activity - {clean_channel_name}', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Number of Messages', fontsize=12)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                plt.setp(ax.get_xticklabels(), rotation=45)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add some statistics to the plot
                avg_messages = sum(message_counts) / len(message_counts)
                max_messages = max(message_counts)
                
                ax.axhline(y=avg_messages, color='red', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_messages:.1f} msg/day')
                
                ax.legend()
                plt.tight_layout()
                
                # Save the chart with sanitized filename
                safe_channel_name = sanitize_filename(channel_name)
                if not safe_channel_name:
                    safe_channel_name = "unknown_channel"
                
                chart_filename = f"channel_activity_{safe_channel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                chart_path = os.path.join('temp', chart_filename)
                
                # Ensure temp directory exists
                ensure_temp_dir_exists()
                
                plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
                
            finally:
                # Always close the figure to free memory
                plt.close(fig)
                # Force aggressive cleanup
                cleanup_matplotlib()
    
    return chart_path
