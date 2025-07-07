"""
Chart generation functions for Pepino Analytics.

Provides functions to create various charts and visualizations
that can be embedded in Discord messages or CLI output.
"""

import logging
import base64
import io
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import matplotlib and related libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # Define Figure as None when matplotlib is not available
    logger.warning("Matplotlib not available - chart generation disabled")

# Try to import seaborn for better styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available - using basic matplotlib styling")


def _setup_matplotlib_style():
    """Setup matplotlib with a clean, modern style."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Set style
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    else:
        plt.style.use('default')
    
    # Configure for better rendering
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def _figure_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    if not MATPLOTLIB_AVAILABLE or fig is None:
        return ""
    
    try:
        # Create a canvas and render the figure
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Get the PNG data
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Clean up
        buf.close()
        plt.close(fig)
        
        return img_base64
    except Exception as e:
        logger.error(f"Error converting figure to base64: {e}")
        return ""


def create_activity_graph(dates: List[str], counts: List[int], 
                         title: str = "Daily Message Activity",
                         channel_name: Optional[str] = None) -> str:
    """
    Create a line chart showing activity over time.
    
    Args:
        dates: List of date strings (YYYY-MM-DD format)
        counts: List of message counts corresponding to dates
        title: Chart title
        channel_name: Optional channel name to include in title
        
    Returns:
        Base64-encoded PNG string for Discord embedding
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    try:
        _setup_matplotlib_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert dates to datetime objects
        date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        
        # Create the line plot
        ax.plot(date_objects, counts, marker='o', linewidth=2, markersize=4)
        
        # Customize the plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Message Count', fontsize=11)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//7)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add some padding
        plt.tight_layout()
        
        return _figure_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Error creating activity graph: {e}")
        return ""


def create_channel_activity_pie(channel_names: List[str], message_counts: List[int],
                               title: str = "Channel Activity Distribution") -> str:
    """
    Create a pie chart showing message distribution across channels.
    
    Args:
        channel_names: List of channel names
        message_counts: List of message counts for each channel
        title: Chart title
        
    Returns:
        Base64-encoded PNG string for Discord embedding
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    try:
        _setup_matplotlib_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(message_counts, labels=channel_names, 
                                         autopct='%1.1f%%', startangle=90)
        
        # Customize
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        return _figure_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Error creating channel activity pie chart: {e}")
        return ""


def create_user_activity_bar(user_names: List[str], message_counts: List[int],
                            title: str = "Top User Activity",
                            limit: int = 10) -> str:
    """
    Create a bar chart showing top users by message count.
    
    Args:
        user_names: List of user names
        message_counts: List of message counts for each user
        title: Chart title
        limit: Maximum number of users to show
        
    Returns:
        Base64-encoded PNG string for Discord embedding
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    try:
        _setup_matplotlib_style()
        
        # Limit the number of users
        if len(user_names) > limit:
            user_names = user_names[:limit]
            message_counts = message_counts[:limit]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(user_names))
        bars = ax.barh(y_pos, message_counts)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(user_names)
        ax.set_xlabel('Message Count', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{int(width):,}', ha='left', va='center', fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return _figure_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Error creating user activity bar chart: {e}")
        return ""


def create_word_cloud(text_data: Union[str, List[str]], 
                     title: str = "Word Cloud",
                     max_words: int = 100) -> str:
    """
    Create a word cloud from text data.
    
    Args:
        text_data: Text string or list of text strings
        title: Chart title
        max_words: Maximum number of words to include
        
    Returns:
        Base64-encoded PNG string for Discord embedding
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    try:
        # Try to import wordcloud
        try:
            from wordcloud import WordCloud
            WORDCLOUD_AVAILABLE = True
        except ImportError:
            logger.warning("WordCloud not available - skipping word cloud generation")
            return ""
        
        if not WORDCLOUD_AVAILABLE:
            return ""
        
        _setup_matplotlib_style()
        
        # Prepare text data
        if isinstance(text_data, list):
            text = ' '.join(text_data)
        else:
            text = text_data
        
        if not text.strip():
            return ""
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return _figure_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}")
        return ""


# Convenience function for the legacy activity_trends template
def create_daily_activity_chart(temporal_data: List[Dict[str, Any]], 
                               title: str = "Daily Message Activity") -> str:
    """
    Create a daily activity chart from temporal data.
    
    Args:
        temporal_data: List of dicts with 'period' and 'message_count' keys
        title: Chart title
        
    Returns:
        Base64-encoded PNG string for Discord embedding
    """
    if not temporal_data:
        return ""
    
    # Extract dates and counts
    dates = []
    counts = []
    
    for item in temporal_data:
        period = item.get('period', '')
        count = item.get('message_count', 0)
        
        # Handle different period formats
        if ' ' in period:  # Hourly format: "2025-06-13 14:00"
            date_str = period.split(' ')[0]
        elif period.startswith('2025-W'):  # Weekly format: "2025-W23"
            # Convert week to approximate date (simplified)
            year, week = period.split('-W')
            # This is a simplified conversion - in production you'd want proper week handling
            date_str = f"{year}-01-01"  # Placeholder
        else:  # Daily format: "2025-06-13"
            date_str = period
        
        dates.append(date_str)
        counts.append(count)
    
    return create_activity_graph(dates, counts, title) 