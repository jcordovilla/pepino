"""
Server growth analysis script.

This script analyzes server growth from day 1 by:
1. Finding the first activity date for each user
2. Calculating cumulative number of users each day
3. Calculating daily human messages
4. Calculating weekly average of human messages
5. Plotting all three variables together

Usage:
    poetry run python scripts/server_growth_analysis.py --output charts/server_growth.png
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import click
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from pepino.config import Settings
from pepino.analysis.data_facade import get_analysis_data_facade

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_first_activity_dates(facade) -> Dict[str, str]:
    """Get first activity date for each user (human users only)."""
    logger.info("Fetching first activity dates for all users...")
    
    query = f"""
    SELECT 
        author_name,
        MIN(DATE(timestamp)) as first_activity_date
    FROM messages
    WHERE {facade.message_repository.base_filter}
    AND (author_is_bot = 0 OR author_is_bot IS NULL)
    AND timestamp IS NOT NULL
    GROUP BY author_name
    ORDER BY first_activity_date ASC
    """
    
    results = facade.message_repository.db_manager.execute_query(query)
    
    first_activity = {}
    for row in results:
        first_activity[row['author_name']] = row['first_activity_date']
    
    logger.info(f"Found first activity dates for {len(first_activity)} users")
    return first_activity


def get_daily_human_messages(facade) -> Dict[str, int]:
    """Get daily count of human messages from day 1."""
    logger.info("Fetching daily human message counts...")
    
    query = f"""
    SELECT 
        DATE(timestamp) as date,
        COUNT(*) as message_count
    FROM messages
    WHERE {facade.message_repository.base_filter}
    AND (author_is_bot = 0 OR author_is_bot IS NULL)
    AND timestamp IS NOT NULL
    GROUP BY DATE(timestamp)
    ORDER BY date ASC
    """
    
    results = facade.message_repository.db_manager.execute_query(query)
    
    daily_messages = {}
    for row in results:
        daily_messages[row['date']] = row['message_count']
    
    logger.info(f"Found message data for {len(daily_messages)} days")
    return daily_messages


def calculate_cumulative_users(first_activity: Dict[str, str], 
                               start_date: datetime, 
                               end_date: datetime) -> Dict[str, int]:
    """Calculate cumulative number of users for each day since server start."""
    logger.info("Calculating cumulative user counts...")
    
    # Convert first activity dates to datetime objects
    user_first_dates = []
    for username, first_date_str in first_activity.items():
        try:
            first_date = datetime.strptime(first_date_str, '%Y-%m-%d')
            user_first_dates.append(first_date)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid date format for user {username}: {first_date_str}, error: {e}")
            continue
    
    # Sort first dates
    user_first_dates.sort()
    
    # Build cumulative count for all dates in range
    cumulative_users = {}
    current_date = start_date
    
    while current_date <= end_date:
        date_key = current_date.strftime('%Y-%m-%d')
        # Count how many users had their first activity on or before this date
        count = sum(1 for first_date in user_first_dates if first_date <= current_date)
        cumulative_users[date_key] = count
        current_date += timedelta(days=1)
    
    logger.info(f"Calculated cumulative users for {len(cumulative_users)} days")
    return cumulative_users


def calculate_weekly_average(daily_messages: Dict[str, int], 
                            dates: List[datetime]) -> List[float]:
    """Calculate 7-day moving average of daily messages."""
    logger.info("Calculating weekly moving averages...")
    
    weekly_avg = []
    
    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        message_count = daily_messages.get(date_str, 0)
        
        if i < 6:
            # For first 6 days, use available data
            window_dates = dates[:i+1]
            window_counts = [daily_messages.get(d.strftime('%Y-%m-%d'), 0) for d in window_dates]
            avg = np.mean(window_counts) if window_counts else 0.0
        else:
            # Use 7-day window
            window_dates = dates[i-6:i+1]
            window_counts = [daily_messages.get(d.strftime('%Y-%m-%d'), 0) for d in window_dates]
            avg = np.mean(window_counts) if window_counts else 0.0
        
        weekly_avg.append(avg)
    
    return weekly_avg


def create_growth_chart(cumulative_users: Dict[str, int],
                       daily_messages: Dict[str, int],
                       weekly_avg: List[float],
                       dates: List[datetime],
                       output_path: str):
    """Create a chart showing cumulative users, daily messages, and weekly average."""
    logger.info("Generating growth chart...")
    
    # Prepare data
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    cumulative_user_counts = [cumulative_users.get(ds, 0) for ds in date_strings]
    daily_message_counts = [daily_messages.get(ds, 0) for ds in date_strings]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Left y-axis: Cumulative users
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Users', fontsize=12, color='#5865F2')
    line1 = ax1.plot(dates, cumulative_user_counts, color='#5865F2', linewidth=2, 
                     label='Cumulative Users', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='#5865F2')
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Daily messages and weekly average
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Human Messages', fontsize=12, color='#ED4245')
    
    # Bar chart for daily messages - light grey
    bars = ax2.bar(dates, daily_message_counts, alpha=0.7, color='lightgrey', 
                   label='Daily Human Messages', width=0.8)
    
    # Line for weekly average
    if len(weekly_avg) == len(dates):
        line2 = ax2.plot(dates, weekly_avg, color='#FF6B6B', linestyle='--', 
                        linewidth=2.5, label='7-Day Moving Average', alpha=0.9)
    
    ax2.tick_params(axis='y', labelcolor='#ED4245')
    
    # Formatting
    ax1.set_title('Server Growth Analysis: Cumulative Users, Daily Messages, and Weekly Average', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Format x-axis dates - show months only
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Chart saved to {output_path}")


@click.command()
@click.option(
    '--output', '-o',
    default='charts/server_growth.png',
    help='Output path for the chart (default: charts/server_growth.png)'
)
@click.option(
    '--db-path',
    default=None,
    help='Path to database (default: from config)'
)
def main(output: str, db_path: str):
    """Generate server growth analysis chart from day 1."""
    
    settings = Settings()
    if db_path:
        settings.db_path = db_path
    
    if not os.path.exists(settings.db_path):
        logger.error(f"Database not found at {settings.db_path}")
        click.echo(f"❌ Database not found at {settings.db_path}")
        return
    
    logger.info(f"Starting server growth analysis using database: {settings.db_path}")
    
    # Get data facade
    with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
        # Get first activity dates for all users
        first_activity = get_first_activity_dates(facade)
        
        if not first_activity:
            logger.warning("No user activity data found")
            click.echo("❌ No user activity data found in database")
            return
        
        # Get daily human messages
        daily_messages = get_daily_human_messages(facade)
        
        if not daily_messages:
            logger.warning("No daily message data found")
            click.echo("❌ No daily message data found in database")
            return
        
        # Determine date range (from first activity to last message)
        all_dates = set()
        for date_str in first_activity.values():
            all_dates.add(date_str)
        for date_str in daily_messages.keys():
            all_dates.add(date_str)
        
        if not all_dates:
            logger.error("No dates found in data")
            click.echo("❌ No date data found")
            return
        
        sorted_dates = sorted(all_dates)
        start_date = datetime.strptime(sorted_dates[0], '%Y-%m-%d')
        end_date = datetime.strptime(sorted_dates[-1], '%Y-%m-%d')
        
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Calculate cumulative users
        cumulative_users = calculate_cumulative_users(first_activity, start_date, end_date)
        
        # Create date list for plotting
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Calculate weekly average
        weekly_avg = calculate_weekly_average(daily_messages, dates)
        
        # Generate chart
        create_growth_chart(cumulative_users, daily_messages, weekly_avg, dates, output)
        
        # Print summary statistics
        click.echo("\n📊 Server Growth Analysis Summary:")
        click.echo(f"  First activity date: {start_date.strftime('%Y-%m-%d')}")
        click.echo(f"  Last activity date: {end_date.strftime('%Y-%m-%d')}")
        click.echo(f"  Total days analyzed: {len(dates)}")
        click.echo(f"  Total unique users: {len(first_activity)}")
        click.echo(f"  Final cumulative users: {cumulative_users.get(sorted_dates[-1], 0)}")
        click.echo(f"  Total human messages: {sum(daily_messages.values())}")
        click.echo(f"  Average daily messages: {np.mean(list(daily_messages.values())):.1f}")
        click.echo(f"  Final weekly average: {weekly_avg[-1]:.1f} messages/day")
        click.echo(f"\n✅ Chart saved to: {output}")


if __name__ == '__main__':
    main()

