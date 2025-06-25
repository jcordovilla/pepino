"""
Chart generation utilities for Discord analysis.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
import tempfile
from datetime import datetime
from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Use a temp directory for saving charts
TEMP_DIR = tempfile.mkdtemp()

# Set the style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_activity_graph(
    dates: List[str],
    counts: List[int],
    title: str = "Activity Graph",
    xlabel: str = "Date",
    ylabel: str = "Count",
) -> str:
    """Create a line graph for activity over time."""

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert string dates to datetime objects
    date_objects = []
    for date_str in dates:
        try:
            date_objects.append(datetime.fromisoformat(date_str))
        except:
            # Fallback for different date formats
            try:
                date_objects.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except:
                date_objects.append(datetime.now())

    # Plot the data
    ax.plot(date_objects, counts, marker="o", linewidth=2, markersize=4)

    # Format the plot
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
    plt.close()

    return temp_file.name


def create_channel_activity_pie(
    names: List[str], counts: List[int], title: str = "Channel Activity"
) -> str:
    """Create a pie chart for channel or user activity distribution."""

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Limit to top contributors and group others
    if len(names) > 8:
        top_names = names[:7]
        top_counts = counts[:7]
        others_count = sum(counts[7:])

        if others_count > 0:
            top_names.append("Others")
            top_counts.append(others_count)
    else:
        top_names = names
        top_counts = counts

    # Create pie chart
    colors = sns.color_palette("husl", len(top_names))
    wedges, texts, autotexts = ax.pie(
        top_counts, labels=top_names, autopct="%1.1f%%", colors=colors, startangle=90
    )

    # Enhance text
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis("equal")

    # Adjust layout
    plt.tight_layout()

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
    plt.close()

    return temp_file.name


def create_user_activity_bar(
    names: List[str], counts: List[int], title: str = "User Activity"
) -> str:
    """Create a horizontal bar chart for user activity."""

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.5)))

    # Limit to top users
    if len(names) > 15:
        names = names[:15]
        counts = counts[:15]

    # Create horizontal bar chart
    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(range(len(names)), counts, color=colors)

    # Customize the plot
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Message Count", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # Add grid
    ax.grid(True, alpha=0.3, axis="x")

    # Invert y-axis to show highest values at top
    ax.invert_yaxis()

    # Adjust layout
    plt.tight_layout()

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
    plt.close()

    return temp_file.name


def create_word_cloud(text: str, title: str = "Word Cloud") -> Optional[str]:
    """Create a word cloud from text content."""
    try:
        if not text or len(text.strip()) < 10:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            random_state=42,
        ).generate(text)

        # Display the word cloud
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Adjust layout
        plt.tight_layout()

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
        plt.close()

        return temp_file.name

    except ImportError:
        print("WordCloud not available. Install with: pip install wordcloud")
        return None
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None


def cleanup_chart(file_path: str) -> None:
    """Clean up temporary chart file."""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up chart file {file_path}: {e}")
