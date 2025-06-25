#!/usr/bin/env python3
"""
Simple Chart Integration Demo
Shows how templates can generate charts and reference file paths.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Simple chart generation function (no matplotlib dependencies)
def create_sample_chart(data, title):
    """Create a sample chart file and return its path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    
    # Write some dummy content to simulate a chart file
    with open(temp_file.name, 'wb') as f:
        # PNG header bytes (makes it a valid PNG file)
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82')
    
    return temp_file.name

def main():
    """Demonstrate chart integration concepts"""
    
    print("🎨 Simple Chart Integration Demo")
    print("=" * 50)
    
    print("\n1. 📊 Creating Sample Charts")
    print("-" * 40)
    
    # Create sample charts
    activity_chart = create_sample_chart([1, 2, 3], "Activity Chart")
    user_chart = create_sample_chart([10, 20, 30], "User Chart")
    wordcloud_chart = create_sample_chart(["hello", "world"], "Word Cloud")
    
    print(f"✅ Activity chart: {activity_chart}")
    print(f"✅ User chart: {user_chart}")
    print(f"✅ Word cloud: {wordcloud_chart}")
    
    # Verify files exist
    for chart_path in [activity_chart, user_chart, wordcloud_chart]:
        if os.path.exists(chart_path):
            size = os.path.getsize(chart_path)
            print(f"  • {chart_path} ({size} bytes) ✓")
        else:
            print(f"  • {chart_path} (missing) ❌")
    
    print("\n2. 🧪 Template Output with Chart References")
    print("-" * 40)
    
    # Simulate template output that includes chart references
    template_output = f"""# 📊 Channel Analysis: #general

**Analysis Period:** 2024-01-01 to 2024-01-31
**Total Messages:** 1,234
**Active Users:** 45

## 📈 Activity Trends
**Chart Generated:** `{activity_chart}`

Recent activity shows:
- **2024-01-30**: 67 messages
- **2024-01-31**: 89 messages

## 👥 Top Contributors
**User Activity Chart:** `{user_chart}`

1. **Alice** - 234 messages (19.0%)
2. **Bob** - 156 messages (12.7%)

## 🔍 Content Analysis
**Word Cloud:** `{wordcloud_chart}`

**Most Common Topics:**
- python (45 times)
- discord (23 times)
"""
    
    print("Template output:")
    print(template_output)
    
    print("\n3. 🔍 Extracting Chart Paths")
    print("-" * 40)
    
    # Extract chart paths using regex (like our TemplateMessageFormatter does)
    import re
    
    chart_patterns = [
        r'\*\*Chart Generated:\*\* `([^`]+)`',
        r'\*\*.*Chart.*:\*\* `([^`]+)`',
        r'\*\*Word Cloud:\*\* `([^`]+)`',
        r'\*\*Chart File:\*\* `([^`]+)`'
    ]
    
    found_charts = []
    cleaned_text = template_output
    
    for pattern in chart_patterns:
        matches = re.finditer(pattern, template_output)
        for match in matches:
            file_path = match.group(1)
            if os.path.exists(file_path):
                found_charts.append(file_path)
                print(f"✅ Found chart: {file_path}")
            
            # Remove chart reference from text
            cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Clean up extra blank lines
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text.strip())
    
    print(f"\n📊 Extracted {len(found_charts)} chart files")
    print(f"📝 Cleaned text length: {len(cleaned_text)} characters")
    
    print("\n4. 🎯 Discord Integration Pattern")
    print("-" * 40)
    
    print("Here's how this works in Discord:")
    print("1. Template renders markdown text + generates chart files")
    print("2. TemplateMessageFormatter extracts chart paths from text")
    print("3. Discord bot sends text as message content")
    print("4. Discord bot sends charts as file attachments")
    print("5. User sees: markdown text + chart images below")
    
    print("\n5. 🧹 Cleanup")
    print("-" * 40)
    
    # Cleanup chart files
    for chart_path in found_charts:
        try:
            os.remove(chart_path)
            print(f"🗑️ Cleaned up: {chart_path}")
        except Exception as e:
            print(f"❌ Failed to cleanup {chart_path}: {e}")
    
    print("\n6. 🎁 Key Benefits")
    print("-" * 40)
    print("✅ Templates control when charts are generated")
    print("✅ Charts created only when data supports them")
    print("✅ File paths returned for Discord file attachments") 
    print("✅ Clean text output without chart path clutter")
    print("✅ Automatic cleanup after Discord processes files")
    print("✅ Works with any chart type (activity, pie, bar, wordcloud)")
    
    print("\n🎉 Chart integration concept proven!")


if __name__ == "__main__":
    main() 