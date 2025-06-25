#!/usr/bin/env python3
"""
Demo: Template Chart Integration
Shows how templates can generate charts and reference file paths.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.discord.templates.template_engine import AnalysisTemplateEngine
from pepino.discord.templates.sync_template_executor import SyncTemplateExecutor
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.data.database.manager import DatabaseManager
import tempfile

def main():
    """Demonstrate chart integration in templates"""
    
    print("🎨 Template Chart Integration Demo")
    print("=" * 50)
    
    # Initialize database and analyzer
    db_manager = DatabaseManager("discord_messages.db")
    analyzer = ChannelAnalyzer(db_manager)
    
    # Create template executor with chart-enabled engine
    executor = SyncTemplateExecutor(
        templates_dir="templates",
        analyzers={
            'channel_analyzer': analyzer
        }
    )
    
    print("\n1. 📊 Testing Chart Generation Functions")
    print("-" * 40)
    
    # Test chart generation directly
    engine = executor.template_engine
    
    # Sample data for charts
    dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    counts = [45, 67, 23, 89, 56]
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    
    # Test each chart function
    activity_chart = engine._create_activity_chart(dates, counts, "Sample Activity")
    pie_chart = engine._create_pie_chart(names, counts, "Sample Distribution") 
    bar_chart = engine._create_bar_chart(names, counts, "Sample Ranking")
    wordcloud = engine._create_wordcloud("hello world python discord analysis charts template", "Sample Words")
    
    print(f"✅ Activity chart: {activity_chart}")
    print(f"✅ Pie chart: {pie_chart}")
    print(f"✅ Bar chart: {bar_chart}")
    print(f"✅ Word cloud: {wordcloud}")
    
    print(f"\n📁 Chart files exist:")
    for chart_path in [activity_chart, pie_chart, bar_chart, wordcloud]:
        if chart_path and os.path.exists(chart_path):
            size = os.path.getsize(chart_path)
            print(f"  • {chart_path} ({size:,} bytes)")
        else:
            print(f"  • {chart_path} (missing)")
    
    print("\n2. 🧪 Testing Template with Real Data")
    print("-" * 40)
    
    try:
        # Get real channel data
        channels = analyzer.get_available_channels()
        if channels:
            channel_name = channels[0]['name']
            
            # Analyze channel and render template with charts
            result = executor.render_channel_analysis(
                channel_name=channel_name,
                days_back=30,
                include_chart=True  # This parameter should trigger chart generation
            )
            
            print(f"✅ Rendered template for #{channel_name}")
            print(f"📄 Template length: {len(result):,} characters")
            
            # Look for chart paths in the output
            chart_references = []
            for line in result.split('\n'):
                if 'Chart Generated:' in line or 'Chart:' in line:
                    chart_references.append(line.strip())
            
            if chart_references:
                print(f"🎯 Found {len(chart_references)} chart references:")
                for ref in chart_references:
                    print(f"  • {ref}")
            else:
                print("⚠️  No chart references found in template output")
            
            # Save result to file for inspection
            output_file = f"temp/channel_chart_demo_{channel_name}.md"
            os.makedirs("temp", exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"💾 Full output saved to: {output_file}")
            
        else:
            print("❌ No channels available for testing")
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
    
    print("\n3. 🔧 Template Code Example")
    print("-" * 40)
    
    example_template = '''
# Chart Integration Example

{% set dates = ['2024-01-01', '2024-01-02', '2024-01-03'] %}
{% set counts = [45, 67, 23] %}
{% set chart_path = create_activity_chart(dates, counts, "Daily Activity") %}

## Activity Chart
{% if chart_path %}
**Chart File:** `{{ chart_path }}`
![Activity Chart]({{ chart_path }})
{% else %}
No chart generated.
{% endif %}

The chart file path is: {{ chart_path }}
'''
    
    print("Template code that generates charts:")
    print(example_template)
    
    # Test the example template
    try:
        from jinja2 import Template
        template = Template(example_template)
        
        # Add chart functions to template globals
        template.globals.update({
            'create_activity_chart': engine._create_activity_chart,
            'create_pie_chart': engine._create_pie_chart,
            'create_bar_chart': engine._create_bar_chart,
            'create_wordcloud': engine._create_wordcloud
        })
        
        rendered = template.render()
        print("\nRendered result:")
        print(rendered)
        
    except Exception as e:
        print(f"❌ Error testing example template: {e}")
    
    print("\n4. 🎯 Key Benefits")
    print("-" * 40)
    print("✅ Templates can generate charts on-demand")
    print("✅ Chart file paths returned for Discord attachment")
    print("✅ Charts created only when template requests them")
    print("✅ Full integration: data → chart → template → Discord")
    print("✅ Clean separation: templates handle logic, charts handle visuals")
    
    print("\n🎉 Chart integration demo complete!")


if __name__ == "__main__":
    main() 