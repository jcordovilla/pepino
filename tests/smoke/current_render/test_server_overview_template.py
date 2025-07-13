"""
Smoke tests for server overview and activity trends templates.
"""

import pytest
from pathlib import Path

from .conftest import normalize_output


class TestServerOverviewTemplate:
    """Test server overview template rendering consistency."""
    
    def test_server_overview_discord_template(self, current_analysis_service, sample_server_overview_data):
        """Test Discord server overview template rendering."""
        template_name = "outputs/discord/server_overview.md.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            **sample_server_overview_data,
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "server_overview_discord.md"
        if golden_path.exists():
            golden_output = golden_path.read_text()
            golden_normalized = normalize_output(golden_output)
            
            assert normalized_output == golden_normalized, (
                f"Template {template_name} output has changed!\n"
                f"Expected:\n{golden_output}\n\n"
                f"Actual:\n{output}"
            )
        else:
            # Create golden output on first run
            golden_path.parent.mkdir(exist_ok=True)
            golden_path.write_text(output)
            print(f"✅ Created golden output for {template_name}")
        
        print(f"✅ Template {template_name} passed consistency check")
    
    def test_activity_trends_discord_template(self, current_analysis_service, sample_activity_trends_data):
        """Test Discord activity trends template rendering."""
        template_name = "outputs/discord/activity_trends.md.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            **sample_activity_trends_data,
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "activity_trends_discord.md"
        if golden_path.exists():
            golden_output = golden_path.read_text()
            golden_normalized = normalize_output(golden_output)
            
            assert normalized_output == golden_normalized, (
                f"Template {template_name} output has changed!\n"
                f"Expected:\n{golden_output}\n\n"
                f"Actual:\n{output}"
            )
        else:
            # Create golden output on first run
            golden_path.parent.mkdir(exist_ok=True)
            golden_path.write_text(output)
            print(f"✅ Created golden output for {template_name}")
        
        print(f"✅ Template {template_name} passed consistency check") 