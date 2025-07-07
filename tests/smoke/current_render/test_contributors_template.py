"""
Smoke tests for top contributors templates.
"""

import pytest
from pathlib import Path

from .conftest import normalize_output


class TestContributorsTemplate:
    """Test top contributors template rendering consistency."""
    
    def test_contributors_cli_template(self, current_analysis_service, sample_contributors_data):
        """Test CLI top contributors template rendering."""
        template_name = "outputs/cli/top_contributors.txt.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            channel_name=None,
            contributors=sample_contributors_data['contributors'],
            period=sample_contributors_data['period'],
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "top_contributors_cli.txt"
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
    
    def test_contributors_discord_template(self, current_analysis_service, sample_contributors_data):
        """Test Discord top contributors template rendering."""
        template_name = "outputs/discord/top_contributors.md.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            channel_name=None,
            contributors=sample_contributors_data['contributors'],
            period=sample_contributors_data['period'],
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "top_contributors_discord.md"
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