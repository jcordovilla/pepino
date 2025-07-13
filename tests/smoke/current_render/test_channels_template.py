"""
Smoke tests for channels templates.
"""

import pytest
from pathlib import Path

from .conftest import normalize_output


class TestChannelsTemplate:
    """Test channels template rendering consistency."""
    
    def test_channels_list_cli_template(self, current_analysis_service, sample_channels_data):
        """Test CLI channels list template rendering."""
        template_name = "outputs/cli/channel_list.txt.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            channels=sample_channels_data['channels']
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "channel_list_cli.txt"
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
    
    def test_channels_list_discord_template(self, current_analysis_service, sample_channels_data):
        """Test Discord channels list template rendering."""
        template_name = "outputs/discord/channel_list.md.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            channels=sample_channels_data['channels']
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "channel_list_discord.md"
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
    
    def test_top_channels_cli_template(self, current_analysis_service, sample_top_channels_data):
        """Test CLI top channels template rendering."""
        template_name = "outputs/cli/top_channels.txt.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            data=sample_top_channels_data['data'],
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "top_channels_cli.txt"
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
    
    def test_top_channels_discord_template(self, current_analysis_service, sample_top_channels_data):
        """Test Discord top channels template rendering."""
        template_name = "outputs/discord/top_channels.md.j2"
        
        # Render template
        output = current_analysis_service.template_engine.render_template(
            template_name,
            data=sample_top_channels_data['data'],
            format_number=lambda v: f"{v:,}",
            now=lambda: "2024-07-12T15:45:00Z"
        )
        
        # Normalize output
        normalized_output = normalize_output(output)
        
        # Load golden output
        golden_path = Path(__file__).parent / "golden_outputs" / "top_channels_discord.md"
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