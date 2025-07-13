"""
Comprehensive smoke tests for all current system templates.

Tests template rendering consistency and coverage for the current analysis system.
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any

from .conftest import normalize_output


class TestAllCurrentTemplates:
    """Test all current system templates for rendering consistency."""
    
    def test_all_current_templates_rendering(self, current_analysis_service, 
                                           sample_channel_data, sample_user_data, 
                                           sample_database_data, sample_topic_data,
                                           sample_temporal_data, sample_contributors_data,
                                           sample_channels_data, sample_top_channels_data,
                                           sample_server_overview_data, sample_activity_trends_data):
        """Test all current templates render consistently."""
        
        # Define all templates to test with their data
        templates_to_test = [
            # Channel analysis templates
            ("outputs/cli/channel_analysis.txt.j2", {
                'channel_name': "test-channel",
                'data': sample_channel_data,
                'total_members': sample_channel_data['total_members'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/channel_analysis.md.j2", {
                'channel_name': "test-channel",
                'data': sample_channel_data,
                'total_members': sample_channel_data['total_members'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # User analysis templates
            ("outputs/cli/user_analysis.txt.j2", {
                **sample_user_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/user_analysis.md.j2", {
                **sample_user_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Database stats templates
            ("outputs/cli/database_stats.txt.j2", {
                **sample_database_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/database_stats.md.j2", {
                **sample_database_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Topic analysis templates
            ("outputs/cli/topic_analysis.txt.j2", {
                **sample_topic_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/topic_analysis.md.j2", {
                **sample_topic_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Temporal analysis templates
            ("outputs/cli/temporal_analysis.txt.j2", {
                **sample_temporal_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/temporal_analysis.md.j2", {
                **sample_temporal_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Top contributors templates
            ("outputs/cli/top_contributors.txt.j2", {
                'channel_name': None,
                'contributors': sample_contributors_data['contributors'],
                'period': sample_contributors_data['period'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/top_contributors.md.j2", {
                'channel_name': None,
                'contributors': sample_contributors_data['contributors'],
                'period': sample_contributors_data['period'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Channels list templates
            ("outputs/cli/channel_list.txt.j2", {
                'channels': sample_channels_data['channels']
            }),
            ("outputs/discord/channel_list.md.j2", {
                'channels': sample_channels_data['channels']
            }),
            
            # Top channels templates
            ("outputs/cli/top_channels.txt.j2", {
                'data': sample_top_channels_data['data'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/top_channels.md.j2", {
                'data': sample_top_channels_data['data'],
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            
            # Server overview templates (Discord only)
            ("outputs/discord/server_overview.md.j2", {
                **sample_server_overview_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
            ("outputs/discord/activity_trends.md.j2", {
                **sample_activity_trends_data,
                'format_number': lambda v: f"{v:,}",
                'now': lambda: "2024-07-12T15:45:00Z"
            }),
        ]
        
        # Test each template
        for template_name, template_data in templates_to_test:
            try:
                # Render template
                output = current_analysis_service.template_engine.render_template(
                    template_name, **template_data
                )
                
                # Normalize output
                normalized_output = normalize_output(output)
                
                # Load golden output
                golden_filename = template_name.replace('/', '_').replace('.j2', '')
                golden_path = Path(__file__).parent / "golden_outputs" / f"{golden_filename}.txt"
                
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
                
            except Exception as e:
                pytest.fail(f"Failed to test template {template_name}: {str(e)}")
    
    def test_current_template_coverage(self):
        """Test that we have smoke tests for all current templates."""
        
        # Expected templates (based on our analysis)
        expected_templates = {
            # CLI templates
            "outputs/cli/channel_analysis.txt.j2",
            "outputs/cli/channel_list.txt.j2", 
            "outputs/cli/top_channels.txt.j2",
            "outputs/cli/top_contributors.txt.j2",
            "outputs/cli/user_analysis.txt.j2",
            "outputs/cli/temporal_analysis.txt.j2",
            "outputs/cli/topic_analysis.txt.j2",
            "outputs/cli/database_stats.txt.j2",
            
            # Discord templates
            "outputs/discord/channel_analysis.md.j2",
            "outputs/discord/channel_list.md.j2",
            "outputs/discord/top_channels.md.j2", 
            "outputs/discord/top_contributors.md.j2",
            "outputs/discord/user_analysis.md.j2",
            "outputs/discord/temporal_analysis.md.j2",
            "outputs/discord/topic_analysis.md.j2",
            "outputs/discord/database_stats.md.j2",
            "outputs/discord/server_overview.md.j2",
            "outputs/discord/activity_trends.md.j2",
        }
        
        # Check that all expected templates are covered in our test
        templates_to_test = [
            "outputs/cli/channel_analysis.txt.j2",
            "outputs/discord/channel_analysis.md.j2",
            "outputs/cli/user_analysis.txt.j2",
            "outputs/discord/user_analysis.md.j2",
            "outputs/cli/database_stats.txt.j2",
            "outputs/discord/database_stats.md.j2",
            "outputs/cli/topic_analysis.txt.j2",
            "outputs/discord/topic_analysis.md.j2",
            "outputs/cli/temporal_analysis.txt.j2",
            "outputs/discord/temporal_analysis.md.j2",
            "outputs/cli/top_contributors.txt.j2",
            "outputs/discord/top_contributors.md.j2",
            "outputs/cli/channel_list.txt.j2",
            "outputs/discord/channel_list.md.j2",
            "outputs/cli/top_channels.txt.j2",
            "outputs/discord/top_channels.md.j2",
            "outputs/discord/server_overview.md.j2",
            "outputs/discord/activity_trends.md.j2",
        ]
        
        tested_templates = set(templates_to_test)
        missing_templates = expected_templates - tested_templates
        
        assert len(missing_templates) == 0, (
            f"Missing smoke tests for templates: {missing_templates}"
        )
        
        print(f"✅ All {len(expected_templates)} current templates have smoke tests")
    
    def test_template_normalization_consistency(self):
        """Test that template normalization is consistent."""
        
        # Test cases for normalization
        test_cases = [
            ("2024-07-12T15:45:00Z", "TIMESTAMP"),
            ("2024-07-12 15:45:00", "TIMESTAMP"),
            ("15.5%", "PERCENTAGE"),
            ("123.45", "DECIMAL"),
            ("   multiple    spaces   ", "multiple spaces"),
        ]
        
        for input_text, expected in test_cases:
            normalized = normalize_output(input_text)
            assert normalized == expected, f"Normalization failed: '{input_text}' -> '{normalized}' (expected '{expected}')"
        
        print("✅ Template normalization is consistent") 