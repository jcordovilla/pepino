"""
Smoke test for legacy topic analysis template rendering.

This test ensures that the legacy topic analysis template renders consistently
during refactoring by comparing against a golden output.
"""

import pytest
from pathlib import Path
from .conftest import save_golden_output, load_golden_output, normalize_output


class TestLegacyTopicTemplate:
    """Test legacy topic analysis template rendering consistency."""
    
    def test_topic_analysis_cli_template_rendering(self, legacy_service, sample_topic_data, golden_outputs_dir):
        """Test that legacy topic analysis CLI template renders consistently."""
        
        # Render the template using current legacy system
        with legacy_service() as service:
            # Use actual service method
            rendered_output = service.analyze_topics(
                channel_name=None,
                n_topics=8,
                days_back=30,
                output_format="cli"
            )
        
        # Normalize the output for comparison
        normalized_output = normalize_output(rendered_output)
        
        # Check if golden output exists
        golden_file = "legacy_topic_analysis_cli.txt"
        golden_path = golden_outputs_dir / golden_file
        
        if not golden_path.exists():
            # First run - save as golden output
            save_golden_output(golden_outputs_dir, golden_file, normalized_output)
            pytest.skip(f"Created golden output file: {golden_file}")
        
        # Load golden output and compare
        golden_output = load_golden_output(golden_outputs_dir, golden_file)
        golden_normalized = normalize_output(golden_output)
        
        # Compare outputs
        assert normalized_output == golden_normalized, (
            f"Topic analysis CLI template output has changed!\n"
            f"Expected:\n{golden_normalized}\n\n"
            f"Actual:\n{normalized_output}"
        )
    
    def test_topic_analysis_discord_template_rendering(self, legacy_service, sample_topic_data, golden_outputs_dir):
        """Test that legacy topic analysis Discord template renders consistently."""
        
        # Render the template using current legacy system
        with legacy_service() as service:
            # Use actual service method
            rendered_output = service.analyze_topics(
                channel_name=None,
                n_topics=8,
                days_back=30,
                output_format="discord"
            )
        
        # Normalize the output for comparison
        normalized_output = normalize_output(rendered_output)
        
        # Check if golden output exists
        golden_file = "legacy_topic_analysis_discord.md"
        golden_path = golden_outputs_dir / golden_file
        
        if not golden_path.exists():
            # First run - save as golden output
            save_golden_output(golden_outputs_dir, golden_file, normalized_output)
            pytest.skip(f"Created golden output file: {golden_file}")
        
        # Load golden output and compare
        golden_output = load_golden_output(golden_outputs_dir, golden_file)
        golden_normalized = normalize_output(golden_output)
        
        # Compare outputs
        assert normalized_output == golden_normalized, (
            f"Topic analysis Discord template output has changed!\n"
            f"Expected:\n{golden_normalized}\n\n"
            f"Actual:\n{normalized_output}"
        )
    
    def test_topic_analysis_template_structure(self, legacy_service, sample_topic_data):
        """Test that topic analysis template contains expected sections."""
        
        with legacy_service() as service:
            rendered_output = service.analyze_topics(
                channel_name=None,
                n_topics=8,
                days_back=30,
                output_format="cli"
            )
        
        # Check for expected sections
        expected_sections = [
            "Topic Analysis Report",
            "Analysis Overview",
            "Top Topics",
            "Topic Evolution",
            "Technical Details"
        ]
        
        for section in expected_sections:
            assert section in rendered_output, f"Missing section: {section}"
        
        # Check for topic data
        assert str(sample_topic_data["analysis_overview"]["total_messages"]) in rendered_output
        assert sample_topic_data["most_discussed_topic"] in rendered_output 