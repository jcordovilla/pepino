"""
Smoke test for legacy temporal analysis template rendering.

This test ensures that the legacy temporal analysis template renders consistently
during refactoring by comparing against a golden output.
"""

import pytest
from pathlib import Path
from .conftest import save_golden_output, load_golden_output, normalize_output


class TestLegacyTemporalTemplate:
    """Test legacy temporal analysis template rendering consistency."""
    
    def test_temporal_analysis_cli_template_rendering(self, legacy_service, sample_temporal_data, golden_outputs_dir):
        """Test that legacy temporal analysis CLI template renders consistently."""
        
        # Render the template using current legacy system
        with legacy_service() as service:
            # Use actual service method
            rendered_output = service.analyze_temporal(
                channel_name=None,
                days_back=30,
                granularity="daily",
                output_format="cli"
            )
        
        # Normalize the output for comparison
        normalized_output = normalize_output(rendered_output)
        
        # Check if golden output exists
        golden_file = "legacy_temporal_analysis_cli.txt"
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
            f"Temporal analysis CLI template output has changed!\n"
            f"Expected:\n{golden_normalized}\n\n"
            f"Actual:\n{normalized_output}"
        )
    
    def test_temporal_analysis_discord_template_rendering(self, legacy_service, sample_temporal_data, golden_outputs_dir):
        """Test that legacy temporal analysis Discord template renders consistently."""
        
        # Render the template using current legacy system
        with legacy_service() as service:
            # Use actual service method
            rendered_output = service.analyze_temporal(
                channel_name=None,
                days_back=30,
                granularity="daily",
                output_format="discord"
            )
        
        # Normalize the output for comparison
        normalized_output = normalize_output(rendered_output)
        
        # Check if golden output exists
        golden_file = "legacy_temporal_analysis_discord.md"
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
            f"Temporal analysis Discord template output has changed!\n"
            f"Expected:\n{golden_normalized}\n\n"
            f"Actual:\n{normalized_output}"
        )
    
    def test_temporal_analysis_template_structure(self, legacy_service, sample_temporal_data):
        """Test that temporal analysis template contains expected sections."""
        
        with legacy_service() as service:
            rendered_output = service.analyze_temporal(
                channel_name=None,
                days_back=30,
                granularity="daily",
                output_format="cli"
            )
        
        # Check for expected sections
        expected_sections = [
            "Temporal Analysis Report",
            "Analysis Overview",
            "Activity Patterns",
            "Pattern Analysis",
            "Statistical Summary",
            "Insights"
        ]
        
        for section in expected_sections:
            assert section in rendered_output, f"Missing section: {section}"
        
        # Check for temporal data
        assert str(sample_temporal_data["statistical_summary"]["total_messages"]) in rendered_output
        assert str(sample_temporal_data["statistical_summary"]["peak_activity"]) in rendered_output 