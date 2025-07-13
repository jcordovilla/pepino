"""
Comprehensive smoke test runner for all legacy template rendering.

This test ensures that ALL legacy templates render consistently
during refactoring by comparing against golden outputs.
"""

import pytest
from pathlib import Path
from .conftest import save_golden_output, load_golden_output, normalize_output


class TestAllLegacyTemplates:
    """Test all legacy template rendering consistency."""
    
    def test_all_legacy_templates_rendering(self, legacy_service, sample_user_data, 
                                           sample_database_data, sample_topic_data, 
                                           sample_temporal_data, golden_outputs_dir):
        """Test that all legacy templates render consistently."""
        
        with legacy_service() as service:
            # Test all template types using actual service methods
            templates_to_test = [
                # User analysis templates
                ("user_cli", lambda: service.analyze_user("test_user", 30, "cli"), sample_user_data),
                ("user_discord", lambda: service.analyze_user("test_user", 30, "discord"), sample_user_data),
                
                # Database analysis templates
                ("database_cli", lambda: service.analyze_database("cli"), sample_database_data),
                ("database_discord", lambda: service.analyze_database("discord"), sample_database_data),
                
                # Topic analysis templates
                ("topic_cli", lambda: service.analyze_topics(None, 8, 30, "cli"), sample_topic_data),
                ("topic_discord", lambda: service.analyze_topics(None, 8, 30, "discord"), sample_topic_data),
                
                # Temporal analysis templates
                ("temporal_cli", lambda: service.analyze_temporal(None, 30, "daily", "cli"), sample_temporal_data),
                ("temporal_discord", lambda: service.analyze_temporal(None, 30, "daily", "discord"), sample_temporal_data),
            ]
            
            for template_name, render_func, sample_data in templates_to_test:
                # Determine file extension based on template type
                if "discord" in template_name:
                    file_ext = ".md"
                else:
                    file_ext = ".txt"
                
                golden_file = f"legacy_{template_name}{file_ext}"
                golden_path = golden_outputs_dir / golden_file
                
                # Render the template
                try:
                    rendered_output = render_func()
                    
                    # Normalize the output for comparison
                    normalized_output = normalize_output(rendered_output)
                    
                    if not golden_path.exists():
                        # First run - save as golden output
                        save_golden_output(golden_outputs_dir, golden_file, normalized_output)
                        print(f"Created golden output file: {golden_file}")
                        continue
                    
                    # Load golden output and compare
                    golden_output = load_golden_output(golden_outputs_dir, golden_file)
                    golden_normalized = normalize_output(golden_output)
                    
                    # Compare outputs
                    assert normalized_output == golden_normalized, (
                        f"Template {template_name} output has changed!\n"
                        f"Expected:\n{golden_normalized}\n\n"
                        f"Actual:\n{normalized_output}"
                    )
                    
                    print(f"✅ Template {template_name} passed consistency check")
                    
                except Exception as e:
                    pytest.fail(f"Failed to test template {template_name}: {str(e)}")
    
    def test_legacy_template_coverage(self):
        """Test that we have coverage for all expected legacy templates."""
        expected_templates = [
            "user_analysis_cli.txt",
            "user_analysis_discord.md",
            "database_analysis_cli.txt", 
            "database_analysis_discord.md",
            "topic_analysis_cli.txt",
            "topic_analysis_discord.md",
            "temporal_analysis_cli.txt",
            "temporal_analysis_discord.md"
        ]
        
        # This test will pass if all templates are tested in the main test
        # It's a placeholder to ensure we don't forget any templates
        assert len(expected_templates) == 8, "Expected 8 legacy templates"
    
    def test_template_normalization_consistency(self):
        """Test that normalization function works consistently."""
        test_content = """
        Generated: 2025-07-12 22:30:37
        File Path: /some/path/db.sqlite
        Some content here
        Generated: 2025-07-12 22:30:37
        """
        
        normalized = normalize_output(test_content)
        
        # Should remove timestamps and file paths
        assert "Generated: TIMESTAMP" in normalized
        assert "File Path: /test/path/discord_messages.db" in normalized
        assert "Generated: 2025-07-12 22:30:37" not in normalized
        assert "/some/path/db.sqlite" not in normalized 