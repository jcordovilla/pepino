"""
Template Coverage Tests

Tests that verify all templates can be rendered with the comprehensive fixtures.
This ensures that the data provided to templates is actually fed by the template engine.
"""

import pytest
from pepino.analysis.templates.template_engine import TemplateEngine
from .comprehensive_conftest import (
    sample_pulsecheck_data,
    sample_top_contributors_data,
    sample_database_stats_data,
    sample_detailed_temporal_analysis_data,
    sample_detailed_topic_analysis_data,
    sample_detailed_user_analysis_data,
    sample_server_overview_analysis_data,
    sample_activity_trends_analysis_data,
    sample_top_channels_data,
    sample_list_channels_data
)


@pytest.fixture
def template_engine():
    """Get template engine for testing."""
    return TemplateEngine(
        templates_dir="src/pepino/analysis/templates",
        analyzers={},
        data_facade=None,
        nlp_service=None
    )


class TestTemplateRendering:
    """Test that all templates render correctly with comprehensive fixtures."""
    
    def test_channel_analysis_template(self, template_engine, sample_pulsecheck_data):
        """Test channel analysis template rendering."""
        from datetime import datetime
        result = template_engine.render_template(
            "outputs/cli/channel_analysis.txt.j2", 
            channel_name=sample_pulsecheck_data["channel_name"],
            data=sample_pulsecheck_data["data"],
            total_members=sample_pulsecheck_data["total_members"],
            now=datetime.now,
            format_number=lambda v: f"{v:,}"
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        # Check that no template variables are left unresolved
        assert '{{' not in result
        assert '}}' not in result
    
    def test_top_contributors_template(self, template_engine, sample_top_contributors_data):
        """Test top contributors template rendering."""
        result = template_engine.render_template(
            "outputs/cli/top_contributors.txt.j2", 
            **sample_top_contributors_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_database_stats_template(self, template_engine, sample_database_stats_data):
        """Test database stats template rendering."""
        result = template_engine.render_template(
            "outputs/cli/database_stats.txt.j2", 
            **sample_database_stats_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_detailed_temporal_analysis_template(self, template_engine, sample_detailed_temporal_analysis_data):
        """Test detailed temporal analysis template rendering."""
        result = template_engine.render_template(
            "outputs/cli/detailed_temporal_analysis.txt.j2", 
            **sample_detailed_temporal_analysis_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_detailed_topic_analysis_template(self, template_engine, sample_detailed_topic_analysis_data):
        """Test detailed topic analysis template rendering."""
        result = template_engine.render_template(
            "outputs/cli/detailed_topic_analysis.txt.j2", 
            **sample_detailed_topic_analysis_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_detailed_user_analysis_template(self, template_engine, sample_detailed_user_analysis_data):
        """Test detailed user analysis template rendering."""
        from datetime import datetime
        result = template_engine.render_template(
            "outputs/cli/detailed_user_analysis.txt.j2", 
            user_info=sample_detailed_user_analysis_data["user_info"],
            statistics=sample_detailed_user_analysis_data["statistics"],
            channel_activity=sample_detailed_user_analysis_data["channel_activity"],
            time_patterns=sample_detailed_user_analysis_data["time_patterns"],
            messages=sample_detailed_user_analysis_data["messages"],
            summary=sample_detailed_user_analysis_data["summary"],
            now=datetime.now,
            format_number=lambda v: f"{v:,}"
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_server_overview_template(self, template_engine, sample_server_overview_analysis_data):
        """Test server overview template rendering."""
        from datetime import datetime
        result = template_engine.render_template(
            "outputs/discord/server_overview.md.j2", 
            total_messages=sample_server_overview_analysis_data["total_messages"],
            total_users=sample_server_overview_analysis_data["total_users"],
            total_channels=sample_server_overview_analysis_data["total_channels"],
            active_users=sample_server_overview_analysis_data["active_users"],
            messages_per_day=sample_server_overview_analysis_data["messages_per_day"],
            messages_per_user=sample_server_overview_analysis_data["messages_per_user"],
            top_channels=sample_server_overview_analysis_data["top_channels"],
            top_users=sample_server_overview_analysis_data["top_users"],
            format_number=lambda v: f"{v:,}"
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_activity_trends_template(self, template_engine, sample_activity_trends_analysis_data):
        """Test activity trends template rendering."""
        result = template_engine.render_template(
            "outputs/discord/activity_trends.md.j2", 
            **sample_activity_trends_analysis_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_top_channels_template(self, template_engine, sample_top_channels_data):
        """Test top channels template rendering."""
        from datetime import datetime
        result = template_engine.render_template(
            "outputs/cli/top_channels.txt.j2", 
            data=sample_top_channels_data["data"],
            now=datetime.now,
            format_number=lambda v: f"{v:,}"
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result
    
    def test_channel_list_template(self, template_engine, sample_list_channels_data):
        """Test channel list template rendering."""
        result = template_engine.render_template(
            "outputs/cli/channel_list.txt.j2", 
            **sample_list_channels_data
        )
        
        assert result is not None
        assert len(result) > 0
        assert not result.startswith('❌')
        assert '{{' not in result
        assert '}}' not in result


class TestTemplateDataFlow:
    """Test that data flows correctly from analysis services to templates."""
    
    def test_channel_analysis_data_structure(self, sample_pulsecheck_data):
        """Test that channel analysis data has the expected structure."""
        data = sample_pulsecheck_data
        
        # Check required top-level keys
        assert 'channel_name' in data
        assert 'total_members' in data
        assert 'data' in data
        
        # Check nested data structure
        assert 'period' in data['data']
        assert 'user_stats' in data['data']
        assert 'message_stats' in data['data']
        assert 'top_contributors' in data['data']
        
        # Check period data
        assert 'start_date' in data['data']['period']
        assert 'end_date' in data['data']['period']
    
    def test_database_stats_data_structure(self, sample_database_stats_data):
        """Test that database stats data has the expected structure."""
        data = sample_database_stats_data
        
        assert 'database_info' in data
        assert 'table_stats' in data
        assert 'summary' in data
        
        # Check database info
        assert 'file_path' in data['database_info']
        assert 'size_mb' in data['database_info']
        
        # Check table stats is a list
        assert isinstance(data['table_stats'], list)
        
        # Check summary
        assert 'total_messages' in data['summary']
        assert 'total_users' in data['summary']
        assert 'total_channels' in data['summary']
    
    def test_user_analysis_data_structure(self, sample_detailed_user_analysis_data):
        """Test that user analysis data has the expected structure."""
        data = sample_detailed_user_analysis_data
        
        assert 'user_info' in data
        assert 'statistics' in data
        assert 'channel_activity' in data
        assert 'time_patterns' in data
        assert 'messages' in data
        
        # Check user info
        assert 'author_id' in data['user_info']
        assert 'display_name' in data['user_info']
        
        # Check statistics
        assert 'message_count' in data['statistics']
        assert 'channels_active' in data['statistics']
        
        # Check channel activity is a list
        assert isinstance(data['channel_activity'], list)
        
        # Check time patterns is a dict with hourly_activity and daily_activity
        assert isinstance(data['time_patterns'], dict)
        assert 'hourly_activity' in data['time_patterns']
        assert 'daily_activity' in data['time_patterns']
        
        # Check messages is a list
        assert isinstance(data['messages'], list)


def test_template_variable_coverage():
    """Test that all template variables are covered by fixtures."""
    # This test ensures that the comprehensive fixtures provide complete coverage
    # for all template variables used in the templates
    
    import re
    from pathlib import Path
    
    # Get all template files
    template_dir = Path("src/pepino/analysis/templates")
    template_files = list(template_dir.rglob("*.j2"))
    
    # Import comprehensive fixtures
    from .conftest import (
        sample_pulsecheck_data,
        sample_top_contributors_data,
        sample_database_stats_data,
        sample_detailed_temporal_analysis_data,
        sample_detailed_topic_analysis_data,
        sample_detailed_user_analysis_data,
        sample_server_overview_analysis_data,
        sample_activity_trends_analysis_data,
        sample_top_channels_data,
        sample_list_channels_data
    )
    
    # Map templates to their fixtures
    template_fixture_map = {
        'outputs/cli/channel_analysis.txt.j2': sample_pulsecheck_data,
        'outputs/cli/top_contributors.txt.j2': sample_top_contributors_data,
        'outputs/cli/database_stats.txt.j2': sample_database_stats_data,
        'outputs/cli/detailed_temporal_analysis.txt.j2': sample_detailed_temporal_analysis_data,
        'outputs/cli/detailed_topic_analysis.txt.j2': sample_detailed_topic_analysis_data,
        'outputs/cli/detailed_user_analysis.txt.j2': sample_detailed_user_analysis_data,
        'outputs/discord/server_overview.md.j2': sample_server_overview_analysis_data,
        'outputs/discord/activity_trends.md.j2': sample_activity_trends_analysis_data,
        'outputs/cli/top_channels.txt.j2': sample_top_channels_data,
        'outputs/cli/channel_list.txt.j2': sample_list_channels_data
    }
    
    def flatten_dict(d, parent_key='', sep='.'):
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Test each template
    for template_path, fixture_func in template_fixture_map.items():
        full_path = template_dir / template_path
        
        if not full_path.exists():
            continue
            
        # Read template and extract variables
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Find template variables
        variables = set()
        var_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}'
        matches = re.findall(var_pattern, content)
        variables.update(matches)
        
        # Get fixture data
        fixture_data = fixture_func()
        flat_fixture = flatten_dict(fixture_data)
        
        # Check coverage
        missing_vars = []
        for var in variables:
            if var not in flat_fixture:
                missing_vars.append(var)
        
        # Assert complete coverage
        assert len(missing_vars) == 0, f"Template {template_path} has missing variables: {missing_vars}" 