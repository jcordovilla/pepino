import pytest
from pepino.analysis.service import AnalysisService

def test_detailed_user_analysis_cli():
    service = AnalysisService()
    output = service.detailed_user_analysis("oscarsan.chez", days_back=30, output_format="cli")
    assert "User Analysis" in output
    assert "Statistical Summary" in output
    assert "Channel Activity" in output
    assert "Activity Patterns" in output
    assert "Content Analysis" in output
    assert "Summary" in output
    assert "User Profile" in output
    assert "oscarsan.chez" in output 