#!/usr/bin/env python3
"""
Analysis Data Facade Demo

Demonstrates the new data facade pattern for analysis operations.
Shows how to use centralized repository management with transaction support.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.analysis import (
    AnalysisDataFacade,
    UserAnalyzer,
    ChannelAnalyzer,
    TopicAnalyzer,
    TemporalAnalyzer,
    get_analysis_data_facade,
    analysis_transaction,
)


def demo_basic_facade_usage():
    """Demonstrate basic data facade usage."""
    print("=== Basic Data Facade Usage ===")
    
    # Option 1: Create facade explicitly
    with get_analysis_data_facade() as facade:
        print(f"Created data facade with database: {facade.db_manager.db_path}")
        
        # Use analyzers with the shared facade
        user_analyzer = UserAnalyzer(facade)
        channel_analyzer = ChannelAnalyzer(facade)
        
        # All analyzers share the same database connection and repositories
        users = user_analyzer.get_available_users()[:5]
        channels = channel_analyzer.get_available_channels()[:5]
        
        print(f"Found {len(users)} users (showing first 5): {users}")
        print(f"Found {len(channels)} channels (showing first 5): {channels}")


def demo_auto_facade_creation():
    """Demonstrate automatic facade creation."""
    print("\n=== Automatic Facade Creation ===")
    
    # Option 2: Let analyzers create their own facades
    user_analyzer = UserAnalyzer()  # Creates its own facade automatically
    channel_analyzer = ChannelAnalyzer()  # Creates its own facade automatically
    
    users = user_analyzer.get_available_users()[:3]
    channels = channel_analyzer.get_available_channels()[:3]
    
    print(f"User analyzer found: {users}")
    print(f"Channel analyzer found: {channels}")


def demo_transactional_operations():
    """Demonstrate transactional operations with the data facade."""
    print("\n=== Transactional Operations ===")
    
    # Use transaction context for operations requiring consistency
    try:
        with analysis_transaction() as facade:
            print("Starting transaction...")
            
            # All operations within this block are part of the same transaction
            user_analyzer = UserAnalyzer(facade)
            channel_analyzer = ChannelAnalyzer(facade)
            
            # Get data that should be consistent
            users = user_analyzer.get_available_users()
            channels = channel_analyzer.get_available_channels()
            
            print(f"Transaction completed - users: {len(users)}, channels: {len(channels)}")
            
    except Exception as e:
        print(f"Transaction would be rolled back on error: {e}")


def demo_shared_facade_multiple_analyzers():
    """Demonstrate sharing a single facade across multiple analyzers."""
    print("\n=== Shared Facade Across Analyzers ===")
    
    with get_analysis_data_facade() as facade:
        # Create all analyzers with the same facade
        analyzers = {
            'user': UserAnalyzer(facade),
            'channel': ChannelAnalyzer(facade),
            'topic': TopicAnalyzer(facade),
            'temporal': TemporalAnalyzer(facade),
        }
        
        print("Created analyzers sharing the same data facade:")
        for name, analyzer in analyzers.items():
            print(f"  - {name.capitalize()}Analyzer: {type(analyzer).__name__}")
        
        # All analyzers use the same database connection and repository instances
        print(f"All analyzers share database: {facade.db_manager.db_path}")


def demo_repository_access():
    """Demonstrate direct repository access through the facade."""
    print("\n=== Direct Repository Access ===")
    
    with get_analysis_data_facade() as facade:
        # Access repositories directly through the facade
        print("Available repositories:")
        print(f"  - User Repository: {type(facade.user_repository).__name__}")
        print(f"  - Channel Repository: {type(facade.channel_repository).__name__}")
        print(f"  - Message Repository: {type(facade.message_repository).__name__}")
        print(f"  - Embedding Repository: {type(facade.embedding_repository).__name__}")
        
        # Use repositories directly if needed
        user_stats = facade.user_repository.get_top_users_by_message_count(limit=3)
        channel_stats = facade.channel_repository.get_top_channels_by_message_count(limit=3)
        
        print(f"\nDirect repository usage:")
        print(f"  Top users: {[u['author_name'] for u in user_stats]}")
        print(f"  Top channels: {[c['channel_name'] for c in channel_stats]}")


if __name__ == "__main__":
    print("🔍 Analysis Data Facade Demo")
    print("=" * 50)
    
    try:
        demo_basic_facade_usage()
        demo_auto_facade_creation()
        demo_transactional_operations()
        demo_shared_facade_multiple_analyzers()
        demo_repository_access()
        
        print("\n✅ All demos completed successfully!")
        print("\nKey Benefits of the Data Facade:")
        print("• Centralized repository management")
        print("• Automatic database connection handling")
        print("• Transaction support for consistency")
        print("• Resource cleanup with context managers")
        print("• Shared connections across analyzers")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 