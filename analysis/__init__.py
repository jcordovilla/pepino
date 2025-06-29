"""
Analysis utilities for Discord message analysis.

This module provides embedding operations, topic analysis, statistical analysis,
and insights functionality using machine learning models and advanced NLP.
"""

from .embeddings import (
    EmbeddingManager,
    embedding_manager,
    ensure_model_loaded,
    get_embedding,
    generate_message_embeddings,
    find_similar_messages_data
)

from .topics import (
    analyze_topics_spacy,
    extract_concepts_from_content,
    filter_boilerplate_phrases,
    perform_topic_modeling
)

from .statistics import (
    update_user_statistics,
    update_word_frequencies,
    update_conversation_chains,
    update_temporal_stats,
    run_all_analyses
)

from .insights import (
    resolve_channel_name,
    get_user_insights,
    get_channel_insights
)

__all__ = [
    # Embeddings
    "EmbeddingManager",
    "embedding_manager",
    "ensure_model_loaded",
    "get_embedding", 
    "generate_message_embeddings",
    "find_similar_messages_data",
    
    # Topics
    "analyze_topics_spacy",
    "extract_concepts_from_content",
    "filter_boilerplate_phrases",
    "perform_topic_modeling",
    
    # Statistics
    "update_user_statistics",
    "update_word_frequencies",
    "update_conversation_chains",
    "update_temporal_stats",
    "run_all_analyses",
    
    # Insights
    "resolve_channel_name",
    "get_user_insights",
    "get_channel_insights"
]
