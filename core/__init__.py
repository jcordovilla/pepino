"""
Core text processing and NLP utilities for Discord message analysis.

This module provides text preprocessing, cleaning, and advanced NLP operations
using spaCy for concept extraction and pattern matching.
"""

from .text_processing import (
    download_nltk_data,
    load_spacy_model, 
    preprocess_text,
    clean_content,
    clean_content_extended,
    extract_complex_concepts,
    get_analysis_patterns,
    get_topic_analysis_patterns,
    nlp
)

__all__ = [
    "download_nltk_data",
    "load_spacy_model", 
    "preprocess_text",
    "clean_content", 
    "clean_content_extended",
    "extract_complex_concepts",
    "get_analysis_patterns",
    "get_topic_analysis_patterns", 
    "nlp"
]
