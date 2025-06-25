"""
Topic analysis plugins.
"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..data.config import Settings
from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from .models import AnalysisErrorResponse, TopicAnalysisResponse, TopicItem


class TopicAnalyzer:
    """Topic analysis plugin."""

    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None, base_filter: Optional[str] = None):
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        # Use provided base_filter or fallback to settings
        if base_filter is None:
            from pepino.data.config import Settings
            settings = Settings()
            base_filter = settings.base_filter
        self.base_filter = base_filter

    def analyze(
        self, **kwargs
    ) -> Union[TopicAnalysisResponse, AnalysisErrorResponse]:
        """
        Analyze topics based on provided parameters.

        Expected kwargs:
            - channel_name: str (optional)
            - days_back: int (optional, default 30)
            - min_word_length: int (optional, default 4)
            - top_n: int (optional, default 20)
        """
        try:
            # Get messages for analysis
            messages = self._get_messages_for_analysis(**kwargs)
            if not messages:
                return AnalysisErrorResponse(
                    error="No messages found for topic analysis", plugin="TopicAnalyzer"
                )

            # Extract topics
            topics = self._extract_topics(messages, **kwargs)

            # Create and return response
            return TopicAnalysisResponse(
                topics=[TopicItem(**topic) for topic in topics],
                message_count=len(messages),
            )

        except Exception as e:
            return AnalysisErrorResponse(
                error=f"Analysis failed: {str(e)}", plugin="TopicAnalyzer"
            )

    def _get_messages_for_analysis(self, **kwargs) -> List[str]:
        """Get messages for topic analysis using repository."""
        channel_name = kwargs.get("channel_name")
        days_back = kwargs.get("days_back", 30)
        limit = 1000  # Default limit for topic analysis

        # Use data facade for message repository access (now synchronous)
        return self.data_facade.message_repository.get_messages_for_topic_analysis(
            channel_name=channel_name, days_back=days_back, limit=limit
        )

    def _extract_topics(
        self, messages: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract topics using simple word frequency analysis."""
        min_word_length = kwargs.get("min_word_length", 4)

        # Enhanced stop words list
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "not",
            "no",
            "yes",
            "so",
            "if",
            "then",
            "than",
            "when",
            "where",
            "why",
            "how",
            "just",
            "now",
            "here",
            "there",
            "very",
            "too",
            "also",
            "much",
            "many",
            "more",
            "like",
            "get",
            "got",
            "go",
            "going",
            "come",
            "came",
            "see",
            "know",
            "think",
            "want",
            "need",
            "make",
            "made",
            "take",
            "took",
            "look",
            "say",
            "said",
            "tell",
            "good",
            "bad",
            "new",
            "old",
            "first",
            "last",
            "next",
            "back",
            "up",
            "down",
            "out",
            "over",
            "under",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "through",
            "during",
            "before",
            "after",
        }

        word_freq = Counter()

        for message in messages:
            # Simple preprocessing
            content = message.lower()
            # Remove common punctuation and split
            words = content.replace("\n", " ").replace("\t", " ")
            for punct in '.,!?";:()[]{}/@#$%^&*-_=+|\\<>~`':
                words = words.replace(punct, " ")

            # Count words
            for word in words.split():
                word = word.strip()
                if (
                    len(word) >= min_word_length
                    and word not in stop_words
                    and word.isalpha()
                    and not word.startswith("http")
                ):
                    word_freq[word] += 1

        # Get top topics
        top_words = word_freq.most_common(kwargs.get("top_n", 20))

        return [
            {
                "topic": word,
                "frequency": count,
                "relevance_score": round(count / len(messages), 3),
            }
            for word, count in top_words
        ]
