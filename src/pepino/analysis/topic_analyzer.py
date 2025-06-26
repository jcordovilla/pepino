"""
Topic analysis plugins.
"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from .models import AnalysisErrorResponse, TopicAnalysisResponse, TopicItem


class TopicAnalyzer:
    """
    Topic analysis plugin.
    
    Analyzes topics and word patterns in message data using the data facade pattern
    for centralized repository management and proper separation of concerns.
    
    All database operations are abstracted through the data facade for proper
    dependency injection support and testability.
    """

    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize topic analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False

    def analyze(
        self,
        channel_name: Optional[str] = None,
        days_back: int = 30,
        min_word_length: int = 4,
        top_n: int = 20
    ) -> Union[TopicAnalysisResponse, AnalysisErrorResponse]:
        """
        Analyze topics based on provided parameters.

        Args:
            channel_name: Optional channel to analyze (None for all channels)
            days_back: Number of days to look back (default 30)
            min_word_length: Minimum word length to consider (default 4)
            top_n: Number of top topics to return (default 20)
        """
        try:
            # Get messages for analysis
            messages = self._get_messages_for_analysis(
                channel_name=channel_name,
                days_back=days_back
            )
            if not messages:
                return AnalysisErrorResponse(
                    error="No messages found for topic analysis", plugin="TopicAnalyzer"
                )

            # Extract topics with enhanced algorithm
            topics = self._extract_top_topics(messages, top_n, min_word_length)

            # Create and return response
            return TopicAnalysisResponse(
                topics=[TopicItem(**topic) for topic in topics],
                message_count=len(messages),
            )

        except Exception as e:
            return AnalysisErrorResponse(
                error=f"Analysis failed: {str(e)}", plugin="TopicAnalyzer"
            )

    def _get_messages_for_analysis(
        self,
        channel_name: Optional[str] = None,
        days_back: int = 30,
    ) -> List[str]:
        """Get messages for topic analysis using repository."""
        # Use the correct repository method that exists
        return self.data_facade.message_repository.get_messages_for_topic_analysis(
            channel_name=channel_name,
            days_back=days_back,
            limit=1000
        )

    def _extract_top_topics(
        self, messages: List[str], top_n: int = 20, min_word_length: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Enhanced topic extraction focusing on meaningful discussion topics.
        
        Uses a three-tier approach:
        1. Compound technology terms (e.g., "Generative AI", "Machine Learning")
        2. Single meaningful keywords (technical terms, concepts)
        3. Domain-specific terms and acronyms
        """
        # Enhanced compound terms for AI/tech discussions
        compound_terms = {
            'generative ai': 'Generative AI',
            'artificial intelligence': 'Artificial Intelligence', 
            'machine learning': 'Machine Learning',
            'deep learning': 'Deep Learning',
            'neural network': 'Neural Networks',
            'large language model': 'Large Language Models',
            'natural language': 'Natural Language Processing',
            'computer vision': 'Computer Vision',
            'data science': 'Data Science',
            'open source': 'Open Source',
            'ai model': 'AI Models',
            'ai system': 'AI Systems',
            'ai tool': 'AI Tools',
            'ai application': 'AI Applications',
            'ai research': 'AI Research',
            'ai development': 'AI Development',
            'ai ethics': 'AI Ethics',
            'ai safety': 'AI Safety',
            'ai alignment': 'AI Alignment',
            'ai governance': 'AI Governance',
            'ai literacy': 'AI Literacy',
            'prompt engineering': 'Prompt Engineering',
            'fine tuning': 'Fine Tuning',
            'transfer learning': 'Transfer Learning',
            'reinforcement learning': 'Reinforcement Learning',
            'supervised learning': 'Supervised Learning',
            'unsupervised learning': 'Unsupervised Learning',
            'computer science': 'Computer Science',
            'software engineering': 'Software Engineering',
            'data analysis': 'Data Analysis',
            'business intelligence': 'Business Intelligence',
            'user experience': 'User Experience',
            'user interface': 'User Interface',
            'product management': 'Product Management',
            'project management': 'Project Management',
            'agile development': 'Agile Development',
            'software development': 'Software Development',
            'web development': 'Web Development',
            'mobile development': 'Mobile Development',
            'cloud computing': 'Cloud Computing',
            'edge computing': 'Edge Computing',
            'quantum computing': 'Quantum Computing',
            'blockchain technology': 'Blockchain Technology',
            'cyber security': 'Cybersecurity',
            'information security': 'Information Security'
        }
        
        # Enhanced stop words (removed duplicates and improved)
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'not', 'no', 'yes', 'so', 'if', 'then', 'than', 'when', 'where', 'why', 'how',
            'just', 'now', 'here', 'there', 'very', 'too', 'also', 'much', 'many', 'more',
            'like', 'get', 'got', 'go', 'going', 'come', 'came', 'see', 'know', 'think',
            'want', 'need', 'make', 'made', 'take', 'took', 'look', 'say', 'said', 'tell',
            'good', 'bad', 'new', 'old', 'first', 'last', 'next', 'back', 'up', 'down',
            'out', 'over', 'under', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among',
            # Additional noise words
            'really', 'actually', 'probably', 'maybe', 'perhaps', 'seems', 'looks', 'feels',
            'pretty', 'quite', 'rather', 'somewhat', 'little', 'bit', 'kind', 'sort', 'type',
            'thing', 'stuff', 'something', 'anything', 'everything', 'nothing', 'someone',
            'anyone', 'everyone', 'nobody', 'somewhere', 'anywhere', 'everywhere', 'nowhere',
            # Common web/tech noise
            'http', 'https', 'www', 'com', 'org', 'net', 'link', 'url', 'website', 'site',
            'click', 'here', 'there', 'check', 'file', 'image', 'video', 'post', 'message'
        }
        
        # Combine all messages for compound term detection
        all_text = ' '.join(messages).lower()
        
        # Count compound terms first
        topic_freq = Counter()
        
        # Find compound terms
        for term, clean_term in compound_terms.items():
            count = all_text.count(term)
            if count >= 3:  # Minimum threshold for compound terms
                topic_freq[clean_term] = count
        
        # Process individual words
        word_freq = Counter()
        for message in messages:
            content = message.lower()
            # Clean and tokenize
            words = self._clean_and_tokenize(content)
            
            for word in words:
                if self._is_meaningful_topic(word, min_word_length, stop_words):
                    # Clean up the word for display
                    clean_word = self._clean_topic(word)
                    word_freq[clean_word] += 1
        
        # Add meaningful single words that appear frequently enough
        for word, count in word_freq.items():
            if count >= 3 and word not in topic_freq:  # Avoid duplicates with compound terms
                topic_freq[word] = count
        
        # Get top topics sorted by frequency
        top_topics = topic_freq.most_common(top_n)
        
        return [
            {
                "topic": topic,
                "frequency": count,
                "relevance_score": round(count / max(len(messages), 1), 3),
            }
            for topic, count in top_topics
                ]
    
    def _clean_topic(self, topic: str) -> str:
        """Clean and properly capitalize topic for display."""
        # Handle special capitalizations
        special_caps = {
            'ai': 'AI',
            'api': 'API', 
            'ui': 'UI',
            'ux': 'UX',
            'ml': 'ML',
            'nlp': 'NLP',
            'gpt': 'GPT',
            'llm': 'LLM',
            'sql': 'SQL',
            'css': 'CSS',
            'html': 'HTML',
            'json': 'JSON',
            'xml': 'XML',
            'rest': 'REST',
            'oauth': 'OAuth',
            'saas': 'SaaS',
            'paas': 'PaaS',
            'iaas': 'IaaS',
            'devops': 'DevOps',
            'cicd': 'CI/CD'
        }
        
        topic_lower = topic.lower()
        if topic_lower in special_caps:
            return special_caps[topic_lower]
        
        # Capitalize first letter of each word for compound terms
        return ' '.join(word.capitalize() for word in topic.split())
