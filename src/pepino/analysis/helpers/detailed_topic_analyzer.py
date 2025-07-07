"""
Detailed Topic Analyzer

Provides comprehensive topic analysis for the new system, adapted from the legacy analyzer but with unique naming and structure.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import re
from .data_facade import get_analysis_data_facade
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DetailedTopicItem(BaseModel):
    topic: str
    frequency: int
    relevance_score: Optional[float] = None

class DetailedTopicAnalysisResponse(BaseModel):
    topics: List[DetailedTopicItem]
    message_count: int
    capabilities_used: List[str]
    n_topics: int
    days_back: Optional[int]
    channel_name: Optional[str]
    summary: Dict[str, Any]

class DetailedTopicAnalyzer:
    def __init__(self, data_facade=None):
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
        logger.info("DetailedTopicAnalyzer initialized with data facade pattern")

    def analyze(self, channel_name: Optional[str] = None, n_topics: int = 10, days_back: Optional[int] = None) -> Optional[DetailedTopicAnalysisResponse]:
        try:
            logger.info(f"Starting detailed topic analysis for channel: {channel_name}")
            # Get messages
            if channel_name:
                messages = self.data_facade.message_repository.get_channel_messages(
                    channel_name, days_back=days_back, limit=1000
                )
            else:
                messages = self.data_facade.message_repository.get_recent_messages(
                    limit=1000, days_back=days_back
                )
            if not messages:
                logger.warning("No messages found for topic analysis")
                return None
            # Extract text content
            text_content = [msg.content for msg in messages if msg.content]
            if not text_content:
                logger.warning("No text content found for topic analysis")
                return None
            
            # Try advanced analysis first, fallback to simple
            try:
                topics, domain_analysis, capabilities = self._advanced_topic_analysis(text_content, n_topics)
            except Exception as e:
                logger.warning(f"Advanced topic analysis failed, using fallback: {e}")
                topics, domain_analysis, capabilities = self._fallback_topic_analysis(text_content, n_topics)
            
            summary = {
                'top_words': [topic.topic for topic in topics],
                'total_messages': len(messages),
                'domain_analysis': domain_analysis
            }
            
            return DetailedTopicAnalysisResponse(
                topics=topics,
                message_count=len(messages),
                capabilities_used=capabilities,
                n_topics=n_topics,
                days_back=days_back,
                channel_name=channel_name,
                summary=summary
            )
        except Exception as e:
            logger.error(f"Detailed topic analysis failed: {e}")
            return None

    def _advanced_topic_analysis(self, text_content: List[str], n_topics: int) -> Tuple[List[DetailedTopicItem], Dict[str, Any], List[str]]:
        """Advanced topic analysis using BERTopic and spaCy."""
        try:
            # Try to import advanced libraries
            try:
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer
                import spacy
                bertopic_available = True
                spacy_available = True
            except ImportError:
                bertopic_available = False
                spacy_available = False
            
            topics = []
            domain_analysis = {}
            capabilities = []
            
            if bertopic_available:
                # BERTopic analysis
                try:
                    model = SentenceTransformer('all-mpnet-base-v2')
                    embeddings = model.encode(text_content)
                    
                    # Improved BERTopic configuration
                    topic_model = BERTopic(
                        min_topic_size=3,  # Lower threshold for smaller datasets
                        nr_topics=n_topics,
                        verbose=True,
                        # Better topic representation
                        calculate_probabilities=True,
                        # Topic reduction settings
                        top_n_words=10,      # More words per topic
                        n_gram_range=(1, 2), # Include bigrams
                    )
                    
                    bertopic_topics, _ = topic_model.fit_transform(text_content, embeddings)
                    topic_info = topic_model.get_topic_info()
                    
                    # Get detailed topic information
                    topic_words = topic_model.get_topics()
                    
                    for _, row in topic_info.head(n_topics).iterrows():
                        if row['Topic'] != -1:  # Skip noise topic
                            topic_id = row['Topic']
                            
                            # Create meaningful topic name from top words
                            if topic_id in topic_words:
                                words = topic_words[topic_id]
                                # Take top 3-4 most representative words
                                top_words = [word for word, _ in words[:4]]
                                # Create a readable topic name
                                topic_name = " & ".join(top_words[:3]).title()
                            else:
                                topic_name = row['Name']
                            
                            topics.append(DetailedTopicItem(
                                topic=topic_name,
                                frequency=int(row['Count']),
                                relevance_score=float(row['Count']) / len(text_content)
                            ))
                    
                    capabilities.append("bertopic_modeling")
                    
                except Exception as e:
                    logger.warning(f"BERTopic analysis failed: {e}")
            
            if spacy_available:
                # spaCy analysis
                try:
                    nlp = spacy.load("en_core_web_sm")
                    
                    # Process all text
                    all_text = " ".join(text_content)
                    doc = nlp(all_text)
                    
                    # Extract technical terms
                    technical_terms = []
                    for token in doc:
                        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
                            technical_terms.append(token.text.lower())
                    
                    # Extract business concepts
                    business_concepts = []
                    for ent in doc.ents:
                        if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                            business_concepts.append(ent.text)
                    
                    domain_analysis = {
                        'technical_terms': Counter(technical_terms),
                        'business_concepts': Counter(business_concepts),
                        'key_discussions': Counter(),
                        'emerging_topics': []
                    }
                    
                    capabilities.append("spacy_nlp")
                    
                except Exception as e:
                    logger.warning(f"spaCy analysis failed: {e}")
            
            return topics, domain_analysis, capabilities
            
        except Exception as e:
            logger.error(f"Advanced topic analysis failed: {e}")
            raise

    def _fallback_topic_analysis(self, text_content: List[str], n_topics: int) -> Tuple[List[DetailedTopicItem], Dict[str, Any], List[str]]:
        """Fallback topic analysis using simple keyword extraction."""
        try:
            # Combine all text
            all_text = " ".join(text_content).lower()
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            
            # Enhanced stop words list
            stop_words = {
                'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this',
                'have', 'from', 'not', 'but', 'can', 'all', 'any', 'had', 'her', 'one', 'our',
                'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say', 'its',
                'two', 'more', 'very', 'what', 'know', 'just', 'first', 'also', 'after',
                'will', 'would', 'could', 'should', 'been', 'good', 'much', 'some', 'time',
                'very', 'into', 'just', 'only', 'over', 'think', 'also', 'back', 'after',
                'work', 'first', 'well', 'way', 'even', 'want', 'because', 'these', 'give',
                'most', 'us', 'here', 'make', 'look', 'like', 'going', 'see', 'other', 'around',
                'know', 'take', 'than', 'them', 'people', 'come', 'did', 'number', 'sound',
                'no', 'most', 'make', 'over', 'think', 'also', 'back', 'after', 'work',
                'first', 'well', 'way', 'even', 'want', 'because', 'these', 'give', 'most'
            }
            
            # Count word frequencies
            word_freq = Counter(word for word in words if word not in stop_words)
            
            # Group related words into topics
            topic_groups = self._group_related_words(word_freq)
            
            # Create topics from groups
            topics = []
            total_words = sum(word_freq.values())
            
            # Add grouped topics first
            for group_name, group_words in topic_groups.items():
                group_count = sum(word_freq.get(word, 0) for word in group_words)
                if group_count >= 3:  # Minimum threshold for a topic
                    relevance_score = group_count / total_words if total_words > 0 else 0
                    topics.append(DetailedTopicItem(
                        topic=group_name,
                        frequency=group_count,
                        relevance_score=round(relevance_score, 3)
                    ))
            
            # Add individual high-frequency words as topics
            for word, count in word_freq.most_common(n_topics):
                if count >= 5 and not any(word in group for group in topic_groups.values()):
                    relevance_score = count / total_words if total_words > 0 else 0
                    topics.append(DetailedTopicItem(
                        topic=word.title(),
                        frequency=count,
                        relevance_score=round(relevance_score, 3)
                    ))
            
            # Limit to requested number of topics
            topics = sorted(topics, key=lambda x: x.frequency, reverse=True)[:n_topics]
            
            # Simple domain analysis
            domain_analysis = {
                'technical_terms': word_freq,
                'business_concepts': Counter(),
                'key_discussions': Counter(),
                'emerging_topics': []
            }
            
            capabilities = ["keyword_analysis"]
            
            return topics, domain_analysis, capabilities
            
        except Exception as e:
            logger.error(f"Fallback topic analysis failed: {e}")
            return [], {}, ["basic_analysis"]

    def _group_related_words(self, word_freq: Counter) -> Dict[str, List[str]]:
        """Group related words into meaningful topics."""
        groups = {
            "Technology & Development": [
                "code", "programming", "development", "software", "app", "api", "database",
                "server", "client", "frontend", "backend", "framework", "library", "git",
                "deploy", "test", "debug", "bug", "feature", "version", "update", "release"
            ],
            "Business & Work": [
                "business", "company", "work", "job", "career", "project", "team", "meeting",
                "client", "customer", "product", "service", "market", "industry", "startup",
                "funding", "revenue", "profit", "strategy", "planning", "management"
            ],
            "Learning & Education": [
                "learn", "study", "course", "tutorial", "education", "training", "skill",
                "knowledge", "research", "reading", "book", "article", "documentation",
                "practice", "exercise", "assignment", "homework", "exam", "test"
            ],
            "Community & Social": [
                "community", "group", "people", "friend", "social", "network", "discussion",
                "chat", "message", "help", "support", "share", "connect", "meet", "event",
                "conference", "workshop", "meetup", "discord", "slack", "channel"
            ],
            "Time & Schedule": [
                "time", "schedule", "meeting", "appointment", "deadline", "due", "date",
                "today", "tomorrow", "week", "month", "year", "hour", "minute", "morning",
                "afternoon", "evening", "night", "weekend", "holiday", "vacation"
            ]
        }
        
        # Filter groups to only include words that actually appear in the data
        filtered_groups = {}
        for group_name, words in groups.items():
            found_words = [word for word in words if word in word_freq]
            if found_words:
                filtered_groups[group_name] = found_words
        
        return filtered_groups

    def __del__(self):
        if hasattr(self, '_owns_facade') and self._owns_facade and hasattr(self, 'data_facade'):
            try:
                self.data_facade.close()
            except:
                pass 