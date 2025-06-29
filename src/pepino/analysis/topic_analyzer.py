"""
Enhanced Topic Analyzer using BERTopic and spaCy hybrid approach.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pepino.config import Settings
from pepino.logging_config import get_logger

logger = get_logger(__name__)


class TopicAnalyzer:
    """Hybrid topic analyzer using BERTopic + spaCy domain analysis."""

    def __init__(self, data_facade_or_settings=None):
        # Support both data_facade pattern and direct settings
        if data_facade_or_settings is None:
            self.settings = Settings()
            self.data_facade = None
        elif hasattr(data_facade_or_settings, 'message_repository'):
            # It's a data facade
            self.data_facade = data_facade_or_settings
            self.settings = Settings()
        else:
            # It's settings
            self.settings = data_facade_or_settings
            self.data_facade = None
            
        self._topic_model = None
        self._embedding_model = None
        self._nlp_model = None
        self._model_loaded = False
        self._spacy_loaded = False
        
    def initialize(self) -> None:
        """Initialize BERTopic and spaCy models."""
        if self._model_loaded:
            return
            
        try:
            # Import BERTopic and related libraries
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            
            logger.info("Initializing BERTopic topic analyzer...")
            
            # Use a domain-appropriate sentence transformer model
            # all-mpnet-base-v2 is excellent for semantic similarity and technical content
            self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Store model components to be configured later with actual data
            self._umap_model = UMAP
            self._hdbscan_model = HDBSCAN
            self._vectorizer_model = CountVectorizer
            
            self._model_loaded = True
            logger.info("BERTopic components loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing required dependencies for BERTopic: {e}")
            logger.info("Please install with: poetry add bertopic umap-learn hdbscan transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BERTopic: {e}")
            raise

    def _initialize_spacy(self) -> None:
        """Initialize spaCy model for advanced NLP analysis."""
        if self._spacy_loaded:
            return
            
        try:
            import spacy
            
            logger.info("Loading spaCy model for domain analysis...")
            try:
                self._nlp_model = spacy.load("en_core_web_sm")
                self._spacy_loaded = True
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self._spacy_loaded = False
        except ImportError:
            logger.warning("spaCy not available. Install with: pip install spacy")
            self._spacy_loaded = False

    def _get_tech_stop_words(self) -> List[str]:
        """Get comprehensive stop words including tech/discord noise."""
        # Start with English stop words and add tech/discord specific ones
        stop_words = [
            # English stop words
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            
            # Discord/community noise that we previously had trouble with
            'group', 'time', 'from', 'what', 'buddy', 'meeting', 'today', 'week', 'thanks',
            'please', 'welcome', 'session', 'team', 'join', 'channel', 'question', 'community',
            'people', 'person', 'user', 'member', 'everyone', 'someone',
            
            # Generic business noise
            'work', 'working', 'project', 'process', 'approach', 'way', 'ways', 'thing', 'things',
            'really', 'think', 'know', 'want', 'need', 'like', 'get', 'go', 'see', 'look',
            
            # Time/date
            'day', 'days', 'week', 'month', 'year', 'time', 'today', 'yesterday', 'tomorrow'
        ]
        return stop_words

    def _clean_content_discord(self, text: str) -> str:
        """Clean Discord-specific patterns (adapted from core.py)."""
        # Remove Discord-specific patterns
        text = re.sub(r"<@[!&]?\d+>", "", text)  # Remove mentions
        text = re.sub(r"<#\d+>", "", text)  # Remove channel references
        text = re.sub(r"<:\w+:\d+>", "", text)  # Remove custom emojis
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r"`[^`]+`", "", text)  # Remove inline code
        text = re.sub(r"[🎯🏷️💡🔗🔑📝🌎🏭]", "", text)  # Remove emoji noise
        text = re.sub(
            r"\b(time zone|buddy group|display name|main goal|learning topics)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(the server|the session|the recording|the future)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(messages?|channel|group|topic|session|meeting)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_domain_patterns(self, messages_with_timestamps: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Extract domain-specific patterns using spaCy (adapted from core.py)."""
        if not self._spacy_loaded:
            return {}
            
        technical_terms = Counter()
        business_concepts = Counter()
        key_discussions = Counter()
        temporal_trends = defaultdict(list)
        
        # Define focused patterns
        tech_patterns = [
            r"\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework|metrics)\b"
        ]
        
        business_patterns = [
            r"\b(team|collaboration|workflow|process|efficiency|optimization|integration|strategy|growth|solution|deployment)\b"
        ]
        
        for content, timestamp in messages_with_timestamps:
            cleaned_content = self._clean_content_discord(content)
            
            if len(cleaned_content.split()) < 5:
                continue
                
            try:
                doc = self._nlp_model(cleaned_content)
                
                # Extract technical terms
                for pattern in tech_patterns:
                    matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                    for match in matches:
                        technical_terms[match.upper()] += 1
                
                # Extract business concepts
                for pattern in business_patterns:
                    matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                    for match in matches:
                        business_concepts[match.lower()] += 1
                
                # Extract multi-word discussion themes using spaCy noun chunks
                for chunk in doc.noun_chunks:
                    if (len(chunk.text.split()) >= 3 and len(chunk.text) > 15 
                        and chunk.text.lower() not in ["the conversational leaders", "the community coordinators"]):
                        clean_chunk = re.sub(r"^(the|a|an)\s+", "", chunk.text.lower()).strip()
                        if len(clean_chunk.split()) >= 2:
                            key_discussions[clean_chunk.title()] += 1
                
                # Time-based trends
                try:
                    if timestamp:
                        msg_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
                        week_key = msg_date.strftime("%Y-W%U")
                        
                        # Add significant concepts to trends
                        main_concepts = [
                            token.lemma_.lower() for token in doc
                            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 3
                        ]
                        temporal_trends[week_key].extend(main_concepts[:3])
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"Error processing message with spaCy: {e}")
                continue
        
        # Calculate emerging topics
        emerging_topics = []
        if len(temporal_trends) >= 3:
            all_weeks = sorted(temporal_trends.keys())
            recent_weeks = all_weeks[-2:]
            older_weeks = all_weeks[:-2]
            
            recent_concepts = Counter()
            older_concepts = Counter()
            
            for week in recent_weeks:
                recent_concepts.update(temporal_trends[week])
            for week in older_weeks:
                older_concepts.update(temporal_trends[week])
            
            for concept, recent_count in recent_concepts.most_common(10):
                older_count = older_concepts.get(concept, 0)
                if recent_count >= 3:
                    ratio = recent_count / max(older_count, 1)
                    if ratio > 2.0:
                        emerging_topics.append((concept, recent_count, ratio))
        
        return {
            'technical_terms': technical_terms,
            'business_concepts': business_concepts,
            'key_discussions': key_discussions,
            'emerging_topics': emerging_topics,
            'temporal_trends': temporal_trends
        }

    def _extract_complex_concepts_spacy(self, doc) -> List[str]:
        """Extract complex concepts using spaCy dependency parsing (from core.py)."""
        concepts = []
        
        # Extract compound subjects with their predicates
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
                subject_span = doc[token.left_edge.i:token.right_edge.i + 1]
                if token.head.pos_ == "VERB":
                    verb = token.head
                    predicate_parts = [verb.text]
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj", "attr", "prep"]:
                            obj_span = doc[child.left_edge.i:child.right_edge.i + 1]
                            predicate_parts.append(obj_span.text)
                    if len(predicate_parts) > 1:
                        full_concept = f"{subject_span.text} {' '.join(predicate_parts)}"
                        if len(full_concept.split()) >= 3 and len(full_concept) > 15:
                            concepts.append(full_concept.lower().strip())
        
        # Extract extended noun phrases
        for chunk in doc.noun_chunks:
            extended_phrase = chunk.text
            for token in chunk:
                for child in token.children:
                    if child.dep_ == "prep":
                        prep_phrase = doc[child.i:child.right_edge.i + 1]
                        extended_phrase += f" {prep_phrase.text}"
            if len(extended_phrase.split()) >= 3 and len(extended_phrase) > 20:
                concepts.append(extended_phrase.lower().strip())
        
        # Extract technical compounds
        for i, token in enumerate(doc[:-2]):
            if (token.pos_ in ["NOUN", "PROPN"] and doc[i + 1].pos_ in ["NOUN", "PROPN", "ADJ"] 
                and doc[i + 2].pos_ in ["NOUN", "PROPN"]):
                compound = f"{token.text} {doc[i+1].text} {doc[i+2].text}"
                j = i + 3
                while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and j < i + 6:
                    compound += f" {doc[j].text}"
                    j += 1
                if len(compound.split()) >= 3:
                    concepts.append(compound.lower())
        
        return concepts

    def extract_topics(
        self, 
        messages: List[str], 
        messages_with_timestamps: Optional[List[Tuple[str, str]]] = None,
        min_topic_size: int = 2,
        nr_topics: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract topics using hybrid BERTopic + spaCy approach.
        
        Args:
            messages: List of message texts
            messages_with_timestamps: List of (message, timestamp) tuples for temporal analysis
            min_topic_size: Minimum cluster size for topics
            nr_topics: Number of topics to reduce to (None for auto)
            
        Returns:
            Tuple of (bertopic_results, domain_analysis)
        """
        if not self._model_loaded:
            self.initialize()
        
        # Initialize spaCy for domain analysis
        self._initialize_spacy()
            
        if not messages or len(messages) < 2:
            return [], {}
        
        # Run domain analysis if we have spaCy and timestamps
        domain_analysis = {}
        if self._spacy_loaded and messages_with_timestamps:
            logger.info("Running spaCy-based domain analysis...")
            domain_analysis = self._extract_domain_patterns(messages_with_timestamps)
        
        # Run BERTopic analysis
        bertopic_results = []
        try:
            # Clean and filter messages
            cleaned_messages = self._clean_messages(messages)
            if len(cleaned_messages) < 2:
                return [], domain_analysis
            
            logger.info(f"Extracting topics from {len(cleaned_messages)} messages using BERTopic")
            
            # Configure models based on actual data size
            n_messages = len(cleaned_messages)
            
            # Configure UMAP for dimensionality reduction (adaptive to data size)
            umap_model = self._umap_model(
                n_neighbors=min(15, max(2, n_messages // 3)),
                n_components=min(5, max(2, n_messages // 5)), 
                min_dist=0.0, 
                metric='cosine',
                random_state=42
            )
            
            # Configure HDBSCAN for clustering (adaptive cluster size)
            hdbscan_model = self._hdbscan_model(
                min_cluster_size=max(2, min(n_messages // 8, 5)),
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Configure vectorizer with AI/tech-aware stop words
            tech_stop_words = self._get_tech_stop_words()
            vectorizer_model = self._vectorizer_model(
                stop_words=tech_stop_words,
                ngram_range=(1, 3),  # Include 3-word technical phrases
                min_df=max(1, min(2, n_messages // 10)),  # Adaptive min_df
                max_features=min(1000, max(50, n_messages * 5))  # Adaptive max_features
            )
            
            # Initialize BERTopic with configured components
            from bertopic import BERTopic
            topic_model = BERTopic(
                embedding_model=self._embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=None,  # Use default c-TF-IDF
                verbose=True,
                calculate_probabilities=False  # Faster processing
            )
            
            # Fit BERTopic model
            topics, probabilities = topic_model.fit_transform(cleaned_messages)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            
            # Convert to our format
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                # Skip outlier topic (-1)
                if topic_id == -1:
                    continue
                    
                # Get topic words and scores
                topic_words = topic_model.get_topic(topic_id)
                if not topic_words:
                    continue
                
                # Calculate topic strength and coherence
                count = row['Count']
                total_docs = len(cleaned_messages)
                relevance_score = count / total_docs
                
                # Get representative documents
                representative_docs = self._get_representative_docs(
                    topic_id, topics, cleaned_messages, max_docs=3
                )
                
                # Create topic summary
                topic_name = self._generate_topic_name(topic_words, representative_docs)
                
                # Filter out empty keywords and scores
                filtered_keywords = [(word, score) for word, score in topic_words[:10] if word.strip()]
                
                topic_result = {
                    "topic": topic_name,
                    "topic_id": int(topic_id),
                    "frequency": int(count),
                    "relevance_score": round(relevance_score, 3),
                    "keywords": [word for word, score in filtered_keywords],
                    "keyword_scores": [round(score, 3) for word, score in filtered_keywords],
                    "representative_docs": representative_docs,
                    "coherence_score": self._calculate_topic_coherence(filtered_keywords)
                }
                
                bertopic_results.append(topic_result)
            
            # Sort by frequency (most common first)
            bertopic_results.sort(key=lambda x: x['frequency'], reverse=True)
            
            logger.info(f"Successfully extracted {len(bertopic_results)} BERTopic topics")
            
        except Exception as e:
            logger.error(f"Error extracting topics with BERTopic: {e}")
            # Fallback to simple keyword extraction for BERTopic part
            bertopic_results = self._fallback_topic_extraction(messages)
        
        return bertopic_results, domain_analysis

    def _clean_messages(self, messages: List[str]) -> List[str]:
        """Clean and filter messages for topic modeling."""
        cleaned = []
        
        for message in messages:
            if not message or not message.strip():
                continue
                
            # Basic cleaning
            text = message.strip()
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # Remove Discord mentions and channels
            text = re.sub(r'<@[!&]?\d+>', '', text)
            text = re.sub(r'<#\d+>', '', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Filter out very short messages
            if len(text.split()) >= 3:  # At least 3 words
                cleaned.append(text)
        
        return cleaned

    def _get_representative_docs(
        self, 
        topic_id: int, 
        topics: List[int], 
        documents: List[str], 
        max_docs: int = 3
    ) -> List[str]:
        """Get representative documents for a topic."""
        topic_docs = [doc for i, doc in enumerate(documents) if topics[i] == topic_id]
        
        # Return first few documents (could be enhanced with similarity scoring)
        return topic_docs[:max_docs]

    def _generate_topic_name(self, topic_words: List[Tuple], representative_docs: List[str]) -> str:
        """Generate a human-readable topic name."""
        if not topic_words:
            return "Unknown Topic"
        
        # Use top 2-3 words to create topic name
        top_words = [word for word, score in topic_words[:3]]
        
        # Capitalize and join
        topic_name = " + ".join(word.title() for word in top_words)
        
        return topic_name

    def _calculate_topic_coherence(self, topic_words: List[Tuple]) -> float:
        """Calculate a simple coherence score for a topic."""
        if not topic_words:
            return 0.0
        
        # Simple coherence based on word scores
        scores = [score for word, score in topic_words]
        if not scores:
            return 0.0
            
        # Average of top word scores as coherence proxy
        return round(sum(scores[:5]) / min(len(scores), 5), 3)

    def _fallback_topic_extraction(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Fallback to simple keyword-based topic extraction if BERTopic fails."""
        logger.warning("Using fallback topic extraction method")
        
        # Simple keyword frequency analysis
        word_freq = Counter()
        
        for message in messages:
            # Basic cleaning and tokenization
            words = re.findall(r'\b[a-zA-Z]{4,}\b', message.lower())
            for word in words:
                if word not in self._get_tech_stop_words():
                    word_freq[word] += 1
        
        # Create topics from most frequent words
        topics = []
        for word, count in word_freq.most_common(10):
            if count >= 2:  # Must appear at least twice
                topics.append({
                    "topic": word.title(),
                    "topic_id": len(topics),
                    "frequency": count,
                    "relevance_score": round(count / len(messages), 3),
                    "keywords": [word],
                    "keyword_scores": [1.0],
                    "representative_docs": [],
                    "coherence_score": 0.5
                })
        
        return topics

    def analyze(
        self,
        channel_name: Optional[str] = None,
        days_back: int = 30,
        top_n: int = 20,
        min_word_length: int = 4
    ) -> "TopicAnalysisResponse":
        """
        Main analysis method for integration with the CLI and Discord commands.
        Uses hybrid BERTopic + spaCy approach for comprehensive topic analysis.
        
        Args:
            channel_name: Optional channel to analyze
            days_back: Days to look back for messages
            top_n: Number of top topics to return
            min_word_length: Minimum word length for analysis
            
        Returns:
            TopicAnalysisResponse with BERTopic-enhanced results and domain analysis
        """
        try:
            from .models import TopicAnalysisResponse, TopicItem, AnalysisErrorResponse
            
            # Get messages using data_facade (if available) or direct access
            messages = []
            messages_with_timestamps = []
            
            if hasattr(self, 'data_facade') and self.data_facade:
                # Using data facade pattern from CLI/Discord integration
                try:
                    if channel_name:
                        message_data = self.data_facade.message_repository.get_channel_messages(
                            channel_name, days_back=days_back, limit=1000
                        )
                    else:
                        message_data = self.data_facade.message_repository.get_recent_messages(
                            limit=1000, days_back=days_back
                        )
                    
                    # Extract both content and timestamps for hybrid analysis
                    for msg in message_data:
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        if content:
                            messages.append(content)
                            messages_with_timestamps.append((content, timestamp))
                    
                except Exception as e:
                    logger.error(f"Error fetching messages via data facade: {e}")
                    return AnalysisErrorResponse(
                        error=f"Failed to fetch messages: {e}",
                        plugin="TopicAnalyzer"
                    )
            else:
                # Fallback for direct usage without data facade
                logger.warning("TopicAnalyzer used without data facade - limited functionality")
                messages = []
                messages_with_timestamps = []
            
            if not messages:
                return AnalysisErrorResponse(
                    error="No messages found for topic analysis",
                    plugin="TopicAnalyzer"
                )
            
            logger.info(f"Analyzing {len(messages)} messages with hybrid BERTopic + spaCy approach")
            
            # Use the new hybrid extraction
            bertopic_results, domain_analysis = self.extract_topics(
                messages, 
                messages_with_timestamps=messages_with_timestamps, 
                min_topic_size=2
            )
            
            # Convert BERTopic results to TopicItem models
            topic_items = []
            for topic_data in bertopic_results[:top_n]:
                topic_item = TopicItem(
                    topic=topic_data["topic"],
                    frequency=topic_data["frequency"],
                    relevance_score=topic_data["relevance_score"]
                )
                topic_items.append(topic_item)
            
            # Determine capabilities used
            capabilities = ["bertopic_modeling", "sentence_transformers", "semantic_clustering"]
            if domain_analysis:
                capabilities.extend(["spacy_nlp", "domain_patterns", "temporal_trends"])
            
            # Create enhanced response with domain analysis
            response = TopicAnalysisResponse(
                success=True,
                plugin="TopicAnalyzer",
                topics=topic_items,
                message_count=len(messages),
                capabilities_used=capabilities
            )
            
            # Add domain analysis data to response (if available)
            if domain_analysis:
                # Store domain analysis for template access
                response._domain_analysis = domain_analysis
                
                # Add some domain metrics to the main response
                if 'technical_terms' in domain_analysis:
                    response._technical_terms_count = len(domain_analysis['technical_terms'])
                if 'emerging_topics' in domain_analysis:
                    response._emerging_topics_count = len(domain_analysis['emerging_topics'])
            
            return response
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return AnalysisErrorResponse(
                error=f"Analysis failed: {e}",
                plugin="TopicAnalyzer"
            )

    # Legacy method for backward compatibility
    def _extract_top_topics(
        self, messages: List[str], top_n: int = 20, min_word_length: int = 4
    ) -> List[Dict[str, Any]]:
        """Legacy method - delegates to new extract_topics method."""
        bertopic_results, _ = self.extract_topics(messages)
        return bertopic_results[:top_n]
