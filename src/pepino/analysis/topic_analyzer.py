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
            'me', 'my', 'mine', 'your', 'yours', 'his', 'hers', 'its', 'our', 'ours', 'their', 'theirs',
            'who', 'whom', 'whose', 'which', 'what', 'when', 'where', 'why', 'how',
            
            # Generic conversational words that create noise
            'thank', 'thanks', 'thanking', 'please', 'sorry', 'welcome', 'hello', 'hey', 'hi',
            'yes', 'yeah', 'yep', 'no', 'nope', 'okay', 'ok', 'sure', 'right', 'exactly',
            'very', 'really', 'quite', 'pretty', 'much', 'many', 'most', 'more', 'less',
            'good', 'great', 'nice', 'cool', 'awesome', 'amazing', 'excellent', 'perfect',
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'best', 'better', 'worse',
            
            # Action words that don't represent topics
            'sharing', 'share', 'shared', 'shares', 'tell', 'telling', 'told', 'ask', 'asking',
            'show', 'showing', 'showed', 'give', 'giving', 'gave', 'take', 'taking', 'took',
            'make', 'making', 'made', 'create', 'creating', 'created', 'build', 'building', 'built',
            'use', 'using', 'used', 'try', 'trying', 'tried', 'test', 'testing', 'tested',
            
            # Discord/community noise that we previously had trouble with
            'group', 'time', 'from', 'what', 'buddy', 'meeting', 'today', 'week', 'weeks',
            'session', 'sessions', 'team', 'teams', 'join', 'joined', 'joining', 'channel', 'channels',
            'question', 'questions', 'community', 'communities', 'server', 'servers',
            'people', 'person', 'user', 'users', 'member', 'members', 'everyone', 'someone',
            'anybody', 'nobody', 'somebody', 'everything', 'nothing', 'something', 'anything',
            
            # Generic business/work noise
            'work', 'working', 'worked', 'works', 'job', 'jobs', 'task', 'tasks',
            'project', 'projects', 'process', 'processes', 'approach', 'approaches',
            'way', 'ways', 'method', 'methods', 'thing', 'things', 'stuff', 'item', 'items',
            'idea', 'ideas', 'thought', 'thoughts', 'concept', 'concepts',
            'think', 'thinking', 'thought', 'know', 'knowing', 'knew', 'learn', 'learning', 'learned',
            'want', 'wanting', 'wanted', 'need', 'needing', 'needed', 'like', 'liking', 'liked',
            'get', 'getting', 'got', 'go', 'going', 'went', 'come', 'coming', 'came',
            'see', 'seeing', 'saw', 'look', 'looking', 'looked', 'find', 'finding', 'found',
            
            # Time/date/temporal
            'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
            'time', 'times', 'today', 'yesterday', 'tomorrow', 'now', 'then', 'later',
            'before', 'after', 'during', 'while', 'since', 'until', 'when', 'whenever',
            
            # Vague quantifiers and modifiers
            'some', 'any', 'all', 'every', 'each', 'other', 'another', 'different', 'same',
            'new', 'old', 'first', 'last', 'next', 'previous', 'current', 'recent',
            'big', 'small', 'large', 'little', 'huge', 'tiny', 'long', 'short',
            'high', 'low', 'top', 'bottom', 'left', 'right', 'up', 'down',
            
            # Common discourse markers
            'also', 'too', 'either', 'neither', 'both', 'however', 'therefore', 'thus',
            'hence', 'moreover', 'furthermore', 'additionally', 'meanwhile', 'otherwise',
            'instead', 'rather', 'actually', 'basically', 'generally', 'specifically',
            'particularly', 'especially', 'mainly', 'mostly', 'usually', 'typically',
            
            # Generic tech noise (keep domain-specific terms)
            'app', 'apps', 'tool', 'tools', 'system', 'systems', 'platform', 'platforms',
            'service', 'services', 'product', 'products', 'feature', 'features',
            'version', 'versions', 'update', 'updates', 'change', 'changes',
            'issue', 'issues', 'problem', 'problems', 'solution', 'solutions',
            'option', 'options', 'setting', 'settings', 'config', 'configuration',
            
            # Communication noise
            'message', 'messages', 'chat', 'chats', 'talk', 'talking', 'talked',
            'discuss', 'discussing', 'discussed', 'discussion', 'discussions',
            'mention', 'mentioned', 'mentioning', 'call', 'called', 'calling',
            'email', 'emails', 'text', 'texts', 'post', 'posts', 'comment', 'comments',
            
            # Location/direction noise
            'here', 'there', 'where', 'everywhere', 'somewhere', 'nowhere', 'anywhere',
            'above', 'below', 'inside', 'outside', 'around', 'through', 'across',
            'over', 'under', 'between', 'among', 'within', 'without', 'beyond',
            
            # Single letters and numbers that might slip through
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            
            # ADDITIONAL STOP WORDS based on the problem examples
            # Conversational noise from the examples
            'tip', 'tips', 'prompt', 'prompts', 'step', 'steps', 'love', 'lot', 'lots',
            'personal', 'morning', 'having', 'already', 'via', 'attend', 'attending',
            'gen', 'global', 'accessibility', 'takeaway', 'takeaways', 'empowers', 'empower',
            'discord', 'dm', 'dms', 'chat', 'chats', 'end', 'ends', 'care', 'caring',
            'development', 'dev', 'code', 'coding', 'healthcare', 'health',
            
            # Common names that appear in the data (causing noise)
            'arturo', 'don', 'abel', 'jose', 'maria', 'john', 'jane', 'mike', 'sarah',
            'david', 'lisa', 'tom', 'anna', 'paul', 'emma', 'alex', 'chris', 'sam',
            'minutes', 'minute', 'mins', 'min', 'just', 'well', 'her', 'him', 'his',
            'hers', 'theirs', 'mine', 'yours', 'ours', 'dont', 'cant', 'wont', 'isnt',
            'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt',
            'wouldnt', 'couldnt', 'shouldnt', 'not', 'nor', 'neither', 'either',
            
            # Additional noise words found in recent results
            'forward', 'trip', 'family', 'daily', 'than', 'them', 'then', 'there',
            'back', 'away', 'around', 'along', 'across', 'through', 'under', 'over',
            'into', 'onto', 'upon', 'within', 'without', 'behind', 'beside', 'below',
            'above', 'inside', 'outside', 'between', 'among', 'against', 'toward',
            'towards', 'until', 'since', 'during', 'before', 'after', 'while',
            'due', 'news', 'dear', 'amp', 'email', 'emails', 'mail', 'letter', 'letters',
            'note', 'notes', 'memo', 'memos', 'document', 'documents', 'file', 'files',
            
            # Additional conversational patterns
            'feel', 'feeling', 'feels', 'felt', 'hope', 'hoping', 'hoped', 'wish', 'wishing',
            'start', 'starting', 'started', 'begin', 'beginning', 'began', 'finish', 'finishing',
            'help', 'helping', 'helped', 'support', 'supporting', 'supported',
            'understand', 'understanding', 'understood', 'explain', 'explaining', 'explained',
            'remember', 'remembering', 'remembered', 'forget', 'forgetting', 'forgot',
            'happen', 'happening', 'happened', 'occur', 'occurring', 'occurred',
            'seem', 'seeming', 'seemed', 'appear', 'appearing', 'appeared',
            'become', 'becoming', 'became', 'turn', 'turning', 'turned',
            'keep', 'keeping', 'kept', 'stay', 'staying', 'stayed',
            'move', 'moving', 'moved', 'change', 'changing', 'changed',
            'continue', 'continuing', 'continued', 'stop', 'stopping', 'stopped',
            
            # Emotional/subjective terms
            'love', 'loving', 'loved', 'hate', 'hating', 'hated', 'enjoy', 'enjoying', 'enjoyed',
            'excited', 'exciting', 'boring', 'bored', 'interesting', 'interested',
            'surprised', 'surprising', 'confused', 'confusing', 'clear', 'unclear',
            'easy', 'easier', 'easiest', 'hard', 'harder', 'hardest', 'difficult', 'simple',
            'important', 'unimportant', 'useful', 'useless', 'helpful', 'unhelpful',
            
            # Temporal discourse markers
            'recently', 'lately', 'soon', 'early', 'late', 'quick', 'quickly', 'slow', 'slowly',
            'fast', 'faster', 'fastest', 'immediate', 'immediately', 'eventual', 'eventually',
            'sudden', 'suddenly', 'gradual', 'gradually', 'frequent', 'frequently',
            'occasional', 'occasionally', 'rare', 'rarely', 'never', 'always', 'sometimes',
            
            # Generic descriptive terms
            'kind', 'type', 'sort', 'form', 'part', 'piece', 'bit', 'section', 'area',
            'place', 'location', 'position', 'spot', 'point', 'level', 'stage', 'phase',
            'side', 'end', 'beginning', 'middle', 'center', 'edge', 'corner', 'line',
            'space', 'room', 'field', 'domain', 'scope', 'range', 'scale', 'size',
            
            # Generic organizational terms
            'list', 'lists', 'group', 'groups', 'set', 'sets', 'collection', 'collections',
            'series', 'sequence', 'order', 'orders', 'arrangement', 'arrangements',
            'organization', 'structure', 'structures', 'format', 'formats', 'style', 'styles',
            
            # Generic relational terms
            'between', 'among', 'within', 'across', 'through', 'throughout', 'around',
            'about', 'concerning', 'regarding', 'related', 'relating', 'connection',
            'relationship', 'relationships', 'association', 'associations', 'link', 'links',
            
            # Generic activity terms
            'activity', 'activities', 'action', 'actions', 'operation', 'operations',
            'function', 'functions', 'procedure', 'procedures', 'routine', 'routines',
            'practice', 'practices', 'exercise', 'exercises', 'event', 'events',
            
            # Generic outcome terms
            'result', 'results', 'outcome', 'outcomes', 'effect', 'effects', 'impact', 'impacts',
            'consequence', 'consequences', 'benefit', 'benefits', 'advantage', 'advantages',
            'disadvantage', 'disadvantages', 'gain', 'gains', 'loss', 'losses',
            
            # Generic evaluation terms
            'quality', 'qualities', 'value', 'values', 'worth', 'merit', 'merits',
            'strength', 'strengths', 'weakness', 'weaknesses', 'pro', 'pros', 'con', 'cons',
            'positive', 'negative', 'neutral', 'success', 'successful', 'failure', 'failed'
        ]
        return stop_words

    def _clean_content_discord(self, text: str) -> str:
        """Clean Discord-specific patterns and conversational noise aggressively."""
        # Remove Discord-specific patterns
        text = re.sub(r"<@[!&]?\d+>", "", text)  # Remove mentions
        text = re.sub(r"<#\d+>", "", text)  # Remove channel references
        text = re.sub(r"<:\w+:\d+>", "", text)  # Remove custom emojis
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r"`[^`]+`", "", text)  # Remove inline code
        text = re.sub(r"[🎯🏷️💡🔗🔑📝🌎🏭]", "", text)  # Remove emoji noise
        
        # Remove common Discord/community phrases
        text = re.sub(
            r"\b(time zone|buddy group|display name|main goal|learning topics)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(the server|the session|the recording|the future|the channel|the group)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(messages?|channel|group|topic|session|meeting|discussion|conversation)\b",
            "", text, flags=re.IGNORECASE
        )
        
        # Remove conversational noise patterns
        text = re.sub(
            r"\b(thanks?|thank you|please|sorry|welcome|hello|hey|hi|yes|yeah|no|okay|ok|sure|right|exactly)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(very|really|quite|pretty|much|good|great|nice|cool|awesome|amazing|excellent|perfect)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(sharing|share|tell|ask|show|give|take|make|create|build|use|try|test)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(tip|tips|prompt|prompts|step|steps|love|lot|lots|personal|morning|having|already|via|attend)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(gen|global|accessibility|takeaway|takeaways|empowers|empower|discord|dm|dms|chat|chats|end|ends)\b",
            "", text, flags=re.IGNORECASE
        )
        
        # Remove temporal and vague terms
        text = re.sub(
            r"\b(today|yesterday|tomorrow|now|then|later|before|after|during|while|since|until|when|time|times)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(some|any|all|every|each|other|another|different|same|new|old|first|last|next|previous|current|recent)\b",
            "", text, flags=re.IGNORECASE
        )
        
        # Remove action and state verbs
        text = re.sub(
            r"\b(think|know|want|need|like|get|go|come|see|look|find|feel|hope|wish|start|begin|finish|help|support)\b",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(understand|explain|remember|forget|happen|occur|seem|appear|become|turn|keep|stay|move|change|continue|stop)\b",
            "", text, flags=re.IGNORECASE
        )
        
        # Clean up whitespace and return
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
        
        # Assess message quality for topic modeling
        quality_assessment = self._assess_message_quality(messages)
        logger.info(f"Message quality assessment: {quality_assessment}")
        
        # Decide on analysis approach based on message quality
        if quality_assessment['suitable_for_bertopic']:
            logger.info("Messages suitable for BERTopic analysis")
            bertopic_results = self._extract_topics_with_bertopic(messages, min_topic_size, nr_topics)
        else:
            logger.info("Messages not suitable for BERTopic - using Discord-optimized approach")
            bertopic_results = self._extract_topics_discord_optimized(messages, quality_assessment)
        
        return bertopic_results, domain_analysis

    def _assess_message_quality(self, messages: List[str]) -> Dict[str, Any]:
        """Assess if messages are suitable for advanced topic modeling."""
        if not messages:
            return {'suitable_for_bertopic': False, 'reason': 'no_messages'}
        
        # Analyze message characteristics
        total_messages = len(messages)
        short_messages = 0  # < 30 chars (increased threshold)
        medium_messages = 0  # 30-150 chars
        long_messages = 0   # > 150 chars
        mention_heavy = 0   # > 20% mentions/emojis (stricter)
        substantive_messages = 0  # Messages with meaningful content
        conversational_messages = 0  # Messages that are clearly conversational
        
        # Conversational patterns that indicate casual Discord chat
        conversational_patterns = [
            r'\b(thanks?|thank you|lol|haha|😂|😊|👍|❤️)\b',
            r'\b(hey|hi|hello|bye|see you|good luck|good job)\b',
            r'\b(awesome|amazing|great|nice|cool|wow|omg)\b',
            r'\b(i think|i feel|i hope|i wish|i love|i hate)\b',
            r'\b(you know|you see|you think|you feel)\b',
            r'@everyone|@here',
            r'<@\d+>',  # User mentions
        ]
        
        for msg in messages:
            msg_len = len(msg.strip())
            
            if msg_len < 30:  # Increased from 20
                short_messages += 1
            elif msg_len < 150:  # Increased from 100
                medium_messages += 1
            else:
                long_messages += 1
            
            # Check for mention/emoji heavy content (stricter threshold)
            mention_count = msg.count('@') + msg.count('<:') + len([c for c in msg if ord(c) > 127])
            if mention_count / max(len(msg), 1) > 0.2:  # Reduced from 0.3
                mention_heavy += 1
            
            # Check for conversational patterns
            is_conversational = False
            for pattern in conversational_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    is_conversational = True
                    break
            
            if is_conversational:
                conversational_messages += 1
            
            # Check for substantive content (after cleaning) - stricter criteria
            cleaned = self._clean_content_discord(msg)
            words = cleaned.split()
            meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
            domain_words = [w for w in meaningful_words if self._is_domain_specific_term(w)]
            
            # Require either 5+ meaningful words OR 2+ domain-specific words
            if len(meaningful_words) >= 5 or len(domain_words) >= 2:
                substantive_messages += 1
        
        # Calculate percentages
        short_pct = short_messages / total_messages
        long_pct = long_messages / total_messages
        mention_heavy_pct = mention_heavy / total_messages
        substantive_pct = substantive_messages / total_messages
        conversational_pct = conversational_messages / total_messages
        
        # Much stricter criteria for BERTopic suitability
        suitable_for_bertopic = (
            total_messages >= 20 and  # Increased minimum
            substantive_pct >= 0.5 and  # Increased from 0.3
            short_pct < 0.5 and  # Reduced from 0.7 (less tolerance for short messages)
            mention_heavy_pct < 0.4 and  # Reduced from 0.6
            conversational_pct < 0.6 and  # New criterion: less than 60% conversational
            long_pct >= 0.2  # At least 20% long messages for depth
        )
        
        return {
            'suitable_for_bertopic': suitable_for_bertopic,
            'total_messages': total_messages,
            'short_messages_pct': round(short_pct, 2),
            'long_messages_pct': round(long_pct, 2),
            'mention_heavy_pct': round(mention_heavy_pct, 2),
            'substantive_pct': round(substantive_pct, 2),
            'conversational_pct': round(conversational_pct, 2),
            'reason': 'quality_check_complete'
        }

    def _extract_topics_with_bertopic(
        self, 
        messages: List[str], 
        min_topic_size: int, 
        nr_topics: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Extract topics using BERTopic for high-quality messages."""
        try:
            # Clean and filter messages
            cleaned_messages = self._clean_messages(messages)
            if len(cleaned_messages) < 2:
                return []
            
            logger.info(f"Extracting topics from {len(cleaned_messages)} messages using BERTopic")
            
            # Configure models based on actual data size
            n_messages = len(cleaned_messages)
            
            # Configure UMAP for dimensionality reduction (more conservative for better clustering)
            umap_model = self._umap_model(
                n_neighbors=min(20, max(5, n_messages // 4)),  # Larger neighborhoods for stability
                n_components=min(10, max(5, n_messages // 10)),  # More dimensions for complex topics
                min_dist=0.1,  # Allow some overlap for related topics
                metric='cosine',
                random_state=42
            )
            
            # Configure HDBSCAN for clustering (optimized for coherent topics)
            hdbscan_model = self._hdbscan_model(
                min_cluster_size=max(5, min(n_messages // 6, 15)),  # Larger clusters for coherence
                min_samples=max(3, min(n_messages // 10, 8)),  # More samples for stability
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Configure vectorizer with AI/tech-aware stop words (simpler, safer config)
            tech_stop_words = self._get_tech_stop_words()
            
            vectorizer_model = self._vectorizer_model(
                stop_words=tech_stop_words,
                ngram_range=(1, 2),  # Limit to bigrams for stability
                min_df=2,  # Simple minimum frequency
                max_df=0.95,  # Very conservative max_df
                max_features=min(200, max(50, n_messages)),  # Conservative feature count
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b'  # At least 3 chars, allow underscores/hyphens
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
            bertopic_results = []
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
            
            # Post-process topics for quality and coherence
            bertopic_results = self._post_process_topics(bertopic_results)
            
            # Sort by frequency (most common first)
            bertopic_results.sort(key=lambda x: x['frequency'], reverse=True)
            
            logger.info(f"Successfully extracted {len(bertopic_results)} high-quality BERTopic topics")
            return bertopic_results
            
        except Exception as e:
            logger.error(f"Error extracting topics with BERTopic: {e}")
            # Fallback to Discord-optimized approach
            return self._extract_topics_discord_optimized(messages, {})

    def _extract_topics_discord_optimized(
        self, 
        messages: List[str], 
        quality_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract topics optimized for Discord conversations (short, casual messages)."""
        logger.info("Using Discord-optimized topic extraction")
        
        # For Discord conversations, focus on:
        # 1. Recurring themes and subjects
        # 2. Tools, technologies, and platforms mentioned
        # 3. Activities and actions discussed
        # 4. Questions and problems being solved
        
        from collections import Counter, defaultdict
        import re
        
        # Extract different types of content
        tools_mentioned = Counter()
        activities_discussed = Counter()
        questions_asked = Counter()
        topics_mentioned = Counter()
        recurring_phrases = Counter()
        meeting_related = Counter()
        time_related = Counter()
        
        # Enhanced patterns for different content types
        tool_patterns = [
            r'\b(zapier|n8n|discord|github|google|calendar|zoom|slack|teams|notion|airtable|figma|stripe|paypal)\b',
            r'\b(api|webhook|automation|integration|workflow|pipeline|dashboard|analytics|database|server)\b',
            r'\b(ai|ml|gpt|chatgpt|openai|claude|llm|neural|model|algorithm|machine learning)\b'
        ]
        
        activity_patterns = [
            r'\b(meeting|discussion|session|workshop|demo|presentation|review|planning|brainstorming)\b',
            r'\b(building|creating|developing|designing|implementing|testing|deploying|launching)\b',
            r'\b(learning|studying|researching|exploring|investigating|analyzing|documenting)\b'
        ]
        
        question_patterns = [
            r'\b(how to|how do|what is|what are|where can|when should|why does|which one)\b',
            r'\b(anyone know|does anyone|has anyone|can someone|help with|need help)\b'
        ]
        
        # Specific patterns for common Discord conversation topics
        meeting_patterns = [
            r'\b(meeting|session|call|zoom|teams|recording|agenda|schedule)\b',
            r'\b(today|tomorrow|next week|this week|yesterday|later)\b'
        ]
        
        time_patterns = [
            r'\b(\d{1,2}:\d{2}|morning|afternoon|evening|tonight|today|tomorrow)\b',
            r'\b(hour|minute|time|schedule|calendar|appointment)\b'
        ]
        
        for message in messages:
            if not message or len(message.strip()) < 5:
                continue
            
            # Clean message but preserve more content than BERTopic cleaning
            cleaned = self._clean_content_discord_light(message)
            if len(cleaned.split()) < 2:
                continue
            
            # Extract tools and technologies
            for pattern in tool_patterns:
                matches = re.findall(pattern, cleaned, re.IGNORECASE)
                for match in matches:
                    tools_mentioned[match.lower()] += 1
            
            # Extract activities
            for pattern in activity_patterns:
                matches = re.findall(pattern, cleaned, re.IGNORECASE)
                for match in matches:
                    activities_discussed[match.lower()] += 1
            
            # Extract meeting-related content
            for pattern in meeting_patterns:
                matches = re.findall(pattern, cleaned, re.IGNORECASE)
                for match in matches:
                    meeting_related[match.lower()] += 1
            
            # Extract time-related content
            for pattern in time_patterns:
                matches = re.findall(pattern, cleaned, re.IGNORECASE)
                for match in matches:
                    time_related[match.lower()] += 1
            
            # Extract questions and help requests
            for pattern in question_patterns:
                if re.search(pattern, cleaned, re.IGNORECASE):
                    # Extract the main subject of the question
                    words = cleaned.lower().split()
                    # Look for nouns after question words
                    for i, word in enumerate(words[:-1]):
                        if word in ['about', 'with', 'to', 'how', 'what', 'where']:
                            next_word = words[i + 1]
                            if len(next_word) > 3 and next_word.isalpha():
                                questions_asked[next_word] += 1
            
            # Extract meaningful 2-3 word phrases (improved)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned.lower())
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i + 1]) > 3:
                    phrase = f"{words[i]} {words[i + 1]}"
                    if not self._is_generic_phrase(phrase) and self._is_meaningful_phrase(phrase):
                        recurring_phrases[phrase] += 1
            
            # Extract single meaningful terms
            for word in words:
                if (len(word) > 4 and 
                    self._is_domain_specific_term(word) and 
                    not self._is_generic_word(word)):
                    topics_mentioned[word] += 1
        
        # Build topics from extracted content
        topics = []
        
        # Add meeting/scheduling topics (very common in Discord channels)
        meeting_count = sum(meeting_related.values())
        if meeting_count >= 3:
            topics.append({
                "topic": "Meeting & Scheduling",
                "topic_id": len(topics),
                "frequency": meeting_count,
                "relevance_score": round(meeting_count / len(messages), 3),
                "keywords": list(meeting_related.keys())[:3],
                "keyword_scores": [1.0] * min(3, len(meeting_related)),
                "representative_docs": [],
                "coherence_score": 0.8
            })
        
        # Add tool/technology topics
        for tool, count in tools_mentioned.most_common(5):
            if count >= 2:
                topics.append({
                    "topic": f"{tool.title()} Technology",
                    "topic_id": len(topics),
                    "frequency": count,
                    "relevance_score": round(count / len(messages), 3),
                    "keywords": [tool],
                    "keyword_scores": [1.0],
                    "representative_docs": [],
                    "coherence_score": 0.8
                })
        
        # Add activity topics
        for activity, count in activities_discussed.most_common(3):
            if count >= 2:
                topics.append({
                    "topic": f"{activity.title()} Activities",
                    "topic_id": len(topics),
                    "frequency": count,
                    "relevance_score": round(count / len(messages), 3),
                    "keywords": [activity],
                    "keyword_scores": [1.0],
                    "representative_docs": [],
                    "coherence_score": 0.7
                })
        
        # Add time/scheduling topics
        time_count = sum(time_related.values())
        if time_count >= 3:
            topics.append({
                "topic": "Time & Scheduling",
                "topic_id": len(topics),
                "frequency": time_count,
                "relevance_score": round(time_count / len(messages), 3),
                "keywords": list(time_related.keys())[:3],
                "keyword_scores": [1.0] * min(3, len(time_related)),
                "representative_docs": [],
                "coherence_score": 0.6
            })
        
        # Add recurring phrase topics (more selective)
        for phrase, count in recurring_phrases.most_common(3):
            if count >= 4:  # Higher threshold for phrases
                topics.append({
                    "topic": self._create_topic_name_from_phrase(phrase),
                    "topic_id": len(topics),
                    "frequency": count,
                    "relevance_score": round(count / len(messages), 3),
                    "keywords": phrase.split(),
                    "keyword_scores": [1.0] * len(phrase.split()),
                    "representative_docs": [],
                    "coherence_score": 0.6
                })
        
        # Add question/help topics
        help_topics = [topic for topic, count in questions_asked.most_common(3) if count >= 2]
        if help_topics:
            topics.append({
                "topic": "Help & Questions",
                "topic_id": len(topics),
                "frequency": sum(questions_asked.values()),
                "relevance_score": round(sum(questions_asked.values()) / len(messages), 3),
                "keywords": help_topics,
                "keyword_scores": [1.0] * len(help_topics),
                "representative_docs": [],
                "coherence_score": 0.5
            })
        
        # If no specific topics found, create general conversation categories
        if not topics and len(messages) > 10:
            # Analyze for general conversation patterns
            total_words = sum(len(msg.split()) for msg in messages)
            avg_length = total_words / len(messages)
            
            if avg_length < 10:
                topics.append({
                    "topic": "Quick Updates & Check-ins",
                    "topic_id": 0,
                    "frequency": len(messages),
                    "relevance_score": 1.0,
                    "keywords": ["updates", "check-ins", "quick"],
                    "keyword_scores": [1.0, 1.0, 1.0],
                    "representative_docs": [],
                    "coherence_score": 0.4
                })
            else:
                topics.append({
                    "topic": "Team Discussion",
                    "topic_id": 0,
                    "frequency": len(messages),
                    "relevance_score": 1.0,
                    "keywords": ["team", "discussion", "conversation"],
                    "keyword_scores": [1.0, 1.0, 1.0],
                    "representative_docs": [],
                    "coherence_score": 0.4
                })
        
        logger.info(f"Discord-optimized extraction found {len(topics)} topics")
        return topics

    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful for Discord topic analysis."""
        words = phrase.split()
        if len(words) != 2:
            return False
        
        # At least one word should be domain-specific or meaningful
        meaningful_count = sum(1 for word in words if 
                             self._is_domain_specific_term(word) or 
                             len(word) > 5 or
                             word in ['meeting', 'session', 'team', 'project', 'work', 'time', 'schedule'])
        
        return meaningful_count >= 1

    def _clean_content_discord_light(self, text: str) -> str:
        """Light cleaning for Discord content that preserves more meaningful content."""
        # Remove Discord-specific patterns but preserve more content
        text = re.sub(r"<@[!&]?\d+>", "", text)  # Remove mentions
        text = re.sub(r"<#\d+>", "", text)  # Remove channel references
        text = re.sub(r"<:\w+:\d+>", "", text)  # Remove custom emojis
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r"`[^`]+`", "", text)  # Remove inline code
        
        # Remove excessive punctuation but keep some
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[.]{3,}", "...", text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _is_generic_phrase(self, phrase: str) -> bool:
        """Check if a phrase is too generic for topic analysis."""
        generic_phrases = {
            'you know', 'i think', 'we can', 'it is', 'this is', 'that is',
            'will be', 'can be', 'have been', 'has been', 'going to',
            'need to', 'want to', 'able to', 'try to', 'how to',
            'thank you', 'thanks for', 'let me', 'let us', 'right now',
            'make sure', 'as well', 'of course', 'by the', 'in the',
            'on the', 'at the', 'for the', 'with the', 'from the'
        }
        return phrase.lower() in generic_phrases

    def _create_topic_name_from_phrase(self, phrase: str) -> str:
        """Create a topic name from a recurring phrase."""
        words = phrase.split()
        if len(words) == 2:
            return f"{words[0].title()} & {words[1].title()}"
        else:
            return phrase.title()

    def _is_domain_specific_term(self, word: str) -> bool:
        """Check if a word is domain-specific and worth including in topics."""
        # Technical domains
        tech_terms = {
            'ai', 'ml', 'api', 'cloud', 'data', 'algorithm', 'neural', 'model', 'llm', 'gpt',
            'automation', 'pipeline', 'framework', 'metrics', 'analytics', 'deployment',
            'infrastructure', 'database', 'server', 'frontend', 'backend', 'integration',
            'optimization', 'workflow', 'strategy', 'architecture', 'microservices',
            'kubernetes', 'docker', 'aws', 'azure', 'gcp', 'terraform', 'ansible',
            'python', 'javascript', 'typescript', 'react', 'vue', 'angular', 'node',
            'sql', 'nosql', 'mongodb', 'postgresql', 'redis', 'elasticsearch',
            'machine', 'learning', 'deep', 'neural', 'network', 'transformer',
            'bert', 'chatgpt', 'openai', 'anthropic', 'claude', 'gemini'
        }
        
        # Business domains
        business_terms = {
            'finance', 'marketing', 'sales', 'customer', 'business', 'operations',
            'collaboration', 'team', 'management', 'leadership', 'strategy', 'planning',
            'budget', 'revenue', 'profit', 'investment', 'roi', 'kpi', 'metrics',
            'analytics', 'reporting', 'dashboard', 'visualization', 'insights',
            'growth', 'scaling', 'expansion', 'acquisition', 'partnership',
            'compliance', 'security', 'privacy', 'governance', 'audit'
        }
        
        # Industry-specific terms
        industry_terms = {
            'healthcare', 'education', 'fintech', 'edtech', 'biotech', 'medtech',
            'saas', 'paas', 'iaas', 'b2b', 'b2c', 'enterprise', 'startup',
            'ecommerce', 'retail', 'manufacturing', 'logistics', 'supply',
            'chain', 'inventory', 'warehouse', 'distribution', 'fulfillment'
        }
        
        # Tools and platforms
        tool_terms = {
            'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'teams',
            'zoom', 'notion', 'airtable', 'trello', 'asana', 'monday', 'linear',
            'figma', 'sketch', 'adobe', 'canva', 'miro', 'lucidchart', 'draw.io',
            'stripe', 'paypal', 'square', 'shopify', 'woocommerce', 'magento',
            'salesforce', 'hubspot', 'mailchimp', 'sendgrid', 'twilio', 'zapier'
        }
        
        # Proper nouns (likely companies, products, technologies)
        if word[0].isupper() and len(word) > 3:
            return True
        
        word_lower = word.lower()
        return (word_lower in tech_terms or 
                word_lower in business_terms or 
                word_lower in industry_terms or 
                word_lower in tool_terms)

    def _post_process_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process topics to improve quality and remove noise."""
        if not topics:
            return topics
        
        # Filter out low-quality topics
        filtered_topics = []
        for topic in topics:
            if self._is_high_quality_topic(topic):
                filtered_topics.append(topic)
        
        # If no topics pass quality filter, use moderately relaxed criteria (not too relaxed)
        if not filtered_topics and topics:
            logger.info("No topics passed strict quality filter, using moderately relaxed criteria")
            for topic in topics:
                # Moderately relaxed criteria - still maintain quality
                keywords = topic.get('keywords', [])
                meaningful_keywords = [
                    kw for kw in keywords[:3] 
                    if (len(kw) > 3 and 
                        self._is_domain_specific_term(kw) and 
                        not self._is_generic_word(kw))
                ]
                
                if (topic['frequency'] >= 3 and 
                    topic['relevance_score'] >= 0.015 and  # At least 1.5% of messages
                    len(meaningful_keywords) >= 1 and  # At least 1 domain-specific term
                    topic.get('coherence_score', 0) >= 0.05):
                    filtered_topics.append(topic)
        
        # If still no topics, return empty list rather than junk
        if not filtered_topics:
            logger.warning("No topics met quality criteria - returning empty list")
            return []
        
        # Merge similar topics
        merged_topics = self._merge_similar_topics(filtered_topics)
        
        # Final quality check after merging
        final_topics = []
        for topic in merged_topics:
            # Re-check quality after merging
            if (topic['frequency'] >= 3 and 
                topic['relevance_score'] >= 0.015 and
                len(topic.get('keywords', [])) >= 1):
                final_topics.append(topic)
        
        logger.info(f"Post-processing: {len(topics)} -> {len(filtered_topics)} -> {len(merged_topics)} -> {len(final_topics)} topics")
        return final_topics

    def _is_high_quality_topic(self, topic: Dict[str, Any]) -> bool:
        """Check if a topic meets quality criteria with strict filtering."""
        # Minimum frequency threshold - topics must appear in multiple messages
        if topic['frequency'] < 3:
            return False
        
        # Minimum relevance threshold - at least 2% of messages
        if topic['relevance_score'] < 0.02:
            return False
        
        # Check for meaningful keywords
        keywords = topic.get('keywords', [])
        if not keywords:
            return False
        
        # Comprehensive noise word detection
        noise_words = {
            # From the problematic examples
            'step', 'prompt', 'tip', 'sharing', 'thank', 'very', 'dm', 'chat', 'end', 'gen', 'new', 'care', 'dr', 'abel',
            'love', 'lot', 'personal', 'morning', 'having', 'already', 'via', 'attend', 'accessibility', 'takeaway', 'empowers',
            'global', 'discord', 'development', 'healthcare', 'code', 'coding', 'health',
            
            # Additional conversational noise
            'feel', 'hope', 'wish', 'start', 'begin', 'finish', 'help', 'support', 'understand', 'explain',
            'remember', 'forget', 'happen', 'occur', 'seem', 'appear', 'become', 'turn', 'keep', 'stay',
            'move', 'change', 'continue', 'stop', 'excited', 'boring', 'interesting', 'surprised', 'confused',
            'easy', 'hard', 'difficult', 'simple', 'important', 'useful', 'helpful', 'recently', 'lately',
            'soon', 'early', 'late', 'quick', 'slow', 'fast', 'immediate', 'sudden', 'gradual', 'frequent',
            'occasional', 'rare', 'never', 'always', 'sometimes', 'kind', 'type', 'sort', 'form', 'part',
            'piece', 'bit', 'section', 'area', 'place', 'location', 'position', 'spot', 'point', 'level',
            'stage', 'phase', 'side', 'beginning', 'middle', 'center', 'edge', 'corner', 'line', 'space',
            'room', 'field', 'domain', 'scope', 'range', 'scale', 'size', 'list', 'group', 'set',
            'collection', 'series', 'sequence', 'order', 'arrangement', 'organization', 'structure',
            'format', 'style', 'activity', 'action', 'operation', 'function', 'procedure', 'routine',
            'practice', 'exercise', 'event', 'result', 'outcome', 'effect', 'impact', 'consequence',
            'benefit', 'advantage', 'disadvantage', 'gain', 'loss', 'quality', 'value', 'worth',
            'merit', 'strength', 'weakness', 'positive', 'negative', 'neutral', 'success', 'failure',
            
            # Generic tech terms that don't represent specific topics
            'app', 'tool', 'system', 'platform', 'service', 'product', 'feature', 'version',
            'update', 'issue', 'problem', 'solution', 'option', 'setting', 'config',
            'message', 'text', 'post', 'comment', 'work', 'job', 'task', 'project',
            'process', 'approach', 'way', 'method', 'thing', 'stuff', 'item', 'idea',
            'thought', 'concept', 'time', 'day', 'week', 'month', 'year', 'today',
            'yesterday', 'tomorrow', 'now', 'then', 'later', 'before', 'after', 'during',
            'while', 'since', 'until', 'when', 'some', 'any', 'all', 'every', 'each',
            'other', 'another', 'different', 'same', 'first', 'last', 'next', 'previous',
            'current', 'recent', 'big', 'small', 'large', 'little', 'huge', 'tiny',
            'long', 'short', 'high', 'low', 'top', 'bottom', 'left', 'right', 'up', 'down'
        }
        
        # Filter out topics with only generic/noise words
        meaningful_keywords = []
        domain_specific_keywords = []
        
        for keyword in keywords[:5]:  # Check top 5 keywords
            if (len(keyword) > 2 and 
                keyword.lower() not in noise_words and 
                not keyword.isdigit() and
                keyword.isalpha() and
                not self._is_generic_word(keyword)):
                meaningful_keywords.append(keyword)
                
                # Check if it's domain-specific
                if self._is_domain_specific_term(keyword):
                    domain_specific_keywords.append(keyword)
        
        # Must have at least 2 meaningful keywords for topic quality
        if len(meaningful_keywords) < 2:
            return False
        
        # For high-quality topics, prefer those with at least 1 domain-specific term
        # This helps prioritize technical/business topics over general conversation
        if len(domain_specific_keywords) == 0 and topic['frequency'] < 10:
            return False
        
        # Check coherence score - must be reasonably coherent
        coherence = topic.get('coherence_score', 0)
        if coherence < 0.1:
            return False
        
        # Additional check: topic name should not be generic
        topic_name = topic.get('topic', '')
        if self._is_generic_topic_name(topic_name):
            return False
        
        return True
    
    def _is_generic_word(self, word: str) -> bool:
        """Check if a word is too generic to be meaningful for topics."""
        # Very common English words that slip through
        generic_words = {
            'about', 'above', 'across', 'after', 'again', 'against', 'along', 'among',
            'around', 'back', 'before', 'behind', 'below', 'between', 'both', 'but',
            'down', 'during', 'each', 'even', 'every', 'few', 'from', 'into', 'many',
            'more', 'most', 'much', 'off', 'only', 'other', 'over', 'same', 'some',
            'such', 'than', 'them', 'through', 'too', 'under', 'until', 'very',
            'well', 'what', 'when', 'where', 'which', 'while', 'who', 'why', 'with',
            'within', 'without', 'would', 'yet', 'your'
        }
        return word.lower() in generic_words
    
    def _is_generic_topic_name(self, topic_name: str) -> bool:
        """Check if a topic name is too generic to be meaningful."""
        generic_patterns = [
            'unknown topic', 'discussion', 'conversation', 'general', 'misc', 'other',
            'various', 'multiple', 'different', 'several', 'many', 'some', 'all',
            'new & old', 'good & bad', 'big & small', 'high & low', 'first & last',
            'start & end', 'begin & finish', 'easy & hard', 'simple & complex'
        ]
        topic_lower = topic_name.lower()
        return any(pattern in topic_lower for pattern in generic_patterns)

    def _merge_similar_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge topics that are very similar to each other."""
        if len(topics) <= 1:
            return topics
        
        merged = []
        used_indices = set()
        
        for i, topic1 in enumerate(topics):
            if i in used_indices:
                continue
            
            # Find similar topics to merge
            similar_topics = [topic1]
            similar_indices = [i]
            
            for j, topic2 in enumerate(topics[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_topics_similar(topic1, topic2):
                    similar_topics.append(topic2)
                    similar_indices.append(j)
            
            # Merge similar topics
            if len(similar_topics) > 1:
                merged_topic = self._merge_topic_group(similar_topics)
                merged.append(merged_topic)
                used_indices.update(similar_indices)
            else:
                merged.append(topic1)
                used_indices.add(i)
        
        return merged

    def _are_topics_similar(self, topic1: Dict[str, Any], topic2: Dict[str, Any]) -> bool:
        """Check if two topics are similar enough to merge."""
        keywords1 = set(topic1.get('keywords', [])[:5])
        keywords2 = set(topic2.get('keywords', [])[:5])
        
        # Calculate keyword overlap
        overlap = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return False
        
        jaccard_similarity = overlap / union
        
        # Topics are similar if they share significant keywords
        return jaccard_similarity >= 0.4

    def _merge_topic_group(self, topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of similar topics into one."""
        if len(topics) == 1:
            return topics[0]
        
        # Combine frequencies
        total_frequency = sum(topic['frequency'] for topic in topics)
        
        # Combine relevance scores (weighted average)
        total_relevance = sum(topic['relevance_score'] * topic['frequency'] for topic in topics)
        avg_relevance = total_relevance / total_frequency if total_frequency > 0 else 0
        
        # Combine keywords (take most frequent)
        all_keywords = []
        all_scores = []
        
        for topic in topics:
            keywords = topic.get('keywords', [])
            scores = topic.get('keyword_scores', [])
            for i, keyword in enumerate(keywords):
                if i < len(scores):
                    all_keywords.append(keyword)
                    all_scores.append(scores[i])
        
        # Get unique keywords with highest scores
        keyword_score_map = {}
        for keyword, score in zip(all_keywords, all_scores):
            if keyword not in keyword_score_map or score > keyword_score_map[keyword]:
                keyword_score_map[keyword] = score
        
        # Sort by score and take top keywords
        sorted_keywords = sorted(keyword_score_map.items(), key=lambda x: x[1], reverse=True)
        merged_keywords = [kw for kw, score in sorted_keywords[:10]]
        merged_scores = [score for kw, score in sorted_keywords[:10]]
        
        # Create new topic name
        topic_name = self._generate_topic_name(
            [(kw, score) for kw, score in sorted_keywords[:10]], 
            []
        )
        
        # Combine representative docs
        all_docs = []
        for topic in topics:
            all_docs.extend(topic.get('representative_docs', []))
        
        # Take best coherence score
        best_coherence = max(topic.get('coherence_score', 0) for topic in topics)
        
        return {
            "topic": topic_name,
            "topic_id": topics[0]['topic_id'],  # Use first topic's ID
            "frequency": total_frequency,
            "relevance_score": round(avg_relevance, 3),
            "keywords": merged_keywords,
            "keyword_scores": [round(score, 3) for score in merged_scores],
            "representative_docs": all_docs[:5],  # Limit to 5 docs
            "coherence_score": round(best_coherence, 3)
        }

    def _clean_messages(self, messages: List[str]) -> List[str]:
        """Clean and filter messages for topic modeling with aggressive noise removal."""
        cleaned = []
        
        for message in messages:
            if not message or not message.strip():
                continue
                
            # Apply aggressive Discord cleaning
            text = self._clean_content_discord(message)
            
            # Additional cleaning for topic modeling
            # Remove common conversational patterns
            text = re.sub(r'\b(just|well|dont|cant|wont|isnt|arent|wasnt|werent|hasnt|havent|hadnt|didnt|doesnt|wouldnt|couldnt|shouldnt)\b', '', text, flags=re.IGNORECASE)
            
            # Remove single characters and numbers
            text = re.sub(r'\b[a-zA-Z0-9]\b', '', text)
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Filter out messages that are too short or too generic after cleaning
            if len(text.split()) >= 4:  # At least 4 meaningful words
                # Check if message has meaningful content (not just stop words)
                words = text.lower().split()
                stop_words = set(self._get_tech_stop_words())
                meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
                
                # Only keep messages with at least 2 meaningful words
                if len(meaningful_words) >= 2:
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
        """Generate a human-readable topic name using intelligent analysis."""
        if not topic_words:
            return "Unknown Topic"
        
        # Extract top words and their scores
        top_words = [word for word, score in topic_words[:10]]
        
        # Look for meaningful patterns in the top words
        topic_name = self._create_semantic_topic_name(top_words, representative_docs)
        
        # Fallback to improved simple naming if semantic fails
        if not topic_name or topic_name == "Unknown Topic":
            topic_name = self._create_simple_topic_name(top_words)
        
        return topic_name

    def _create_semantic_topic_name(self, words: List[str], representative_docs: List[str]) -> str:
        """Create a semantic topic name by analyzing word patterns and context."""
        if not words:
            return "Unknown Topic"
        
        # Define semantic categories and their patterns
        tech_patterns = {
            'ai_ml': ['ai', 'ml', 'machine', 'learning', 'neural', 'model', 'algorithm', 'llm', 'gpt'],
            'development': ['code', 'coding', 'programming', 'development', 'software', 'app', 'application'],
            'data': ['data', 'database', 'analytics', 'analysis', 'metrics', 'statistics'],
            'automation': ['automation', 'pipeline', 'workflow', 'process', 'automated', 'script'],
            'cloud': ['cloud', 'aws', 'azure', 'server', 'deployment', 'infrastructure'],
            'api': ['api', 'endpoint', 'service', 'integration', 'webhook', 'rest'],
            'frontend': ['frontend', 'ui', 'interface', 'design', 'user', 'experience'],
            'backend': ['backend', 'server', 'database', 'architecture', 'system']
        }
        
        business_patterns = {
            'healthcare': ['healthcare', 'health', 'medical', 'patient', 'doctor', 'clinical', 'care'],
            'education': ['education', 'learning', 'training', 'course', 'student', 'teacher'],
            'finance': ['finance', 'financial', 'money', 'payment', 'cost', 'budget', 'investment'],
            'marketing': ['marketing', 'campaign', 'brand', 'customer', 'sales', 'promotion'],
            'operations': ['operations', 'process', 'workflow', 'efficiency', 'optimization'],
            'strategy': ['strategy', 'planning', 'goal', 'objective', 'vision', 'mission'],
            'collaboration': ['collaboration', 'team', 'teamwork', 'communication', 'meeting']
        }
        
        # Check for technical patterns
        for category, patterns in tech_patterns.items():
            if any(word.lower() in patterns for word in words[:5]):
                # Find the most relevant words for this category
                relevant_words = [w for w in words[:5] if w.lower() in patterns]
                if relevant_words:
                    return f"{category.replace('_', ' ').title()}: {' & '.join(relevant_words[:2]).title()}"
        
        # Check for business patterns
        for category, patterns in business_patterns.items():
            if any(word.lower() in patterns for word in words[:5]):
                relevant_words = [w for w in words[:5] if w.lower() in patterns]
                if relevant_words:
                    return f"{category.title()}: {' & '.join(relevant_words[:2]).title()}"
        
        # Look for compound technical terms
        compound_terms = self._find_compound_terms(words)
        if compound_terms:
            return compound_terms[0].title()
        
        # Check for proper nouns (likely tools, companies, technologies)
        proper_nouns = [w for w in words[:5] if w[0].isupper() and len(w) > 2]
        if proper_nouns:
            return f"Technology: {' & '.join(proper_nouns[:2])}"
        
        return "Unknown Topic"

    def _find_compound_terms(self, words: List[str]) -> List[str]:
        """Find meaningful compound terms from word list."""
        compounds = []
        
        # Look for common tech compound patterns
        tech_compounds = [
            ('machine', 'learning'), ('artificial', 'intelligence'), ('data', 'science'),
            ('software', 'development'), ('web', 'development'), ('mobile', 'development'),
            ('cloud', 'computing'), ('api', 'development'), ('database', 'management'),
            ('user', 'experience'), ('user', 'interface'), ('project', 'management'),
            ('business', 'intelligence'), ('customer', 'service'), ('quality', 'assurance'),
            ('software', 'engineering'), ('system', 'architecture'), ('data', 'analysis'),
            ('process', 'automation'), ('workflow', 'optimization')
        ]
        
        words_lower = [w.lower() for w in words]
        for first, second in tech_compounds:
            if first in words_lower and second in words_lower:
                compounds.append(f"{first} {second}")
        
        return compounds

    def _create_simple_topic_name(self, words: List[str]) -> str:
        """Create a simple but improved topic name from top words."""
        if not words:
            return "Unknown Topic"
        
        # Filter out very short words and numbers
        filtered_words = [w for w in words[:5] if len(w) > 2 and not w.isdigit()]
        
        if not filtered_words:
            return "Unknown Topic"
        
        # Use top 2-3 words, but make them more readable
        if len(filtered_words) >= 3:
            return f"{filtered_words[0].title()} & {filtered_words[1].title()} & {filtered_words[2].title()}"
        elif len(filtered_words) == 2:
            return f"{filtered_words[0].title()} & {filtered_words[1].title()}"
        else:
            return f"{filtered_words[0].title()} Discussion"

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

    def analyze(
        self,
        channel_name: Optional[str] = None,
        days_back: Optional[int] = None,
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
