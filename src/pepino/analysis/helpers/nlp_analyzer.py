"""
Advanced NLP service using spaCy for complex text analysis.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import spacy

from pepino.config import Settings
from pepino.logging_config import get_logger

logger = get_logger(__name__)


class NLPService:
    """Advanced NLP service using spaCy for text analysis."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.nlp = None
        self.model_loaded = False
        self._model_loading = False

    def initialize(self) -> None:
        """Initialize the spaCy model."""
        if self.model_loaded:
            return

        if self._model_loading:
            return

        try:
            self._model_loading = True
            logger.info("Loading spaCy model...")

            # Load model directly
            self.nlp = spacy.load("en_core_web_sm")
            self.model_loaded = True
            logger.info("spaCy model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
        finally:
            self._model_loading = False

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using enhanced spaCy processing for AI/GenAI domain."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return []

        try:
            # Run NLP processing directly
            doc = self.nlp(text)
            concepts = []

            # 1. Extract meaningful noun phrases (lowered threshold for AI/tech terms)
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                # Allow shorter chunks if they contain AI/tech keywords
                if (len(chunk_text.split()) >= 2 and len(chunk_text) > 4) or self._is_ai_tech_term(chunk_text):
                    clean_chunk = self._clean_concept(chunk_text)
                    if clean_chunk:
                        concepts.append(clean_chunk)

            # 2. Extract named entities (expanded for tech/AI entities)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                    clean_entity = self._clean_concept(ent.text)
                    if clean_entity:
                        concepts.append(clean_entity)

            # 3. Enhanced technical compounds (up to 3-word compounds)
            for i, token in enumerate(doc[:-2]):
                # Two-word compounds
                if token.pos_ in ["NOUN", "PROPN", "ADJ"] and doc[i + 1].pos_ in ["NOUN", "PROPN"]:
                    compound = f"{token.text} {doc[i + 1].text}"
                    if len(compound) > 4 and self._is_meaningful_compound(compound):
                        concepts.append(self._clean_concept(compound))
                
                # Three-word compounds for technical terms
                if (i < len(doc) - 2 and 
                    token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                    doc[i + 1].pos_ in ["NOUN", "PROPN", "ADJ"] and 
                    doc[i + 2].pos_ in ["NOUN", "PROPN"]):
                    compound = f"{token.text} {doc[i + 1].text} {doc[i + 2].text}"
                    if self._is_ai_tech_term(compound):
                        concepts.append(self._clean_concept(compound))

            # 4. Extract domain-specific single terms that are highly relevant
            for token in doc:
                if (token.pos_ in ["NOUN", "PROPN"] and 
                    len(token.text) >= 4 and 
                    self._is_high_value_single_term(token.text)):
                    concepts.append(self._clean_concept(token.text))

            # 5. Filter and clean results
            filtered_concepts = []
            for concept in concepts:
                if concept and len(concept) > 2:
                    # Remove concepts that are just stop words or too generic
                    if not self._is_generic_noise(concept):
                        filtered_concepts.append(concept)

            return list(set(filtered_concepts))  # Remove duplicates

        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            return []

    def _is_ai_tech_term(self, text: str) -> bool:
        """Check if text contains AI/tech-related terms."""
        ai_tech_keywords = {
            'ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural',
            'generative', 'gpt', 'llm', 'model', 'algorithm', 'data', 'science',
            'nlp', 'computer', 'vision', 'automation', 'digital', 'transformation',
            'cloud', 'api', 'framework', 'python', 'tensorflow', 'pytorch',
            'openai', 'anthropic', 'huggingface', 'prompt', 'engineering',
            'fine-tuning', 'training', 'inference', 'deployment', 'mlops',
            'rag', 'retrieval', 'augmented', 'generation', 'embedding',
            'vector', 'database', 'similarity', 'semantic', 'chatbot',
            'conversational', 'assistant', 'agent', 'autonomous', 'robotics'
        }
        return any(keyword in text.lower() for keyword in ai_tech_keywords)

    def _is_meaningful_compound(self, text: str) -> bool:
        """Check if compound term is meaningful."""
        # Avoid generic combinations
        generic_patterns = {'in the', 'of the', 'to the', 'for the', 'with the', 'and the'}
        return text.lower() not in generic_patterns and len(text) > 6

    def _is_high_value_single_term(self, text: str) -> bool:
        """Check if single term is high-value for AI/tech domain."""
        high_value_terms = {
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'jupyter', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'openai', 'anthropic', 'huggingface', 'langchain', 'llamaindex',
            'chatgpt', 'claude', 'gemini', 'copilot', 'automation', 'optimization',
            'scalability', 'architecture', 'infrastructure', 'deployment',
            'monitoring', 'analytics', 'visualization', 'dashboard', 'pipeline'
        }
        return text.lower() in high_value_terms

    def _is_generic_noise(self, text: str) -> bool:
        """Filter out generic noise terms."""
        noise_patterns = {
            'the', 'this', 'that', 'these', 'those', 'some', 'many', 'much',
            'other', 'another', 'same', 'different', 'new', 'old', 'good', 'bad',
            'big', 'small', 'high', 'low', 'first', 'last', 'next', 'previous',
            'one', 'two', 'three', 'few', 'several', 'lot', 'bit', 'part'
        }
        return text.lower() in noise_patterns or len(text.strip()) < 3

    def _clean_concept(self, text: str) -> str:
        """Clean and standardize concept text."""
        # Remove extra whitespace and normalize
        cleaned = ' '.join(text.split()).strip()
        
        # Capitalize AI/tech acronyms properly
        tech_acronyms = {
            'ai': 'AI', 'ml': 'ML', 'nlp': 'NLP', 'api': 'API', 'ui': 'UI', 'ux': 'UX',
            'gpt': 'GPT', 'llm': 'LLM', 'sql': 'SQL', 'css': 'CSS', 'html': 'HTML',
            'json': 'JSON', 'xml': 'XML', 'rest': 'REST', 'aws': 'AWS', 'gcp': 'GCP',
            'rag': 'RAG', 'devops': 'DevOps', 'mlops': 'MLOps', 'cicd': 'CI/CD'
        }
        
        words = cleaned.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in tech_acronyms:
                normalized_words.append(tech_acronyms[word_lower])
            else:
                normalized_words.append(word.title())
        
        return ' '.join(normalized_words)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using spaCy."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        try:
            # Run NLP processing directly
            doc = self.nlp(text)

            # Simple sentiment analysis based on positive/negative words
            positive_words = {
                "good",
                "great",
                "awesome",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "brilliant",
                "perfect",
                "love",
                "like",
                "enjoy",
                "happy",
                "excited",
                "thrilled",
                "satisfied",
                "pleased",
            }

            negative_words = {
                "bad",
                "terrible",
                "awful",
                "horrible",
                "disappointing",
                "hate",
                "dislike",
                "angry",
                "sad",
                "frustrated",
                "annoyed",
                "upset",
                "worried",
                "concerned",
                "problem",
                "issue",
            }

            words = [token.text.lower() for token in doc if token.is_alpha]
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)

            total_words = len(words)
            if total_words == 0:
                return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

            # Calculate sentiment score
            sentiment_score = (positive_count - negative_count) / total_words

            # Determine sentiment
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Calculate confidence based on word count
            confidence = min(1.0, total_words / 20.0)

            return {
                "sentiment": sentiment,
                "score": round(sentiment_score, 3),
                "confidence": round(confidence, 3),
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": total_words,
            }

        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

    def get_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return []

        try:
            # Run NLP processing directly
            doc = self.nlp(text)

            entities = []
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_),
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

            return entities

        except Exception as e:
            logger.error(f"Failed to extract named entities: {e}")
            return []

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text using spaCy."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return []

        try:
            # Run NLP processing directly
            doc = self.nlp(text)

            phrases = []

            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                    phrases.append(chunk.text.strip())

            # Extract verb phrases
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    # Get the verb and its objects
                    phrase_parts = [token.text]
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            phrase_parts.append(child.text)
                    if len(phrase_parts) > 1:
                        phrases.append(" ".join(phrase_parts))

            # Remove duplicates and limit results
            unique_phrases = list(set(phrases))
            return unique_phrases[:max_phrases]

        except Exception as e:
            logger.error(f"Failed to extract key phrases: {e}")
            return []

    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity using various metrics."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return {}

        try:
            # Run NLP processing directly
            doc = self.nlp(text)

            # Basic statistics
            word_count = len([token for token in doc if token.is_alpha])
            sentence_count = len(list(doc.sents))
            avg_sentence_length = (
                word_count / sentence_count if sentence_count > 0 else 0
            )

            # Part of speech distribution
            pos_counts = Counter([token.pos_ for token in doc])

            # Named entity count
            entity_count = len(doc.ents)

            # Unique words (vocabulary diversity)
            unique_words = len(
                set([token.text.lower() for token in doc if token.is_alpha])
            )
            lexical_diversity = unique_words / word_count if word_count > 0 else 0

            # Average word length
            word_lengths = [len(token.text) for token in doc if token.is_alpha]
            avg_word_length = (
                sum(word_lengths) / len(word_lengths) if word_lengths else 0
            )

            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2),
                "lexical_diversity": round(lexical_diversity, 3),
                "entity_count": entity_count,
                "pos_distribution": dict(pos_counts),
                "complexity_score": self._calculate_complexity_score(
                    avg_sentence_length, lexical_diversity, entity_count
                ),
            }

        except Exception as e:
            logger.error(f"Failed to analyze text complexity: {e}")
            return {}

    def _calculate_complexity_score(
        self, avg_sentence_length: float, lexical_diversity: float, entity_count: int
    ) -> float:
        """Calculate a complexity score based on various metrics."""
        # Normalize metrics to 0-1 scale
        sentence_score = min(avg_sentence_length / 20.0, 1.0)  # 20+ words = complex
        diversity_score = lexical_diversity  # Already 0-1
        entity_score = min(entity_count / 10.0, 1.0)  # 10+ entities = complex

        # Weighted average
        complexity = sentence_score * 0.4 + diversity_score * 0.4 + entity_score * 0.2
        return round(complexity, 3)

    def batch_analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch."""
        if not texts:
            return []

        results = []
        for text in texts:
            try:
                analysis = {
                    "concepts": self.extract_concepts(text),
                    "sentiment": self.analyze_sentiment(text),
                    "entities": self.get_named_entities(text),
                    "key_phrases": self.extract_key_phrases(text),
                    "complexity": self.analyze_text_complexity(text),
                }
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze text: {e}")
                results.append({})

        return results

    def extract_domain_specific_terms(
        self, text: str, domain: str = "tech"
    ) -> List[str]:
        """Extract domain-specific terms from text."""
        if not self.model_loaded:
            self.initialize()

        # Domain-specific patterns
        domain_patterns = {
            "tech": [
                r"\b(AI|ML|LLM|GPT|API|SDK|REST|JSON|XML|SQL|NoSQL|Docker|Kubernetes)\b",
                r"\b(python|javascript|typescript|react|node|vue|angular|django|flask)\b",
                r"\b(algorithm|framework|library|database|server|client|endpoint)\b",
                r"\b(cloud|aws|azure|gcp|deployment|microservices|monolith)\b",
            ],
            "business": [
                r"\b(ROI|KPI|OKR|MVP|B2B|B2C|SaaS|PaaS|IaaS|CRM|ERP|CMS)\b",
                r"\b(strategy|marketing|sales|revenue|profit|growth|scaling)\b",
                r"\b(startup|enterprise|market|competition|pricing|monetization)\b",
            ],
        }

        patterns = domain_patterns.get(domain, [])
        terms = []

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)

        return list(set(terms))  # Remove duplicates
