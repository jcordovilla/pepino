"""
Topic analysis and concept extraction for Discord message analysis
"""
import re
from typing import List, Dict, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from core import nlp, extract_complex_concepts, get_analysis_patterns, get_topic_analysis_patterns


async def analyze_topics_spacy(pool, base_filter: str, args: dict = None) -> str:
    """Enhanced topic analysis with better filtering, normalization, and actionable insights"""
    try:
        # Import required libraries
        try:
            import spacy
            from collections import Counter, defaultdict
            import re
            from datetime import datetime, timedelta
            
            # Load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                return "spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm"
                
        except ImportError:
            return "spaCy not installed. Please install it with: pip install spacy"
        
        # Get channel filter if provided
        channel_filter = None
        if args and "channel_name" in args:
            channel_filter = args["channel_name"]
            
            # Get all available channels
            async with pool.execute(f"""
                SELECT DISTINCT channel_name 
                FROM messages 
                WHERE {base_filter}
            """) as cursor:
                channels = await cursor.fetchall()
                channel_names = [ch[0] for ch in channels]
            
            if channel_filter not in channel_names:
                matches = [ch for ch in channel_names if channel_filter.lower() in ch.lower()]
                if matches:
                    return f"Channel '{channel_filter}' not found. Did you mean: {', '.join(matches[:3])}?"
                else:
                    return f"Channel '{channel_filter}' not found. Available channels: {', '.join(channel_names[:10])}"
        
        # Build query for messages
        if channel_filter:
            query = f"""
                SELECT content, timestamp, channel_name, author_display_name
                FROM messages 
                WHERE {base_filter}
                AND channel_name = ? 
                AND content IS NOT NULL 
                AND LENGTH(content) > 20
                ORDER BY timestamp DESC 
                LIMIT 500
            """
            params = (channel_filter,)
        else:
            query = f"""
                SELECT content, timestamp, channel_name, author_display_name
                FROM messages 
                WHERE {base_filter}
                AND content IS NOT NULL 
                AND LENGTH(content) > 20
                ORDER BY timestamp DESC 
                LIMIT 1000
            """
            params = ()
        
        async with pool.execute(query, params) as cursor:
            messages = await cursor.fetchall()
        
        if not messages:
            return "No messages found for analysis."
        
        # Process with spaCy
        all_text = ' '.join([msg[0] for msg in messages])
        doc = nlp(all_text[:1000000])  # Limit text size
        
        # Enhanced entity extraction with better filtering
        from collections import Counter
        entities = Counter()
        topics = Counter()
        tech_terms = Counter()
        
        # Define generic/boilerplate entities to filter out
        generic_entities = {
            'us', 'the group', 'the server', 'the community', 'the team', 'the meeting',
            'the call', 'the session', 'the link', 'the poll', 'the world', 'the buddy group',
            'your main goal', 'preferred display name', 'time zone', 'learning topics',
            'communication & interaction preferences', 'a lot', 'this week', 'boston time'
        }
        
        # Extract and filter entities
        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            if (ent.label_ in ["ORG", "PRODUCT", "TECH", "PERSON", "GPE", "EVENT"] and
                entity_text not in generic_entities and
                len(entity_text) > 2 and len(entity_text) < 50):
                entities[entity_text] += 1
        
        # Enhanced topic extraction with better filtering
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            # Better filtering for meaningful topics
            if (len(chunk_text.split()) >= 2 and 
                len(chunk_text) > 8 and 
                len(chunk_text) < 60 and
                chunk_text not in generic_entities and
                not chunk_text.startswith(('a ', 'an ', 'the ')) and
                not chunk_text.endswith((' a', ' an', ' the'))):
                topics[chunk_text] += 1
        
        # Extract compound terms using dependency parsing
        for i, token in enumerate(doc[:-1]):
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                doc[i+1].pos_ in ["NOUN", "PROPN"] and
                len(token.text) > 2 and len(doc[i+1].text) > 2):
                compound = f"{token.text} {doc[i+1].text}".lower().strip()
                if (len(compound.split()) >= 2 and 
                    len(compound) > 8 and 
                    len(compound) < 50 and
                    compound not in generic_entities and
                    not compound.startswith(('a ', 'an ', 'the '))):
                    topics[compound] += 1
        
        # Enhanced technology term extraction with grouping
        tech_patterns_tuple = get_analysis_patterns()
        all_patterns = []
        for pattern_group in tech_patterns_tuple:
            all_patterns.extend(pattern_group)
        
        # Define tech term groups for better organization
        tech_groups = {
            'ai_ml': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'genai', 'llm', 'large language model'],
            'cloud': ['cloud', 'aws', 'azure', 'gcp', 'google cloud', 'amazon web services'],
            'collaboration': ['collaboration', 'team', 'strategy', 'innovation', 'transformation'],
            'tools': ['discord', 'linkedin', 'whatsapp', 'slack', 'zoom', 'teams']
        }
        
        # Extract tech terms and group them
        from collections import Counter
        raw_tech_terms = Counter()
        for pattern in all_patterns:
            try:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        for group in match:
                            if group:
                                raw_tech_terms[group.lower()] += 1
                    else:
                        raw_tech_terms[match.lower()] += 1
            except (TypeError, re.error) as e:
                print(f"Error with pattern {pattern}: {e}")
                continue
        
        # Group and normalize tech terms
        for term, count in raw_tech_terms.items():
            if count > 1:  # Filter noise
                # Find which group this term belongs to
                grouped = False
                for group_name, group_terms in tech_groups.items():
                    if any(term in gt or gt in term for gt in group_terms):
                        tech_terms[f"{group_name.upper()}: {term}"] += count
                        grouped = True
                        break
                if not grouped:
                    tech_terms[term.upper()] += count
        
        # Filter out boilerplate topics
        topics = Counter(filter_boilerplate_phrases(list(topics.keys())))
        
        # Format results with better organization
        scope = f"channel #{channel_filter}" if channel_filter else "all channels"
        result = f"**📊 Enhanced Topic Analysis Results ({scope})**\n\n"
        result += f"**Analyzed {len(messages)} messages**\n\n"
        
        # Top entities with context
        if entities:
            result += f"**🏢 Key Entities:**\n"
            for entity, count in entities.most_common(8):
                if count > 2:  # Filter noise
                    # Find example usage
                    example = find_entity_example(messages, entity)
                    if example:
                        result += f"• **{entity.title()}** ({count} mentions) — *{example}*\n"
                    else:
                        result += f"• **{entity.title()}** ({count} mentions)\n"
            result += "\n"
        
        # Top topics with better filtering
        if topics:
            result += f"**🧠 Main Topics:**\n"
            for topic, count in topics.most_common(10):
                if count > 2:  # Filter noise
                    result += f"• **{topic.title()}** ({count} discussions)\n"
            result += "\n"
        
        # Tech terms with grouping
        if tech_terms:
            result += f"**💻 Technology Terms:**\n"
            for term, count in tech_terms.most_common(10):
                if count > 1:
                    result += f"• **{term}** ({count} mentions)\n"
            result += "\n"
        
        # Channel breakdown with unique insights
        if not channel_filter:
            channel_topics = defaultdict(Counter)
            channel_entities = defaultdict(Counter)
            
            for msg in messages:
                content, timestamp, channel, author = msg
                if content and len(content) > 20:
                    msg_doc = nlp(content[:1000])
                    
                    # Extract topics per channel
                    for chunk in msg_doc.noun_chunks:
                        chunk_text = chunk.text.strip().lower()
                        if (len(chunk_text.split()) >= 2 and 
                            chunk_text not in generic_entities and
                            len(chunk_text) > 8):
                            channel_topics[channel][chunk_text] += 1
                    
                    # Extract entities per channel
                    for ent in msg_doc.ents:
                        if ent.label_ in ["ORG", "PRODUCT", "TECH", "PERSON", "GPE"]:
                            entity_text = ent.text.strip().lower()
                            if entity_text not in generic_entities:
                                channel_entities[channel][entity_text] += 1
            
            result += f"**📍 Channel Insights:**\n"
            for channel, topics_counter in list(channel_topics.items())[:5]:
                result += f"• **#{channel}**: "
                
                # Get unique topics for this channel
                top_topics = [topic for topic, count in topics_counter.most_common(5) if count > 1]
                unique_topics = [t for t in top_topics if t not in generic_entities][:3]
                
                if unique_topics:
                    result += ", ".join([t.title() for t in unique_topics])
                else:
                    result += "General discussion"
                
                # Add unique entities if any
                channel_ents = [ent for ent, count in channel_entities[channel].most_common(3) if count > 1]
                if channel_ents:
                    result += f" | Entities: {', '.join([e.title() for e in channel_ents[:2]])}"
                
                result += "\n"
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in topic analysis: {str(e)}"


def find_entity_example(messages, entity):
    """Find an example usage of an entity in messages"""
    try:
        for msg in messages[:50]:  # Check first 50 messages
            if msg[0] and entity.lower() in msg[0].lower():
                # Extract a short context around the entity
                content = msg[0]
                entity_pos = content.lower().find(entity.lower())
                if entity_pos != -1:
                    start = max(0, entity_pos - 30)
                    end = min(len(content), entity_pos + len(entity) + 30)
                    context = content[start:end].strip()
                    if len(context) > 20:
                        return context
        return None
    except:
        return None


async def extract_concepts_from_content(messages) -> List[str]:
    """Extract most relevant and frequent topics from user messages"""
    try:
        if not messages:
            return []
        
        # Combine all message content with better preprocessing
        from core import clean_content
        all_content = ' '.join([msg[0] for msg in messages if msg[0]])
        
        if not all_content.strip():
            return []
        
        # Clean the content before processing
        cleaned_content = clean_content(all_content)
        
        # Check if spaCy is available
        from core import nlp
        if nlp is None:
            # Fallback to basic text processing
            import re
            from collections import Counter as CollectionsCounter
            
            # Simple word-based concept extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', cleaned_content.lower())
            word_counts = CollectionsCounter(words)
            
            # Filter out common words
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'very', 'when', 'there', 'can', 'more', 'about', 'many', 'then', 'them', 'these', 'so', 'people', 'into', 'just', 'like', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'}
            
            concepts = []
            for word, count in word_counts.most_common(20):
                if word not in stop_words and count > 1:
                    concepts.append(word)
            
            return concepts[:10]
        
        # Use spaCy to extract concepts
        doc = nlp(cleaned_content[:500000])  # Limit processing
        
        concepts = []
        
        # Extract named entities with better filtering
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE", "EVENT"]:
                entity_text = ent.text.strip()
                if (len(entity_text) > 2 and 
                    len(entity_text) < 50 and
                    not entity_text.lower() in ['a', 'an', 'the', 'this', 'that']):
                    concepts.append(entity_text)
        
        # Extract noun chunks (phrases) with better filtering
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if (len(chunk_text.split()) >= 2 and 
                len(chunk_text) > 8 and 
                len(chunk_text) < 60 and
                not chunk_text.lower().startswith(('a ', 'an ', 'the ')) and
                not chunk_text.lower().endswith((' a', ' an', ' the'))):
                concepts.append(chunk_text)
        
        # Extract compound terms using dependency parsing with better filtering
        try:
            for token in doc:
                if hasattr(token, 'pos_') and hasattr(token, 'dep_'):
                    if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["compound", "amod"]:
                        head = token.head
                        if hasattr(head, 'pos_') and head.pos_ in ["NOUN", "PROPN"]:
                            compound = f"{token.text} {head.text}".strip()
                            if (len(compound) > 8 and 
                                len(compound) < 50 and
                                not compound.lower().startswith(('a ', 'an ', 'the '))):
                                concepts.append(compound)
        except Exception as e:
            print(f"Error in dependency parsing: {e}")
            # Continue without dependency parsing
        
        # Use custom concept extraction patterns
        try:
            custom_concepts = extract_complex_concepts(doc)  # Pass the spaCy doc, not the string
            # Filter custom concepts
            filtered_custom = []
            for concept in custom_concepts:
                concept_text = concept.strip()
                if (len(concept_text) > 10 and 
                    len(concept_text) < 80 and
                    len(concept_text.split()) >= 3 and
                    not concept_text.lower().startswith(('a ', 'an ', 'the ')) and
                    not concept_text.lower().endswith((' a', ' an', ' the'))):
                    filtered_custom.append(concept_text)
            concepts.extend(filtered_custom)
        except Exception as e:
            print(f"Error in custom concept extraction: {e}")
            # Continue without custom concepts
        
        # Remove duplicates and apply comprehensive filtering
        concepts = list(set(concepts))
        
        # Comprehensive filtering
        filtered_concepts = []
        stop_words = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'meeting', 'call', 'session', 'letter'}
        
        for concept in concepts:
            concept_lower = concept.lower().strip()
            words = concept_lower.split()
            
            # Skip if too short or too long
            if len(concept) < 8 or len(concept) > 60:
                continue
                
            # Skip if starts or ends with articles
            if (concept_lower.startswith(('a ', 'an ', 'the ')) or 
                concept_lower.endswith((' a', ' an', ' the'))):
                continue
                
            # Skip if contains too many stop words
            stop_word_count = sum(1 for word in words if word in stop_words)
            if stop_word_count > len(words) * 0.3:  # More than 30% stop words
                continue
                
            # Skip if it's just generic phrases
            if concept_lower in ['a meeting', 'a call', 'a session', 'a letter', 'the meeting', 'the call']:
                continue
                
            # Skip if it's incomplete (ends with common incomplete patterns)
            if any(concept_lower.endswith(pattern) for pattern in ['* *', '...', 'etc', 'etc.']):
                continue
                
            filtered_concepts.append(concept)
        
        # Count frequency and return most common
        from collections import Counter as CollectionsCounter
        concept_counts = CollectionsCounter(filtered_concepts)
        return [concept for concept, count in concept_counts.most_common(15)]
        
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        return []


def filter_boilerplate_phrases(topics):
    """Remove common template/boilerplate and generic phrases from topic list."""
    stop_phrases = set([
        'quick summary', 'tip summary', 'link :*', 'motivational monday', 'action step',
        '2025 link', 'date :*', 'headline :*', 'summary :*', 'tip:', 'summary:',
        'link:', 'headline:', 'date:', 'quick summary :*', 'tip summary :*',
        'summary', 'tip', 'link', 'headline', 'date', 'action', 'step',
        'summary *', 'tip *', 'link *', 'headline *', 'date *',
        'the group', 'the server', 'the community', 'the team', 'the meeting',
        'the call', 'the session', 'the link', 'the poll', 'the world', 'the buddy group',
        'your main goal', 'preferred display name', 'time zone', 'learning topics',
        'communication & interaction preferences', 'a lot', 'this week', 'boston time',
        'general discussion', "today's meeting"
    ])
    def is_boilerplate(topic):
        t = topic.lower().strip(' :*')
        return t in stop_phrases or t.endswith(':*') or t.endswith(':')
    return [t for t in topics if not is_boilerplate(t)]


def perform_topic_modeling(cursor, conn, channel_id: Optional[str] = None) -> Dict[str, List[str]]:
    """Perform topic modeling on messages"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from core import preprocess_text
    
    query = 'SELECT content FROM messages WHERE content IS NOT NULL'
    params = []
    if channel_id:
        query += ' AND channel_id = ?'
        params.append(channel_id)
    
    cursor.execute(query, params)
    messages = cursor.fetchall()
    
    texts = [preprocess_text(msg['content']) for msg in messages]
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Perform LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    
    # Extract topic words
    feature_names = tfidf.get_feature_names_out()
    topics = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        topics[f'Topic {topic_idx + 1}'] = top_words
    
    return topics
