"""
Topic analysis and concept extraction for Discord message analysis
"""
import re
from typing import List, Dict, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from core import nlp, extract_complex_concepts, get_analysis_patterns, get_topic_analysis_patterns


async def analyze_topics_spacy(pool, base_filter: str, args: dict = None) -> str:
    """Simplified topic analysis with clean, actionable insights"""
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
        
        # Extract entities and topics
        entities = Counter()
        topics = Counter()
        tech_terms = Counter()
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "TECH", "PERSON", "GPE"]:
                entities[ent.text.lower()] += 1
        
        # Extract noun phrases and compound nouns
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                topics[chunk.text.lower().strip()] += 1
        
        # Extract technology patterns
        tech_patterns_tuple = get_analysis_patterns()
        all_patterns = []
        for pattern_group in tech_patterns_tuple:
            all_patterns.extend(pattern_group)
        
        for pattern in all_patterns:
            try:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    tech_terms[match.lower()] += 1
            except (TypeError, re.error) as e:
                print(f"Error with pattern {pattern}: {e}")
                continue
        
        # Format results
        scope = f"channel #{channel_filter}" if channel_filter else "all channels"
        result = f"**📊 Topic Analysis Results ({scope})**\n\n"
        result += f"**Analyzed {len(messages)} messages**\n\n"
        
        # Top entities
        if entities:
            result += f"**🏢 Key Entities:**\n"
            for entity, count in entities.most_common(8):
                if count > 2:  # Filter noise
                    result += f"• {entity.title()}: {count} mentions\n"
            result += "\n"
        
        # Top topics
        if topics:
            result += f"**🧠 Main Topics:**\n"
            for topic, count in topics.most_common(10):
                if count > 2:  # Filter noise
                    result += f"• {topic.title()}: {count} discussions\n"
            result += "\n"
        
        # Tech terms
        if tech_terms:
            result += f"**💻 Technology Terms:**\n"
            for term, count in tech_terms.most_common(8):
                if count > 1:
                    result += f"• {term.upper()}: {count} mentions\n"
            result += "\n"
        
        # Channel breakdown if analyzing all channels
        if not channel_filter:
            channel_topics = defaultdict(Counter)
            for msg in messages:
                content, timestamp, channel, author = msg
                msg_doc = nlp(content[:1000])
                for chunk in msg_doc.noun_chunks:
                    if len(chunk.text.split()) >= 2:
                        channel_topics[channel][chunk.text.lower().strip()] += 1
            
            result += f"**📍 Top Topics by Channel:**\n"
            for channel, topics_counter in list(channel_topics.items())[:5]:
                result += f"• **#{channel}**: "
                top_topics = [topic for topic, count in topics_counter.most_common(3) if count > 1]
                result += ", ".join([t.title() for t in top_topics[:3]]) + "\n"
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in topic analysis: {str(e)}"


async def extract_concepts_from_content(messages) -> List[str]:
    """Extract most relevant and frequent topics from user messages"""
    try:
        if not messages:
            return []
        
        # Combine all message content
        all_content = ' '.join([msg[0] for msg in messages if msg[0]])
        
        if not all_content.strip():
            return []
        
        # Use spaCy to extract concepts
        doc = nlp(all_content[:500000])  # Limit processing
        
        concepts = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE", "EVENT"]:
                if len(ent.text) > 2 and len(ent.text) < 50:
                    concepts.append(ent.text.strip())
        
        # Extract noun chunks (phrases)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                concepts.append(chunk.text.strip())
        
        # Extract compound terms using dependency parsing
        try:
            for token in doc:
                if hasattr(token, 'pos_') and hasattr(token, 'dep_'):
                    if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["compound", "amod"]:
                        head = token.head
                        if hasattr(head, 'pos_') and head.pos_ in ["NOUN", "PROPN"]:
                            compound = f"{token.text} {head.text}"
                            if len(compound) > 5:
                                concepts.append(compound)
        except Exception as e:
            print(f"Error in dependency parsing: {e}")
            # Continue without dependency parsing
        
        # Use custom concept extraction patterns
        try:
            custom_concepts = extract_complex_concepts(doc)  # Pass the spaCy doc, not the string
            concepts.extend(custom_concepts)
        except Exception as e:
            print(f"Error in custom concept extraction: {e}")
            # Continue without custom concepts
        
        # Remove duplicates and filter
        concepts = list(set(concepts))
        concepts = [c for c in concepts if len(c) > 3 and len(c.split()) <= 4]
        
        # Count frequency and return most common
        concept_counts = Counter(concepts)
        return [concept for concept, count in concept_counts.most_common(20)]
        
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        return []


def filter_boilerplate_phrases(topics):
    """Remove common template/boilerplate phrases from topic list."""
    stop_phrases = set([
        'quick summary', 'tip summary', 'link :*', 'motivational monday', 'action step',
        '2025 link', 'date :*', 'headline :*', 'summary :*', 'tip:', 'summary:',
        'link:', 'headline:', 'date:', 'quick summary :*', 'tip summary :*',
        'summary', 'tip', 'link', 'headline', 'date', 'action', 'step',
        'summary *', 'tip *', 'link *', 'headline *', 'date *',
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
