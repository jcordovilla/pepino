"""
User and channel insights for Discord message analysis
"""
import os
import traceback
from typing import List, Optional, Tuple, Union
from datetime import datetime
from visualization import create_user_activity_chart, create_channel_activity_chart, cleanup_matplotlib
from database import get_channel_name_mapping
from .topics import extract_concepts_from_content
from database import filter_boilerplate_phrases
from core import nlp, get_analysis_patterns, clean_content
from collections import Counter
import re


async def resolve_channel_name(pool, user_input: str, base_filter: str, bot_guilds=None) -> str:
    """Resolve user input to the actual database channel name"""
    try:
        # Get channel mapping (current -> old)
        channel_mapping = await get_channel_name_mapping(pool, bot_guilds)
        
        # Create reverse mapping (current -> database_name)
        reverse_mapping = {}
        for db_name, current_name in channel_mapping.items():
            reverse_mapping[current_name] = db_name
        
        # Try to find the database channel name
        # 1. Direct match with database name
        async with pool.execute("""
            SELECT DISTINCT channel_name
            FROM messages 
            WHERE channel_name = ?
            LIMIT 1
        """, (user_input,)) as cursor:
            direct_match = await cursor.fetchone()
        
        if direct_match:
            return user_input
        
        # 2. Try reverse mapping (current name -> database name)
        if user_input in reverse_mapping:
            return reverse_mapping[user_input]
        
        # 3. Try fuzzy matching with database names
        async with pool.execute("""
            SELECT DISTINCT channel_name
            FROM messages 
            WHERE channel_name IS NOT NULL
        """) as cursor:
            db_channels = await cursor.fetchall()
            db_channel_names = [ch[0] for ch in db_channels if ch[0]]
        
        # Case-insensitive exact match
        for db_channel in db_channel_names:
            if user_input.lower() == db_channel.lower():
                return db_channel
        
        # Partial match
        for db_channel in db_channel_names:
            if user_input.lower() in db_channel.lower() or db_channel.lower() in user_input.lower():
                return db_channel
        
        # If no match found, return original input
        return user_input
        
    except Exception as e:
        print(f"Error resolving channel name: {str(e)}")
        return user_input


async def get_user_insights(pool, base_filter: str, user_name: str) -> Union[str, Tuple[str, str]]:
    """Get comprehensive insights for a specific user matching the original format"""
    try:
        # Find the user by name (case-insensitive)
        async with pool.execute(f"""
            SELECT DISTINCT author_id, author_display_name, author_name
            FROM messages
            WHERE {base_filter}
            AND (LOWER(author_name) LIKE ? OR LOWER(author_display_name) LIKE ?)
            LIMIT 10
        """, (f"%{user_name.lower()}%", f"%{user_name.lower()}%")) as cursor:
            matching_users = await cursor.fetchall()
        
        if not matching_users:
            return f"❌ No user found matching '{user_name}'"
        
        # If multiple matches, find the best one
        best_match = None
        for user in matching_users:
            author_id, display_name, author_name = user
            current_name = display_name or author_name
            if current_name and user_name.lower() == current_name.lower():
                best_match = user
                break
        
        if not best_match:
            best_match = matching_users[0]  # Use first match
        
        author_id, display_name, author_name = best_match
        display_name = display_name or author_name
        
        # Get basic user statistics
        async with pool.execute(f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_message_length,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            WHERE author_id = ? AND {base_filter}
            AND content IS NOT NULL
        """, (author_id,)) as cursor:
            stats = await cursor.fetchone()
        
        if not stats or stats[0] == 0:
            return f"❌ No messages found for user '{display_name}'"
        
        total_messages, channels_active, avg_length, active_days, first_msg, last_msg = stats
        
        # Get detailed channel activity with average message lengths
        async with pool.execute(f"""
            SELECT 
                channel_name, 
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_chars_per_message
            FROM messages
            WHERE author_id = ? AND {base_filter}
            AND content IS NOT NULL
            GROUP BY channel_name
            ORDER BY message_count DESC
            LIMIT 5
        """, (author_id,)) as cursor:
            channel_activity = await cursor.fetchall()
        
        # Get activity by time of day (grouped into periods)
        async with pool.execute(f"""
            SELECT 
                CASE 
                    WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 0 AND 5 THEN 'Night (00-05)'
                    WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning (06-11)'
                    WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon (12-17)'
                    WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 18 AND 23 THEN 'Evening (18-23)'
                    ELSE 'Unknown'
                END as time_period,
                COUNT(*) as messages
            FROM messages
            WHERE author_id = ? AND {base_filter}
            AND timestamp IS NOT NULL
            GROUP BY time_period
            ORDER BY messages DESC
        """, (author_id,)) as cursor:
            time_activity = await cursor.fetchall()
        
        # Get recent messages for content analysis
        async with pool.execute(f"""
            SELECT content
            FROM messages
            WHERE author_id = ? AND {base_filter}
            AND content IS NOT NULL 
            AND LENGTH(content) > 20
            ORDER BY timestamp DESC
            LIMIT 200
        """, (author_id,)) as cursor:
            recent_messages = await cursor.fetchall()
        
        # Enhanced semantic analysis using spaCy (same as channel analysis)
        entities = Counter()
        topics = Counter()
        tech_terms = Counter()
        user_concepts = []
        
        if recent_messages:
            try:
                # Check if spaCy is available
                from core import nlp
                if nlp is None:
                    # Fallback to basic concept extraction
                    user_concepts = await extract_concepts_from_content(recent_messages)
                else:
                    # Combine all message content with better preprocessing
                    all_text = ' '.join([msg[0] for msg in recent_messages if msg[0]])
                    
                    if all_text.strip():
                        # Clean the text before processing
                        cleaned_text = clean_content(all_text)
                        
                        # Process with spaCy
                        doc = nlp(cleaned_text[:500000])  # Limit processing
                        
                        # Extract named entities
                        for ent in doc.ents:
                            if ent.label_ in ["ORG", "PRODUCT", "TECH", "PERSON", "GPE"]:
                                # Normalize entity text
                                entity_text = ent.text.strip().lower()
                                if len(entity_text) > 2 and len(entity_text) < 50:
                                    entities[entity_text] += 1
                        
                        # Extract noun phrases and compound nouns with better filtering
                        for chunk in doc.noun_chunks:
                            chunk_text = chunk.text.strip().lower()
                            # Better filtering for meaningful topics
                            if (len(chunk_text.split()) >= 2 and 
                                len(chunk_text) > 8 and 
                                len(chunk_text) < 60 and
                                not chunk_text.startswith(('a ', 'an ', 'the ')) and
                                not chunk_text.endswith((' a', ' an', ' the'))):
                                topics[chunk_text] += 1
                        
                        # Extract technology patterns
                        tech_patterns_tuple = get_analysis_patterns()
                        all_patterns = []
                        for pattern_group in tech_patterns_tuple:
                            all_patterns.extend(pattern_group)
                        
                        for pattern in all_patterns:
                            try:
                                matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                                for match in matches:
                                    if isinstance(match, tuple):
                                        # Handle regex groups
                                        for group in match:
                                            if group:
                                                tech_terms[group.lower()] += 1
                                    else:
                                        tech_terms[match.lower()] += 1
                            except (TypeError, re.error) as e:
                                print(f"Error with pattern {pattern}: {e}")
                                continue
                        
                        # Extract compound terms using dependency parsing with better filtering
                        for i, token in enumerate(doc[:-1]):
                            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                                doc[i+1].pos_ in ["NOUN", "PROPN"] and
                                len(token.text) > 2 and len(doc[i+1].text) > 2):
                                compound = f"{token.text} {doc[i+1].text}".lower().strip()
                                if (len(compound.split()) >= 2 and 
                                    len(compound) > 8 and 
                                    len(compound) < 50 and
                                    not compound.startswith(('a ', 'an ', 'the '))):
                                    topics[compound] += 1
                        
                        # Use custom concept extraction with better filtering
                        custom_concepts = await extract_concepts_from_content(recent_messages)
                        # Filter out poor quality concepts
                        filtered_concepts = []
                        for concept in custom_concepts:
                            concept_lower = concept.lower().strip()
                            if (len(concept_lower) > 8 and 
                                len(concept_lower) < 60 and
                                len(concept_lower.split()) >= 2 and
                                not concept_lower.startswith(('a ', 'an ', 'the ')) and
                                not concept_lower.endswith((' a', ' an', ' the')) and
                                not concept_lower in ['a meeting', 'a letter', 'a call', 'a session']):
                                filtered_concepts.append(concept)
                        user_concepts.extend(filtered_concepts)
                        
                        # Filter boilerplate phrases
                        topics = Counter(filter_boilerplate_phrases(list(topics.keys())))
                        
                        # Additional filtering for topics
                        filtered_topics = Counter()
                        for topic, count in topics.items():
                            if (count > 1 and 
                                len(topic) > 8 and 
                                not topic.startswith(('a ', 'an ', 'the ')) and
                                not topic.endswith((' a', ' an', ' the'))):
                                filtered_topics[topic] = count
                        topics = filtered_topics
                    
            except Exception as e:
                print(f"Error in enhanced semantic analysis: {e}")
                # Fallback to basic concept extraction
                try:
                    user_concepts = await extract_concepts_from_content(recent_messages)
                except Exception as e2:
                    print(f"Error in fallback concept extraction: {e2}")
        
        # Generate user activity chart for past 30 days
        chart_path = None
        try:
            # Get daily message counts for past 30 days for this user
            async with pool.execute(f"""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as messages
                FROM messages 
                WHERE author_id = ? AND {base_filter}
                AND timestamp IS NOT NULL
                AND DATE(timestamp) >= DATE('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            """, (author_id,)) as cursor:
                daily_activity = await cursor.fetchall()
            
            chart_path = create_user_activity_chart(daily_activity, display_name)
            # Force cleanup after chart creation
            cleanup_matplotlib()
                    
        except Exception as e:
            print(f"Error generating user activity chart: {e}")
            # Cleanup even on error
            cleanup_matplotlib()
        
        # Format results to match the original
        result = f"**User Analysis: {display_name}**\n\n"
        
        # General Statistics
        result += f"**📊 General Statistics:**\n"
        result += f"• Total Messages: {total_messages}\n"
        result += f"• Active Channels: {channels_active}\n"
        result += f"• Average Message Length: {avg_length:.1f} characters\n"
        result += f"• Active Days: {active_days}\n"
        
        if first_msg and last_msg:
            # Format timestamps to match original (YYYY-MM-DD HH:MM)
            try:
                from datetime import datetime
                first_dt = datetime.fromisoformat(first_msg.replace('Z', '+00:00'))
                last_dt = datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                result += f"• First Message: {first_dt.strftime('%Y-%m-%d %H:%M')}\n"
                result += f"• Last Message: {last_dt.strftime('%Y-%m-%d %H:%M')}\n"
            except:
                result += f"• First Message: {first_msg[:16]}\n"
                result += f"• Last Message: {last_msg[:16]}\n"
        
        result += "\n"
        
        # Channel Activity
        if channel_activity:
            result += f"**📍 Channel Activity:**\n"
            for channel, count, avg_chars in channel_activity:
                result += f"• #{channel}: {count} messages (avg {avg_chars:.0f} chars)\n"
            result += "\n"
        
        # Activity by Time of Day
        if time_activity:
            result += f"**🕐 Activity by Time of Day:**\n"
            for period, count in time_activity:
                result += f"• {period}: {count} messages\n"
            result += "\n"
        
        # Enhanced Semantic Analysis Results
        result += f"**🧠 Semantic Analysis Results:**\n"
        
        # Key Entities
        if entities:
            result += f"**🏢 Key Entities Mentioned:**\n"
            for entity, count in entities.most_common(6):
                if count > 1:  # Filter noise
                    # Proper case formatting for entities
                    formatted_entity = ' '.join(word.capitalize() for word in entity.split())
                    result += f"• {formatted_entity}: {count} mentions\n"
            result += "\n"
        
        # Main Topics
        if topics:
            result += f"**💬 Main Topics Discussed:**\n"
            for topic, count in topics.most_common(8):
                if count > 1:  # Filter noise
                    # Proper case formatting for topics
                    formatted_topic = ' '.join(word.capitalize() for word in topic.split())
                    result += f"• {formatted_topic}: {count} discussions\n"
            result += "\n"
        
        # Technology Terms
        if tech_terms:
            result += f"**💻 Technology Terms:**\n"
            for term, count in tech_terms.most_common(6):
                if count > 1:
                    # Keep tech terms in uppercase for consistency
                    result += f"• {term.upper()}: {count} mentions\n"
            result += "\n"
        
        # Key Concepts & Topics
        if user_concepts:
            result += f"**🔍 Key Concepts & Topics:**\n"
            # Format concepts with title case and bullet points
            formatted_concepts = []
            for concept in user_concepts[:8]:
                # Better formatting for concepts
                formatted_concept = ' '.join(word.capitalize() for word in concept.split())
                formatted_concepts.append(f"• {formatted_concept}")
            
            result += '\n'.join(formatted_concepts) + "\n"
        
        # Return both text and chart path if chart was generated
        if chart_path and os.path.exists(chart_path):
            return (result, chart_path)
        else:
            return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error getting user insights: {str(e)}"


async def get_channel_insights(pool, base_filter: str, channel_name: str, bot_guilds=None) -> Union[str, Tuple[str, str]]:
    """Get comprehensive channel statistics and insights"""
    try:
        # Resolve the channel name to the actual database name
        resolved_channel = await resolve_channel_name(pool, channel_name, base_filter, bot_guilds)
        
        # Get basic channel statistics with bot/human differentiation
        async with pool.execute(f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
                COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
                COUNT(DISTINCT CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN author_id END) as unique_human_users
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND content IS NOT NULL
        """, (resolved_channel,)) as cursor:
            stats = await cursor.fetchone()
        
        if not stats or stats[0] == 0:
            # Try to find similar channel names
            async with pool.execute(f"""
                SELECT DISTINCT channel_name, COUNT(*) as msg_count
                FROM messages 
                WHERE {base_filter}
                AND LOWER(channel_name) LIKE ?
                GROUP BY channel_name
                ORDER BY msg_count DESC
                LIMIT 5
            """, (f"%{channel_name.lower()}%",)) as cursor:
                similar_channels = await cursor.fetchall()
            
            if similar_channels:
                suggestions = ", ".join([ch[0] for ch in similar_channels])
                return f"❌ No messages found for channel '{channel_name}'. Did you mean: {suggestions}?"
            else:
                return f"❌ No channel found matching '{channel_name}'"
        
        total_messages, unique_users, avg_length, first_msg, last_msg, bot_messages, human_messages, unique_human_users = stats
        
        # Get engagement metrics (excluding bots)
        async with pool.execute(f"""
            SELECT 
                COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                COUNT(CASE WHEN has_reactions = 1 THEN 1 END) as posts_with_reactions
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """, (resolved_channel,)) as cursor:
            engagement = await cursor.fetchone()
        
        total_replies, original_posts, posts_with_reactions = engagement
        replies_per_post = total_replies / original_posts if original_posts > 0 else 0
        reaction_rate = (posts_with_reactions / total_messages * 100) if total_messages > 0 else 0
        
        # Get top contributors (excluding bots)
        async with pool.execute(f"""
            SELECT 
                COALESCE(author_display_name, author_name) as display_name,
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_chars
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND content IS NOT NULL
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
            GROUP BY author_id, author_display_name, author_name
            ORDER BY message_count DESC
            LIMIT 5
        """, (resolved_channel,)) as cursor:
            contributors = await cursor.fetchall()
        
        # Get peak activity hours
        async with pool.execute(f"""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as messages
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND timestamp IS NOT NULL
            GROUP BY strftime('%H', timestamp)
            ORDER BY messages DESC
            LIMIT 3
        """, (resolved_channel,)) as cursor:
            peak_hours = await cursor.fetchall()
        
        # Get activity by day of week
        async with pool.execute(f"""
            SELECT 
                CASE strftime('%w', timestamp)
                    WHEN '0' THEN 'Sunday'
                    WHEN '1' THEN 'Monday'
                    WHEN '2' THEN 'Tuesday'
                    WHEN '3' THEN 'Wednesday'
                    WHEN '4' THEN 'Thursday'
                    WHEN '5' THEN 'Friday'
                    WHEN '6' THEN 'Saturday'
                END as day_name,
                COUNT(*) as messages
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND timestamp IS NOT NULL
            GROUP BY strftime('%w', timestamp)
            ORDER BY messages DESC
            LIMIT 3
        """, (resolved_channel,)) as cursor:
            day_activity = await cursor.fetchall()
        
        # Get recent activity (last 7 days)
        async with pool.execute(f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as messages
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND timestamp IS NOT NULL
            AND DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        """, (resolved_channel,)) as cursor:
            recent_activity = await cursor.fetchall()
        
        # Get channel health metrics (activity in last week, excluding bots)
        async with pool.execute(f"""
            SELECT COUNT(DISTINCT author_id) as weekly_active
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND timestamp >= datetime('now', '-7 days')
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """, (resolved_channel,)) as cursor:
            weekly_active_result = await cursor.fetchone()
        
        weekly_active = weekly_active_result[0] if weekly_active_result else 0
        channel_amp = (weekly_active / unique_human_users * 100) if unique_human_users > 0 else 0
        
        # Get inactive users (human users who posted before but not in last 7 days)
        async with pool.execute(f"""
            SELECT COUNT(DISTINCT author_id) as inactive_users
            FROM messages 
            WHERE channel_name = ? AND {base_filter}
            AND timestamp < datetime('now', '-7 days')
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
            AND author_id NOT IN (
                SELECT DISTINCT author_id 
                FROM messages 
                WHERE channel_name = ? AND {base_filter}
                AND timestamp >= datetime('now', '-7 days')
                AND (author_is_bot = 0 OR author_is_bot IS NULL)
            )
        """, (resolved_channel, resolved_channel)) as cursor:
            inactive_result = await cursor.fetchone()
        
        inactive_users = inactive_result[0] if inactive_result else 0
        inactive_percentage = (inactive_users / unique_human_users * 100) if unique_human_users > 0 else 0
        
        # Get total channel members (from new channel_members table)
        total_channel_members = 0
        lurkers = 0
        participation_rate = 0
        
        try:
            async with pool.execute(f"""
                SELECT COUNT(DISTINCT user_id) as total_members
                FROM channel_members 
                WHERE channel_name = ?
            """, (resolved_channel,)) as cursor:
                member_result = await cursor.fetchone()
            
            if member_result and member_result[0]:
                total_channel_members = member_result[0]
                lurkers = total_channel_members - unique_human_users
                participation_rate = (unique_human_users / total_channel_members * 100) if total_channel_members > 0 else 0
                
        except Exception as e:
            # If channel_members table doesn't exist or has no data, continue without it
            print(f"Note: Channel membership data not available: {e}")
        
        # Extract top topics using advanced spaCy analysis
        top_topics = []
        try:
            # Get content for topic analysis
            async with pool.execute(f"""
                SELECT content
                FROM messages 
                WHERE channel_name = ? AND {base_filter}
                AND content IS NOT NULL 
                AND LENGTH(content) > 30
                ORDER BY timestamp DESC
                LIMIT 200
            """, (resolved_channel,)) as cursor:
                topic_messages = await cursor.fetchall()
            
            if topic_messages:
                all_text = ' '.join([msg[0] for msg in topic_messages if msg[0]])
                doc = nlp(all_text[:500000]) if all_text else None
                spacy_topics = []
                if doc:
                    for chunk in doc.noun_chunks:
                        if len(chunk.text.split()) >= 2 and len(chunk.text) > 8:
                            spacy_topics.append(chunk.text.strip())
                    for i, token in enumerate(doc[:-1]):
                        if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and doc[i+1].pos_ in ["NOUN", "PROPN"]):
                            compound = f"{token.text} {doc[i+1].text}"
                            if len(compound.split()) >= 2 and len(compound) > 6:
                                spacy_topics.append(compound.strip())
                custom_topics = await extract_concepts_from_content(topic_messages)
                all_topics = spacy_topics + custom_topics
                all_topics = list(dict.fromkeys(all_topics))
                all_topics = filter_boilerplate_phrases(all_topics)
                # Filter for meaningful topics
                filtered_topics = [t for t in all_topics if 6 <= len(t) <= 60 and len(t.split()) >= 2]
                top_topics = [' '.join(word.capitalize() for word in t.split()) for t in filtered_topics[:10]]
        except Exception as e:
            print(f"Error extracting topics: {e}")
        
        # Generate activity chart for past 30 days
        chart_path = None
        try:
            # Get daily message counts for past 30 days
            async with pool.execute(f"""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as messages
                FROM messages 
                WHERE channel_name = ? AND {base_filter}
                AND timestamp IS NOT NULL
                AND DATE(timestamp) >= DATE('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            """, (resolved_channel,)) as cursor:
                daily_activity = await cursor.fetchall()
            
            chart_path = create_channel_activity_chart(daily_activity, resolved_channel)
            # Force cleanup after chart creation
            cleanup_matplotlib()
                    
        except Exception as e:
            print(f"Error generating activity chart: {e}")
            # Cleanup even on error
            cleanup_matplotlib()
        
        # Format results
        result = f"**Channel Analysis: #{resolved_channel}**\n\n"
        
        # Basic Statistics
        result += f"**📊 Basic Statistics:**\n"
        result += f"• Total Messages: {total_messages:,}\n"
        result += f"  - Human Messages: {human_messages:,} ({human_messages/total_messages*100:.1f}%)\n"
        result += f"  - Bot Messages: {bot_messages:,} ({bot_messages/total_messages*100:.1f}%)\n"
        result += f"• Total Unique Users: {unique_users:,}\n"
        result += f"• Unique Human Users: {unique_human_users:,}\n"
        result += f"• Average Message Length: {avg_length:.1f} characters\n"
        
        if first_msg and last_msg:
            try:
                from datetime import datetime
                first_dt = datetime.fromisoformat(first_msg.replace('Z', '+00:00'))
                last_dt = datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                result += f"• First Message: {first_dt.strftime('%Y-%m-%d %H:%M')}\n"
                result += f"• Last Message: {last_dt.strftime('%Y-%m-%d %H:%M')}\n"
            except:
                result += f"• First Message: {first_msg[:16]}\n"
                result += f"• Last Message: {last_msg[:16]}\n"
        
        result += "\n"
        
        # Engagement Metrics (Human Activity Only)
        result += f"**📈 Human Engagement Metrics:**\n"
        result += f"• Average Replies per Original Post: {replies_per_post:.2f}\n"
        result += f"• Posts with Reactions: {reaction_rate:.1f}% ({posts_with_reactions}/{human_messages})\n"
        result += f"• Total Replies: {total_replies:,} | Original Posts: {original_posts:,}\n"
        result += f"• Note: Bot messages excluded from engagement calculations\n\n"
        
        # Top Contributors (Humans Only)
        if contributors:
            result += f"**👥 Top Human Contributors:**\n"
            for name, count, avg_chars in contributors:
                result += f"• {name}: {count:,} messages (avg {avg_chars:.0f} chars)\n"
            result += "\n"
        
        # Peak Activity Hours
        if peak_hours:
            result += f"**⏰ Peak Activity Hours:**\n"
            for hour, count in peak_hours:
                result += f"• {hour}:00-{hour}:59: {count} messages\n"
            result += "\n"
        
        # Activity by Day
        if day_activity:
            result += f"**📅 Activity by Day:**\n"
            for day, count in day_activity:
                result += f"• {day}: {count} messages\n"
            result += "\n"
        
        # Recent Activity
        if recent_activity:
            result += f"**📈 Recent Activity (Last 7 Days):**\n"
            for date, count in recent_activity:
                result += f"• {date}: {count} messages\n"
            result += "\n"
        
        # Channel Health Metrics (Human Activity)
        result += f"**📈 Channel Health Metrics (Human Activity):**\n"
        
        if total_channel_members > 0:
            # Enhanced metrics with full membership data
            result += f"• Total Channel Members: {total_channel_members:,}\n"
            result += f"• Human Members Who Ever Posted: {unique_human_users:,} ({participation_rate:.1f}%)\n"
            result += f"• Weekly Active Human Members: {weekly_active:,} ({(weekly_active/total_channel_members*100):.1f}% of total)\n"
            result += f"• Recently Inactive Human Members: {inactive_users:,} ({inactive_percentage:.1f}% of human posters)\n"
            result += f"• Human Lurkers (Never Posted): {lurkers:,} ({(lurkers/total_channel_members*100):.1f}%)\n"
            result += f"• Human Participation Rate: {participation_rate:.1f}% (members who have posted)\n"
            result += f"• Activity Ratio: {weekly_active:,} active / {inactive_users:,} inactive / {lurkers:,} lurkers\n\n"
        else:
            # Fallback to message-based metrics
            result += f"• Total Human Members Ever Active: {unique_human_users:,}\n"
            result += f"• Weekly Active Human Members: {weekly_active:,} ({channel_amp:.1f}%)\n"
            result += f"• Recently Inactive Human Members: {inactive_users:,} ({inactive_percentage:.1f}%)\n"
            result += f"• Activity Ratio: {weekly_active:,} active / {inactive_users:,} inactive\n"
            result += f"• Note: Full membership data not available - showing message-based metrics only\n\n"
        
        # Top Topics Discussed
        if top_topics:
            result += f"**🧠 Top Topics Discussed:**\n"
            for i, topic in enumerate(top_topics, 1):
                result += f"{i}. {topic}\n"
        
        # Return both text and chart path if chart was generated
        if chart_path and os.path.exists(chart_path):
            return (result, chart_path)
        else:
            return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error getting channel insights: {str(e)}"
