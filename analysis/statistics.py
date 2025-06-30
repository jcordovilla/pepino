"""
Statistical analysis functions for Discord message analysis
"""
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime


async def update_user_statistics(pool, base_filter: str, args: dict = None) -> str:
    """Enhanced user activity statistics with concept analysis"""
    try:
        # Get top 10 human users by message count
        async with pool.execute(f"""
            SELECT 
                COALESCE(author_display_name, author_name) as display_name,
                author_id,
                COUNT(*) as message_count,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_message_length,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            WHERE {base_filter}
            AND content IS NOT NULL
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
            GROUP BY author_id, COALESCE(author_display_name, author_name)
            ORDER BY message_count DESC
            LIMIT 10
        """) as cursor:
            users = await cursor.fetchall()
        
        if not users:
            return "❌ No human users found in the database."
        
        # Format results with enhanced information (humans only)
        result = "**📊 Top 10 Human User Activity Statistics**\n\n"
        
        for i, user in enumerate(users, 1):
            display_name, author_id, msg_count, channels, avg_length, active_days, first_msg, last_msg = user
            
            # Format display name safely
            safe_name = str(display_name) if display_name else f"User_{author_id[:8]}"
            
            result += f"**{i}. {safe_name}**\n"
            result += f"   • Messages: {msg_count:,}\n"
            result += f"   • Channels: {channels}\n"
            result += f"   • Avg Length: {avg_length:.0f} chars\n"
            result += f"   • Active Days: {active_days}\n"
            
            # Format dates safely
            if first_msg and last_msg:
                try:
                    first_dt = datetime.fromisoformat(first_msg.replace('Z', '+00:00'))
                    last_dt = datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                    result += f"   • Period: {first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}\n"
                except:
                    result += f"   • Period: {first_msg[:10]} to {last_msg[:10]}\n"
            
            result += "\n"
        
        # Add overall statistics
        total_messages = sum(user[2] for user in users)
        avg_channels_per_user = sum(user[3] for user in users) / len(users) if users else 0
        
        result += f"**📈 Summary:**\n"
        result += f"• Top 10 users: {total_messages:,} messages • Avg channels per user: {avg_channels_per_user:.1f}\n"
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error updating user statistics: {str(e)}"


async def update_word_frequencies(cursor, conn) -> str:
    """Update word frequency analysis"""
    try:
        from core import preprocess_text
        from collections import Counter
        import re
        
        # Get recent messages for word frequency analysis
        cursor.execute("""
            SELECT content FROM messages 
            WHERE content IS NOT NULL 
            AND LENGTH(content) > 10
            ORDER BY timestamp DESC 
            LIMIT 10000
        """)
        messages = cursor.fetchall()
        
        if not messages:
            return "No messages found for word frequency analysis."
        
        # Process messages and extract words
        all_words = []
        for msg in messages:
            content = preprocess_text(msg['content'])
            # Extract meaningful words (3+ characters, not just numbers)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Update database
        cursor.execute('DELETE FROM word_frequencies')
        
        for word, frequency in word_counts.most_common(1000):
            cursor.execute('''
                INSERT INTO word_frequencies (word, frequency, last_updated)
                VALUES (?, ?, ?)
            ''', (word, frequency, datetime.now().isoformat()))
        
        conn.commit()
        
        # Return summary
        total_words = len(all_words)
        unique_words = len(word_counts)
        
        result = f"**📝 Word Frequency Analysis Updated**\n\n"
        result += f"• Total words processed: {total_words:,}\n"
        result += f"• Unique words found: {unique_words:,}\n"
        result += f"• Top words stored: {min(1000, unique_words)}\n\n"
        
        result += f"**🔤 Most Common Words:**\n"
        for word, count in word_counts.most_common(10):
            result += f"• {word}: {count:,} occurrences\n"
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error updating word frequencies: {str(e)}"


async def update_conversation_chains(cursor, conn) -> str:
    """Update conversation chain analysis"""
    try:
        # Find conversation threads (messages with replies)
        cursor.execute("""
            SELECT referenced_message_id, COUNT(*) as reply_count
            FROM messages 
            WHERE referenced_message_id IS NOT NULL
            GROUP BY referenced_message_id
            HAVING reply_count > 1
            ORDER BY reply_count DESC
        """)
        threads = cursor.fetchall()
        
        # Clear existing data
        cursor.execute('DELETE FROM conversation_chains')
        
        # Process conversation chains
        chain_count = 0
        for root_id, reply_count in threads:
            # Get the latest message in this chain
            cursor.execute("""
                SELECT id, timestamp FROM messages 
                WHERE referenced_message_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (root_id,))
            
            last_message = cursor.fetchone()
            if last_message:
                cursor.execute('''
                    INSERT INTO conversation_chains 
                    (root_message_id, last_message_id, message_count, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (root_id, last_message['id'], reply_count + 1, datetime.now().isoformat()))
                chain_count += 1
        
        conn.commit()
        
        result = f"**💬 Conversation Chains Updated**\n\n"
        result += f"• Active conversation threads: {chain_count}\n"
        result += f"• Total reply messages: {sum(count for _, count in threads)}\n\n"
        
        if threads:
            result += f"**🔥 Most Active Threads:**\n"
            for root_id, reply_count in threads[:5]:
                result += f"• Thread {root_id}: {reply_count + 1} messages\n"
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error updating conversation chains: {str(e)}"


async def update_temporal_stats(cursor, conn) -> str:
    """Update temporal statistics"""
    try:
        # Clear existing temporal stats
        cursor.execute('DELETE FROM message_temporal_stats')
        
        # Calculate daily statistics by channel
        cursor.execute("""
            SELECT 
                channel_id,
                DATE(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users,
                AVG(LENGTH(content)) as avg_message_length
            FROM messages 
            WHERE timestamp IS NOT NULL 
            AND content IS NOT NULL
            GROUP BY channel_id, DATE(timestamp)
            ORDER BY date DESC
        """)
        
        daily_stats = cursor.fetchall()
        
        # Insert temporal statistics
        for stat in daily_stats:
            cursor.execute('''
                INSERT INTO message_temporal_stats 
                (channel_id, date, message_count, active_users, avg_message_length)
                VALUES (?, ?, ?, ?, ?)
            ''', stat)
        
        conn.commit()
        
        result = f"**📅 Temporal Statistics Updated**\n\n"
        result += f"• Daily statistics calculated: {len(daily_stats)} entries\n"
        
        # Get recent activity summary
        cursor.execute("""
            SELECT 
                SUM(message_count) as total_messages,
                AVG(active_users) as avg_daily_users,
                COUNT(DISTINCT channel_id) as active_channels
            FROM message_temporal_stats 
            WHERE date >= DATE('now', '-7 days')
        """)
        
        recent_stats = cursor.fetchone()
        if recent_stats and recent_stats[0]:
            total_msg, avg_users, channels = recent_stats
            result += f"• Last 7 days: {total_msg} messages across {channels} channels\n"
            result += f"• Average daily active users: {avg_users:.1f}\n"
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error updating temporal stats: {str(e)}"


async def update_temporal_stats_async(pool, base_filter: str) -> str:
    """Update temporal statistics using async database operations"""
    try:
        # Clear existing temporal stats
        async with pool.execute('DELETE FROM message_temporal_stats') as cursor:
            await pool.commit()
        
        # Calculate daily statistics by channel
        async with pool.execute(f"""
            SELECT 
                channel_name,
                DATE(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users,
                AVG(LENGTH(content)) as avg_message_length
            FROM messages 
            WHERE {base_filter}
            AND timestamp IS NOT NULL 
            AND content IS NOT NULL
            GROUP BY channel_name, DATE(timestamp)
            ORDER BY date DESC
        """) as cursor:
            daily_stats = await cursor.fetchall()
        
        # Insert temporal statistics
        for stat in daily_stats:
            async with pool.execute('''
                INSERT INTO message_temporal_stats 
                (channel_id, date, message_count, active_users, avg_message_length)
                VALUES (?, ?, ?, ?, ?)
            ''', stat) as cursor:
                pass
        
        await pool.commit()
        
        result = f"**📅 Temporal Statistics Updated**\n\n"
        result += f"• Daily statistics calculated: {len(daily_stats)} entries\n"
        
        # Get recent activity summary
        async with pool.execute(f"""
            SELECT 
                SUM(message_count) as total_messages,
                AVG(active_users) as avg_daily_users,
                COUNT(DISTINCT channel_id) as active_channels
            FROM message_temporal_stats 
            WHERE date >= DATE('now', '-7 days')
        """) as cursor:
            recent_stats = await cursor.fetchone()
        
        if recent_stats and recent_stats[0]:
            total_msg, avg_users, channels = recent_stats
            result += f"• Last 7 days: {total_msg} messages across {channels} channels\n"
            result += f"• Average daily active users: {avg_users:.1f}\n"
        
        # Get activity trends for the past 30 days
        async with pool.execute(f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users
            FROM messages 
            WHERE {base_filter}
            AND timestamp IS NOT NULL
            AND DATE(timestamp) >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """) as cursor:
            trend_data = await cursor.fetchall()
        
        if trend_data:
            result += f"\n**📈 Activity Trends (Last 30 Days):**\n"
            
            # Calculate some trend statistics
            total_messages_30d = sum(row[1] for row in trend_data)
            avg_messages_per_day = total_messages_30d / len(trend_data) if trend_data else 0
            max_messages_day = max(row[1] for row in trend_data) if trend_data else 0
            min_messages_day = min(row[1] for row in trend_data) if trend_data else 0
            
            result += f"• Total messages: {total_messages_30d:,}\n"
            result += f"• Average per day: {avg_messages_per_day:.1f}\n"
            result += f"• Peak day: {max_messages_day} messages\n"
            result += f"• Quietest day: {min_messages_day} messages\n"
            
            # Show recent activity
            recent_days = trend_data[-7:] if len(trend_data) >= 7 else trend_data
            if recent_days:
                result += f"\n**📊 Recent Activity (Last 7 Days):**\n"
                for date, count, users in recent_days:
                    result += f"• {date}: {count} messages ({users} users)\n"
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error updating temporal stats: {str(e)}"


async def get_enhanced_activity_trends(pool, base_filter: str) -> tuple[str, str]:
    """Enhanced activity trends with comprehensive analytics and visualization"""
    try:
        from collections import Counter
        from core import nlp, get_analysis_patterns, clean_content
        from visualization import create_activity_graph, cleanup_matplotlib
        import re
        
        result = "**📊 Enhanced Activity Trends Analysis**\n\n"
        
        # Get overall server statistics
        async with pool.execute(f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as total_users,
                COUNT(DISTINCT channel_name) as total_channels,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
                COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages
            FROM messages 
            WHERE {base_filter}
            AND content IS NOT NULL
        """) as cursor:
            overall_stats = await cursor.fetchone()
        
        if not overall_stats or overall_stats[0] == 0:
            return "❌ No messages found for analysis.", None
        
        total_messages, total_users, total_channels, avg_length, first_msg, last_msg, bot_messages, human_messages = overall_stats
        
        # Basic Statistics
        result += f"**📈 Overall Server Statistics:**\n"
        result += f"• Total Messages: {total_messages:,}\n"
        result += f"  - Human Messages: {human_messages:,} ({human_messages/total_messages*100:.1f}%)\n"
        result += f"  - Bot Messages: {bot_messages:,} ({bot_messages/total_messages*100:.1f}%)\n"
        result += f"• Total Unique Users: {total_users:,}\n"
        result += f"• Total Channels: {total_channels:,}\n"
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
        
        # Get activity trends for the past 30 days
        async with pool.execute(f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users,
                COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages
            FROM messages 
            WHERE {base_filter}
            AND timestamp IS NOT NULL
            AND DATE(timestamp) >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """) as cursor:
            trend_data = await cursor.fetchall()
        
        if trend_data:
            result += f"**📅 Activity Trends (Last 30 Days):**\n"
            
            # Calculate trend statistics
            total_messages_30d = sum(row[1] for row in trend_data)
            total_human_30d = sum(row[3] for row in trend_data)
            total_bot_30d = total_messages_30d - total_human_30d
            avg_messages_per_day = total_messages_30d / len(trend_data) if trend_data else 0
            avg_human_per_day = total_human_30d / len(trend_data) if trend_data else 0
            max_messages_day = max(row[1] for row in trend_data) if trend_data else 0
            min_messages_day = min(row[1] for row in trend_data) if trend_data else 0
            max_users_day = max(row[2] for row in trend_data) if trend_data else 0
            
            result += f"• Total Messages: {total_messages_30d:,} ({total_human_30d:,} human, {total_bot_30d} bot)\n"
            result += f"• Average per day: {avg_messages_per_day:.1f} ({avg_human_per_day:.1f} human)\n"
            result += f"• Peak day: {max_messages_day} messages\n"
            result += f"• Quietest day: {min_messages_day} messages\n"
            result += f"• Peak active users: {max_users_day} users\n"
            
            # Activity patterns
            result += f"\n**🕐 Activity Patterns:**\n"
            
            # Get activity by hour
            async with pool.execute(f"""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as messages
                FROM messages 
                WHERE {base_filter}
                AND timestamp IS NOT NULL
                AND DATE(timestamp) >= DATE('now', '-30 days')
                GROUP BY strftime('%H', timestamp)
                ORDER BY messages DESC
                LIMIT 5
            """) as cursor:
                hour_activity = await cursor.fetchall()
            
            if hour_activity:
                result += f"**Peak Activity Hours:**\n"
                for hour, count in hour_activity:
                    result += f"• {hour}:00-{hour}:59: {count} messages\n"
            
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
                WHERE {base_filter}
                AND timestamp IS NOT NULL
                AND DATE(timestamp) >= DATE('now', '-30 days')
                GROUP BY strftime('%w', timestamp)
                ORDER BY messages DESC
            """) as cursor:
                day_activity = await cursor.fetchall()
            
            if day_activity:
                result += f"\n**📅 Activity by Day of Week:**\n"
                for day, count in day_activity:
                    result += f"• {day}: {count} messages\n"
            
            # Recent activity breakdown - simplified
            recent_days = trend_data[-7:] if len(trend_data) >= 7 else trend_data
            if recent_days:
                result += f"\n**📊 Recent Activity (Last 7 Days):**\n"
                for date, count, users, human_count in recent_days:
                    bot_count = count - human_count
                    if bot_count > 0:
                        result += f"• {date}: {count} messages ({human_count} human, {bot_count} bot, {users} users)\n"
                    else:
                        result += f"• {date}: {count} messages ({users} users)\n"
        
        # Top channels by activity
        async with pool.execute(f"""
            SELECT 
                channel_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_length
            FROM messages 
            WHERE {base_filter}
            AND content IS NOT NULL
            GROUP BY channel_name
            ORDER BY message_count DESC
            LIMIT 10
        """) as cursor:
            top_channels = await cursor.fetchall()
        
        if top_channels:
            result += f"\n**📍 Top Channels by Activity:**\n"
            for channel, count, users, avg_len in top_channels:
                # Clean channel name (remove emojis for better display)
                clean_channel = channel.replace('#', '').strip()
                result += f"• #{clean_channel}: {count:,} messages ({users} users, avg {avg_len:.0f} chars)\n"
        
        # Top users by activity
        async with pool.execute(f"""
            SELECT 
                COALESCE(author_display_name, author_name) as display_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_length
            FROM messages 
            WHERE {base_filter}
            AND content IS NOT NULL
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
            GROUP BY author_id, COALESCE(author_display_name, author_name)
            ORDER BY message_count DESC
            LIMIT 10
        """) as cursor:
            top_users = await cursor.fetchall()
        
        if top_users:
            result += f"\n**👥 Top Human Contributors:**\n"
            for name, count, channels, avg_len in top_users:
                # Clean user name (remove emojis for better display)
                clean_name = name.strip()
                result += f"• {clean_name}: {count:,} messages ({channels} channels, avg {avg_len:.0f} chars)\n"
        
        # Semantic Analysis
        result += f"\n**🧠 Server-Wide Semantic Analysis:**\n"
        
        # Get recent messages for semantic analysis
        async with pool.execute(f"""
            SELECT content
            FROM messages 
            WHERE {base_filter}
            AND content IS NOT NULL 
            AND LENGTH(content) > 20
            ORDER BY timestamp DESC
            LIMIT 500
        """) as cursor:
            recent_messages = await cursor.fetchall()
        
        if recent_messages:
            try:
                # Combine all message content
                all_text = ' '.join([msg[0] for msg in recent_messages if msg[0]])
                
                if all_text.strip():
                    # Clean the text before processing
                    cleaned_text = clean_content(all_text)
                    
                    # Process with spaCy
                    doc = nlp(cleaned_text[:500000])  # Limit processing
                    
                    # Extract named entities
                    entities = Counter()
                    for ent in doc.ents:
                        if ent.label_ in ["ORG", "PRODUCT", "TECH", "PERSON", "GPE"]:
                            entity_text = ent.text.strip().lower()
                            if (len(entity_text) > 2 and 
                                len(entity_text) < 50 and
                                not entity_text.startswith('•') and  # Filter out formatting artifacts
                                not entity_text in ['a', 'an', 'the', 'this', 'that']):
                                entities[entity_text] += 1
                    
                    # Extract topics with better filtering
                    topics = Counter()
                    for chunk in doc.noun_chunks:
                        chunk_text = chunk.text.strip().lower()
                        if (len(chunk_text.split()) >= 2 and 
                            len(chunk_text) > 8 and 
                            len(chunk_text) < 60 and
                            not chunk_text.startswith(('a ', 'an ', 'the ')) and
                            not chunk_text.endswith((' a', ' an', ' the')) and
                            not chunk_text in ['this week', 'my question', 'any issues', 'quick recap', 'our community'] and
                            not any(generic in chunk_text for generic in ['this week', 'my question', 'any issues', 'quick recap', 'our community', 'the week', 'next week', 'last week'])):
                            topics[chunk_text] += 1
                    
                    # Extract technology patterns
                    tech_terms = Counter()
                    tech_patterns_tuple = get_analysis_patterns()
                    all_patterns = []
                    for pattern_group in tech_patterns_tuple:
                        all_patterns.extend(pattern_group)
                    
                    for pattern in all_patterns:
                        try:
                            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                            for match in matches:
                                if isinstance(match, tuple):
                                    for group in match:
                                        if group:
                                            tech_terms[group.lower()] += 1
                                else:
                                    tech_terms[match.lower()] += 1
                        except (TypeError, re.error) as e:
                            continue
                    
                    # Key Entities - with better filtering
                    if entities:
                        result += f"**🏢 Key Entities Discussed:**\n"
                        filtered_entities = []
                        for entity, count in entities.most_common(10):
                            if count > 2:  # Filter noise
                                # Additional filtering for poor quality entities
                                if (len(entity) > 3 and 
                                    not entity.startswith('•') and
                                    not entity.endswith('•') and
                                    not entity in ['a', 'an', 'the', 'this', 'that']):
                                    formatted_entity = ' '.join(word.capitalize() for word in entity.split())
                                    filtered_entities.append(f"• {formatted_entity}: {count} mentions")
                        
                        if filtered_entities:
                            result += '\n'.join(filtered_entities[:8]) + "\n"
                        result += "\n"
                    
                    # Main Topics - with better filtering
                    if topics:
                        result += f"**💬 Main Topics Discussed:**\n"
                        filtered_topics = []
                        for topic, count in topics.most_common(15):
                            if count > 2:  # Filter noise
                                # Additional filtering for poor quality topics
                                if (len(topic) > 8 and 
                                    not topic.startswith('•') and
                                    not topic.endswith('•') and
                                    not any(generic in topic for generic in ['this week', 'my question', 'any issues', 'quick recap', 'our community', 'the week', 'next week', 'last week', 'this month', 'next month'])):
                                    formatted_topic = ' '.join(word.capitalize() for word in topic.split())
                                    filtered_topics.append(f"• {formatted_topic}: {count} discussions")
                        
                        if filtered_topics:
                            result += '\n'.join(filtered_topics[:10]) + "\n"
                        result += "\n"
                    
                    # Technology Terms
                    if tech_terms:
                        result += f"**💻 Technology Terms:**\n"
                        tech_list = []
                        for term, count in tech_terms.most_common(10):
                            if count > 1:
                                tech_list.append(f"• {term.upper()}: {count} mentions")
                        
                        if tech_list:
                            result += '\n'.join(tech_list[:8]) + "\n"
                        result += "\n"
                        
            except Exception as e:
                print(f"Error in semantic analysis: {e}")
        
        # Generate activity chart
        chart_path = None
        try:
            if trend_data:
                # Prepare data for chart
                import pandas as pd
                from datetime import datetime
                
                chart_data = []
                for date, count, users, human_count in trend_data:
                    try:
                        date_obj = datetime.strptime(date, '%Y-%m-%d')
                        chart_data.append({
                            'date': date_obj,
                            'message_count': count,
                            'active_users': users,
                            'human_messages': human_count
                        })
                    except:
                        continue
                
                if chart_data:
                    df = pd.DataFrame(chart_data)
                    chart_path = create_activity_graph(df)
                    cleanup_matplotlib()
                    
        except Exception as e:
            print(f"Error generating activity chart: {e}")
            cleanup_matplotlib()
        
        return result, chart_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in enhanced activity trends: {str(e)}", None


async def run_all_analyses(pool, cursor, conn, base_filter: str) -> str:
    """Run comprehensive analysis suite"""
    try:
        results = []
        
        # Run user statistics
        user_stats = await update_user_statistics(pool, base_filter)
        results.append("✅ User Statistics")
        
        # Run word frequencies
        word_stats = await update_word_frequencies(cursor, conn)
        results.append("✅ Word Frequencies")
        
        # Run conversation chains
        chain_stats = await update_conversation_chains(cursor, conn)
        results.append("✅ Conversation Chains")
        
        # Run temporal stats
        temporal_stats = await update_temporal_stats(cursor, conn)
        results.append("✅ Temporal Statistics")
        
        result = f"**🔍 Complete Analysis Suite Executed**\n\n"
        result += f"**Status:**\n"
        for item in results:
            result += f"• {item}\n"
        
        result += f"\n**📊 Analysis Complete!**\n"
        result += f"All statistical analyses have been updated successfully.\n"
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return f"Error running analyses: {str(e)}"
