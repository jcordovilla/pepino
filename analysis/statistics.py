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
