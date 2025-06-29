"""
Base MessageAnalyzer class for Discord message analysis
"""
import sqlite3
from typing import List, Dict, Optional, Any
from database import init_database_schema
from core import preprocess_text
from analysis import ensure_model_loaded, get_embedding, generate_message_embeddings, find_similar_messages_data
from analysis.statistics import update_user_statistics, update_word_frequencies, update_conversation_chains, update_temporal_stats, run_all_analyses
from analysis.topics import perform_topic_modeling, analyze_topics_spacy
from utils.helpers import format_timestamp


class MessageAnalyzer:
    """Base class for Discord message analysis with synchronous database operations"""
    
    def __init__(self, db_path: str = 'discord_messages.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Initialize database schema
        self._init_db()
        
        # Base filter to exclude sesh bot and test channels
        self.base_filter = """
            author_id != 'sesh' 
            AND author_id != '1362434210895364327'
            AND author_name != 'sesh'
            AND LOWER(author_name) != 'pepe'
            AND LOWER(author_name) != 'pepino'
            AND channel_name NOT LIKE '%test%' 
            AND channel_name NOT LIKE '%playground%' 
            AND channel_name NOT LIKE '%pg%'
        """

    def _init_db(self):
        """Initialize database schema"""
        return init_database_schema(self.cursor, self.conn)

    def __del__(self):
        """Cleanup method to handle connection cleanup safely"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except:
            pass

    async def ensure_model_loaded(self):
        """Ensure the embedding model is loaded"""
        return await ensure_model_loaded()

    def get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding for text if model is loaded"""
        return get_embedding(text)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        return preprocess_text(text)

    def generate_message_embeddings(self, batch_size: int = 100) -> None:
        """Generate and store embeddings for messages"""
        return generate_message_embeddings(self.cursor, self.conn, batch_size)

    async def find_similar_messages(self, args: dict = None) -> str:
        """Find messages similar to a given message"""
        try:
            message_id = args.get('message_id') if args else None
            if not message_id:
                return "Please provide a message ID"
            
            try:
                message_id = int(message_id)  # Ensure message_id is an integer
            except ValueError:
                return "Message ID must be a number"
            
            try:
                target_msg, similarities = await find_similar_messages_data(self.cursor, self.conn, message_id)
                
                # Format results
                result = f"**Messages Similar to Message {message_id}:**\n\n"
                result += f"Original Message: {target_msg['content'][:200]}...\n\n"
                
                result += "**Similar Messages:**\n"
                for msg, similarity in similarities[:5]:  # Top 5 similar messages
                    result += f"ID: {msg['id']} (Similarity: {similarity:.2f})\n"
                    result += f"Content: {msg['content'][:200]}...\n"
                    result += f"Author: {msg['author_id']}\n"
                    result += f"Channel: {msg['channel_name']}\n"
                    result += f"Time: {msg['timestamp']}\n\n"
                
                return result
            except ValueError as ve:
                return f"Error: {str(ve)}"
            except Exception as e:
                return f"Error finding similar messages: {str(e)}"
        except Exception as e:
            return f"Error finding similar messages: {str(e)}"

    def perform_topic_modeling(self, channel_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Perform topic modeling on messages"""
        return perform_topic_modeling(self.cursor, self.conn, channel_id)

    async def update_user_statistics(self, args: dict = None) -> str:
        """Update user activity statistics"""
        # Get top users by message count
        self.cursor.execute(f"""
            SELECT 
                author_id,
                COALESCE(author_display_name, author_name) as display_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT channel_name) as channels_active
            FROM messages
            WHERE {self.base_filter}
            AND content IS NOT NULL
            GROUP BY author_id, COALESCE(author_display_name, author_name)
            ORDER BY message_count DESC
            LIMIT 10
        """)
        
        users = self.cursor.fetchall()
        
        if not users:
            return "❌ No users found in the database."
        
        # Format results
        result = "**📊 Top 10 User Activity Statistics**\n\n"
        
        for i, user in enumerate(users, 1):
            author_id, display_name, msg_count, channels = user
            safe_name = str(display_name) if display_name else f"User_{author_id[:8]}"
            
            result += f"**{i}. {safe_name}**\n"
            result += f"   • Messages: {msg_count:,}\n"
            result += f"   • Channels: {channels}\n\n"
        
        # Add overall statistics
        total_messages = sum(user[2] for user in users)
        avg_channels_per_user = sum(user[3] for user in users) / len(users) if users else 0
        
        result += f"**📈 Summary:**\n"
        result += f"• Top 10 users: {total_messages:,} messages • Avg channels per user: {avg_channels_per_user:.1f}\n"
        
        return result

    def update_word_frequencies(self) -> str:
        """Update word frequency analysis"""
        return update_word_frequencies(self.cursor, self.conn)

    def update_conversation_chains(self) -> str:
        """Update conversation chain analysis"""
        return update_conversation_chains(self.cursor, self.conn)

    def update_temporal_stats(self) -> str:
        """Update temporal statistics"""
        return update_temporal_stats(self.cursor, self.conn)

    def run_all_analyses(self) -> str:
        """Run comprehensive analysis suite"""
        return run_all_analyses(None, self.cursor, self.conn, self.base_filter)

    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display"""
        return format_timestamp(timestamp)
