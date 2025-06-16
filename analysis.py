import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from pathlib import Path
import pandas as pd
from wordcloud import WordCloud
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from fuzzywuzzy import fuzz
import base64
import aiosqlite

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create temp directory for graphs
TEMP_DIR = tempfile.mkdtemp()

class MessageAnalyzer:
    """Base class for message analysis with database operations"""
    
    def __init__(self, db_path: str = 'discord_messages.db'):
        """Initialize the analyzer with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.embedding_model = None
        self.model_loaded = False
        self.model_load_error = None
        self._model_loading = False
        
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
        
    async def ensure_model_loaded(self):
        """Ensure the embedding model is loaded"""
        if self.model_loaded:
            return True
            
        if self.model_load_error:
            return False
            
        if self._model_loading:
            return False
            
        try:
            self._model_loading = True
            # Create a new event loop for the model loading
            loop = asyncio.get_event_loop()
            # Run the model loading in a thread pool to avoid blocking
            self.embedding_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            )
            self.model_loaded = True
            return True
        except Exception as e:
            self.model_load_error = str(e)
            return False
        finally:
            self._model_loading = False

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text if model is loaded"""
        if not self.model_loaded:
            return None
        try:
            return self.embedding_model.encode(text)
        except Exception:
            return None

    def _init_db(self):
        """Initialize database schema"""
        # Create messages table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT,
                author_id TEXT,
                content TEXT,
                timestamp TEXT,
                edited_timestamp TEXT,
                is_bot BOOLEAN,
                is_system BOOLEAN,
                is_webhook BOOLEAN,
                has_attachments BOOLEAN,
                has_embeds BOOLEAN,
                has_stickers BOOLEAN,
                has_mentions BOOLEAN,
                has_reactions BOOLEAN,
                has_reference BOOLEAN,
                referenced_message_id TEXT,
                thread_id TEXT,
                thread_archived BOOLEAN,
                thread_archived_at TEXT,
                thread_auto_archive_duration INTEGER,
                thread_locked BOOLEAN,
                thread_member_count INTEGER,
                thread_message_count INTEGER,
                thread_name TEXT,
                thread_owner_id TEXT,
                thread_parent_id TEXT,
                thread_total_message_sent INTEGER,
                thread_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create message_embeddings table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        ''')
        
        # Create word_frequencies table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_frequencies (
                word TEXT,
                frequency INTEGER,
                last_updated TEXT,
                PRIMARY KEY (word)
            )
        ''')
        
        # Create user_statistics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_statistics (
                user_id TEXT,
                channel_id TEXT,
                message_count INTEGER,
                avg_message_length REAL,
                active_hours TEXT,
                last_active TEXT,
                PRIMARY KEY (user_id, channel_id)
            )
        ''')
        
        # Create conversation_chains table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_chains (
                root_message_id TEXT,
                last_message_id TEXT,
                message_count INTEGER,
                created_at TEXT,
                PRIMARY KEY (root_message_id)
            )
        ''')
        
        # Create message_temporal_stats table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_temporal_stats (
                channel_id TEXT,
                date TEXT,
                message_count INTEGER,
                active_users INTEGER,
                avg_message_length REAL,
                PRIMARY KEY (channel_id, date)
            )
        ''')
        
        # Create indexes
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_author ON messages(author_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_reference ON messages(referenced_message_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)')
        
        self.conn.commit()

    def __del__(self):
        self.conn.close()

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions
        text = re.sub(r'<@!?\d+>', '', text)
        # Remove emojis
        text = re.sub(r'<a?:\w+:\d+>', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()

    def generate_message_embeddings(self, batch_size: int = 100) -> None:
        """Generate and store embeddings for messages"""
        self.cursor.execute('SELECT id, content FROM messages WHERE content IS NOT NULL')
        messages = self.cursor.fetchall()
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            texts = [self.preprocess_text(msg['content']) for msg in batch]
            embeddings = self.embedding_model.encode(texts)
            
            for msg, embedding in zip(batch, embeddings):
                self.cursor.execute('''
                    INSERT OR REPLACE INTO message_embeddings (message_id, embedding)
                    VALUES (?, ?)
                ''', (msg['id'], embedding.tobytes()))
            
            self.conn.commit()
            print(f"Processed {i + len(batch)} messages")

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
            
            # Get the target message
            cursor = self.conn.execute("""
                SELECT content, author_id, channel_name, timestamp
                FROM messages 
                WHERE id = ?
            """, (message_id,))
            
            target_msg = cursor.fetchone()
            if not target_msg:
                return f"Message with ID {message_id} not found"
            
            # Get other messages for comparison
            cursor = self.conn.execute("""
                SELECT id, content, author_id, channel_name, timestamp
                FROM messages 
                WHERE id != ? AND content IS NOT NULL AND content != ''
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (message_id,))
            
            messages = cursor.fetchall()
            
            if not messages:
                return "No messages found for comparison"
            
            # Get embeddings
            target_embedding = await self.get_embedding(target_msg['content'])
            if target_embedding is None:
                return "Error: Could not generate embedding for target message"
            
            message_embeddings = []
            for msg in messages:
                embedding = await self.get_embedding(msg['content'])
                if embedding is not None:
                    message_embeddings.append((msg, embedding))
            
            if not message_embeddings:
                return "Error: Could not generate embeddings for comparison messages"
            
            # Calculate similarities
            similarities = []
            for msg, embedding in message_embeddings:
                similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                similarities.append((msg, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
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
            
        except Exception as e:
            return f"Error finding similar messages: {str(e)}"

    def perform_topic_modeling(self, channel_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Perform topic modeling on messages"""
        query = 'SELECT content FROM messages WHERE content IS NOT NULL'
        params = []
        if channel_id:
            query += ' AND channel_id = ?'
            params.append(channel_id)
        
        self.cursor.execute(query, params)
        messages = self.cursor.fetchall()
        
        texts = [self.preprocess_text(msg['content']) for msg in messages]
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Fit LDA model
        lda_output = self.lda.fit_transform(tfidf_matrix)
        
        # Get top words for each topic
        feature_names = self.tfidf.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            topics[f'Topic {topic_idx + 1}'] = top_words
        
        return topics

    async def update_word_frequencies(self, args: dict = None) -> str:
        """Update word frequency statistics"""
        try:
            # Get all messages
            cursor = self.conn.execute(f"""
                WITH filtered_messages AS (
                    SELECT content, author_id, channel_name
                    FROM messages
                    WHERE content IS NOT NULL 
                    AND content != ''
                    AND {self.base_filter}
                )
                SELECT * FROM filtered_messages
            """)
            
            messages = cursor.fetchall()
            if not messages:
                return "No messages found for word frequency analysis"
            
            # Process messages
            word_freq = {}
            for msg in messages:
                words = msg['content'].lower().split()
                for word in words:
                    if word not in stopwords.words('english') and len(word) > 2:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Format results
            result = "**Word Frequency Analysis**\n\n"
            result += "**Most Common Words:**\n"
            for word, freq in sorted_words[:20]:  # Top 20 words
                result += f"{word}: {freq} occurrences\n"
            
            return result
            
        except Exception as e:
            return f"Error updating word frequencies: {str(e)}"

    async def update_user_statistics(self, args: dict = None) -> str:
        """Update user activity statistics"""
        try:
            # Get user statistics
            cursor = self.conn.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE {self.base_filter}
                )
                SELECT 
                    author_id,
                    COALESCE(author_display_name, author_name, author_id) as display_name,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT channel_name) as channels_active,
                    AVG(LENGTH(content)) as avg_message_length
                FROM filtered_messages
                GROUP BY author_id, author_display_name, author_name
                ORDER BY message_count DESC
            """)
            
            users = cursor.fetchall()
            if not users:
                return "No user statistics available"
            
            # Format results
            result = "**User Activity Statistics**\n\n"
            for user in users:
                result += f"User {user['display_name']}:\n"
                result += f"Total Messages: {user['message_count']}\n"
                result += f"Active Channels: {user['channels_active']}\n"
                result += f"Average Message Length: {user['avg_message_length']:.1f} characters\n\n"
            
            return result
            
        except Exception as e:
            return f"Error updating user statistics: {str(e)}"

    def update_conversation_chains(self) -> Dict[str, Any]:
        """Update conversation chain analysis and return the results"""
        # Get all messages with references
        self.cursor.execute('''
            SELECT id, referenced_message_id, timestamp
            FROM messages
            WHERE referenced_message_id IS NOT NULL
            ORDER BY timestamp
        ''')
        
        messages = self.cursor.fetchall()
        chains = defaultdict(list)
        
        for msg in messages:
            chains[msg['referenced_message_id']].append(msg['id'])
        
        # Update database and collect results
        chain_stats = {
            'total_chains': len(chains),
            'longest_chain': 0,
            'avg_chain_length': 0
        }
        
        total_length = 0
        for root_id, chain in chains.items():
            last_id = chain[-1]
            chain_length = len(chain)
            chain_stats['longest_chain'] = max(chain_stats['longest_chain'], chain_length)
            total_length += chain_length
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO conversation_chains 
                (root_message_id, last_message_id, message_count, created_at)
                VALUES (?, ?, ?, ?)
            ''', (root_id, last_id, chain_length, datetime.utcnow().isoformat()))
        
        self.conn.commit()
        
        if chain_stats['total_chains'] > 0:
            chain_stats['avg_chain_length'] = total_length / chain_stats['total_chains']
        
        return chain_stats

    async def update_temporal_stats(self, args: dict = None) -> str:
        """Update temporal activity statistics"""
        try:
            # Get temporal statistics
            cursor = self.conn.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE {self.base_filter}
                )
                SELECT 
                    strftime('%H', timestamp) as hour,
                    strftime('%w', timestamp) as day,
                    COUNT(*) as message_count
                FROM filtered_messages
                GROUP BY hour, day
                ORDER BY day, hour
            """)
            
            stats = cursor.fetchall()
            if not stats:
                return "No temporal statistics available"
            
            # Process statistics
            hourly_stats = {}
            daily_stats = {}
            
            for stat in stats:
                hour = int(stat['hour'])
                day = int(stat['day'])
                count = stat['message_count']
                
                if hour not in hourly_stats:
                    hourly_stats[hour] = 0
                hourly_stats[hour] += count
                
                if day not in daily_stats:
                    daily_stats[day] = 0
                daily_stats[day] += count
            
            # Generate temporal activity chart
            chart_path = await self.generate_temporal_activity_chart(hourly_stats, daily_stats)
            
            # Format results
            result = "**Temporal Activity Analysis**\n\n"
            
            result += "**Activity by Hour:**\n"
            for hour in sorted(hourly_stats.keys()):
                result += f"{hour:02d}:00 - {hour+1:02d}:00: {hourly_stats[hour]} messages\n"
            
            result += "\n**Activity by Day:**\n"
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            for day in range(7):
                result += f"{days[day]}: {daily_stats.get(day, 0)} messages\n"
            
            # Add chart information
            if chart_path:
                result += f"\n**Activity Chart:**\nChart has been generated and saved to: {chart_path}\n"
            
            return result
            
        except Exception as e:
            return f"Error updating temporal statistics: {str(e)}"

    async def generate_temporal_activity_chart(self, hourly_stats: dict, daily_stats: dict) -> str:
        """Generate a chart showing temporal activity patterns"""
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot hourly activity
            hours = sorted(hourly_stats.keys())
            counts = [hourly_stats[h] for h in hours]
            ax1.bar(range(len(hours)), counts)  # Use numeric x-axis
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Message Count')
            ax1.set_title('Message Activity by Hour')
            ax1.set_xticks(range(len(hours)))
            ax1.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)
            
            # Plot daily activity
            days = list(range(7))
            day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            counts = [daily_stats.get(d, 0) for d in days]
            ax2.bar(range(len(days)), counts)  # Use numeric x-axis
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Message Count')
            ax2.set_title('Message Activity by Day')
            ax2.set_xticks(range(len(days)))
            ax2.set_xticklabels(day_names)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot to a temporary file
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, f'temporal_activity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return file_path
            
        except Exception as e:
            print(f"Error generating temporal activity chart: {e}")
            return None

    async def get_channel_insights(self, args: dict = None) -> str:
        """Get insights for a specific channel"""
        try:
            if not args or "channel_name" not in args:
                return "Please specify a channel name. Usage: /analyze channel <channel_name>"
            
            channel_name = args["channel_name"]
            
            # Get all channel names for fuzzy matching
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT DISTINCT channel_name 
                FROM messages 
                WHERE {self.base_filter}
            """)
            channels = [row[0] for row in cursor.fetchall()]
            
            # Find best matching channel
            best_match = None
            best_score = 0
            for channel in channels:
                score = fuzz.ratio(channel.lower(), channel_name.lower())
                if score > best_score and score > 80:  # Require 80% similarity
                    best_score = score
                    best_match = channel
            
            if not best_match:
                return f"Channel '{channel_name}' not found. Available channels: {', '.join(channels)}"
            
            # Get channel statistics
            cursor.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE channel_name = ? AND {self.base_filter}
                )
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT author_id) as unique_users,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM filtered_messages
            """, (str(best_match),))
            
            stats = cursor.fetchone()
            
            # Get top contributors
            cursor.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE channel_name = ? AND {self.base_filter}
                )
                SELECT 
                    COALESCE(author_display_name, author_name, author_id) as display_name,
                    COUNT(*) as message_count
                FROM filtered_messages
                GROUP BY author_id, author_display_name, author_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (str(best_match),))
            
            top_users = cursor.fetchall()
            
            # Get activity by hour
            cursor.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE channel_name = ? AND {self.base_filter}
                )
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as message_count
                FROM filtered_messages
                GROUP BY hour
                ORDER BY hour
            """, (str(best_match),))
            
            hourly_activity = cursor.fetchall()
            
            # Generate activity chart
            chart_path = await self.generate_channel_activity_chart(best_match)
            
            # Format results
            result = f"**Channel Analysis: {best_match}**\n\n"
            
            # Basic statistics
            result += "**Basic Statistics:**\n"
            result += f"Total Messages: {stats['total_messages']}\n"
            result += f"Unique Users: {stats['unique_users']}\n"
            result += f"First Message: {self.format_timestamp(stats['first_message'])}\n"
            result += f"Last Message: {self.format_timestamp(stats['last_message'])}\n\n"
            
            # Top contributors
            result += "**Top Contributors:**\n"
            for user in top_users:
                result += f"{user['display_name']}: {user['message_count']} messages\n"
            
            # Activity by hour
            result += "\n**Activity by Hour:**\n"
            for hour in hourly_activity:
                result += f"{hour['hour']}:00 - {hour['hour']}:59: {hour['message_count']} messages\n"
            
            # Add chart information
            if chart_path:
                result += f"\n**Activity Chart:**\nChart has been generated and saved to: {chart_path}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting channel insights: {str(e)}"

    async def get_user_insights_by_display_name(self, display_name: str) -> str:
        """Get insights for a specific user by their display name"""
        try:
            await self.initialize()
            print(f"Getting insights for display name: {display_name}")

            # Find the username corresponding to the display name
            async with self.pool.execute("""
                SELECT DISTINCT author_name
                FROM messages
                WHERE author_display_name = ?
                LIMIT 1
            """, (display_name,)) as cursor:
                user = await cursor.fetchone()

            if not user:
                return f"Display name '{display_name}' not found."

            author_name = user[0]
            print(f"Matched display name '{display_name}' to username '{author_name}'")

            # Use existing method to get insights by username
            return await self.get_user_insights(author_name)

        except Exception as e:
            return f"Error getting user insights by display name: {str(e)}"

    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all analysis updates and return a comprehensive summary"""
        summary = {}
        
        print("Generating message embeddings...")
        self.generate_message_embeddings()
        
        print("Updating word frequencies...")
        self.update_word_frequencies()
        
        print("Updating user statistics...")
        self.update_user_statistics()
        
        print("Updating conversation chains...")
        self.update_conversation_chains()
        
        print("Updating temporal statistics...")
        self.update_temporal_stats()
        
        print("Performing topic modeling...")
        summary['topics'] = self.perform_topic_modeling()
        
        # Get overall statistics
        self.cursor.execute('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as total_users,
                COUNT(DISTINCT channel_id) as total_channels,
                AVG(LENGTH(content)) as avg_message_length
            FROM messages
        ''')
        summary['overall_stats'] = dict(self.cursor.fetchone())
        
        # Get top channels
        self.cursor.execute('''
            SELECT 
                channel_id,
                channel_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users
            FROM messages
            GROUP BY channel_id
            ORDER BY message_count DESC
            LIMIT 5
        ''')
        summary['top_channels'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Get top users
        self.cursor.execute('''
            SELECT 
                author_id,
                author_name,
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_length
            FROM messages
            GROUP BY author_id
            ORDER BY message_count DESC
            LIMIT 5
        ''')
        summary['top_users'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Get temporal patterns
        self.cursor.execute('''
            SELECT 
                date(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users
            FROM messages
            GROUP BY date(timestamp)
            ORDER BY date DESC
            LIMIT 7
        ''')
        summary['recent_activity'] = [dict(row) for row in self.cursor.fetchall()]
        
        return summary

    def create_activity_graph(self, data: pd.DataFrame) -> str:
        """Create activity over time graph"""
        plt.figure(figsize=(12, 6))
        
        # Plot message count
        plt.plot(data['date'], data['message_count'], label='Messages', linewidth=2)
        
        # Plot active users
        plt.plot(data['date'], data['active_users'], label='Active Users', linewidth=2)
        
        plt.title('Message Activity Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the graph
        graph_path = os.path.join(TEMP_DIR, 'activity_graph.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return graph_path

    def create_channel_activity_pie(self, data: pd.DataFrame) -> str:
        """Create channel activity pie chart"""
        plt.figure(figsize=(10, 10))
        
        # Create pie chart
        plt.pie(data['message_count'], 
                labels=data['channel_name'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True)
        
        plt.title('Channel Activity Distribution', fontsize=14, pad=20)
        
        # Save the graph
        graph_path = os.path.join(TEMP_DIR, 'channel_activity.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return graph_path

    def create_user_activity_bar(self, data: pd.DataFrame) -> str:
        """Create user activity bar chart"""
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(data['username'], data['message_count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        plt.title('Top Users by Message Count', fontsize=14, pad=20)
        plt.xlabel('User', fontsize=12)
        plt.ylabel('Message Count', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the graph
        graph_path = os.path.join(TEMP_DIR, 'user_activity.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return graph_path

    def create_word_cloud(self, word_freq: Dict[str, int]) -> str:
        """Create word cloud from word frequencies"""
        plt.figure(figsize=(12, 8))
        
        # Generate word cloud
        wordcloud = WordCloud(width=1200, height=800,
                            background_color='white',
                            max_words=100,
                            contour_width=3,
                            contour_color='steelblue')
        
        wordcloud.generate_from_frequencies(word_freq)
        
        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words', fontsize=14, pad=20)
        
        # Save the graph
        graph_path = os.path.join(TEMP_DIR, 'word_cloud.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return graph_path

    def cleanup_temp_files(self):
        """Clean up temporary graph files"""
        for file in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except:
                pass
        try:
            os.rmdir(TEMP_DIR)
        except:
            pass

    async def get_available_channels(self) -> List[str]:
        """Get list of available channels in the database"""
        try:
            await self.initialize()
            async with self.pool.execute("""
                SELECT DISTINCT channel_name 
                FROM messages 
                WHERE channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                ORDER BY channel_name
            """) as cursor:
                channels = await cursor.fetchall()
                return [channel[0] for channel in channels]
        except Exception as e:
            print(f"Error getting available channels: {str(e)}")
            return []

    async def get_available_users(self) -> List[str]:
        """Get list of available users in the database"""
        try:
            await self.initialize()
            async with self.pool.execute("""
                SELECT DISTINCT author_name 
                FROM messages 
                WHERE author_name IS NOT NULL 
                AND author_id != 'sesh'
                AND author_id != '1362434210895364327'
                AND author_name != 'sesh'
                AND LOWER(author_name) != 'pepe'
                AND LOWER(author_name) != 'pepino'
                ORDER BY author_name
            """) as cursor:
                users = await cursor.fetchall()
                return [user[0] for user in users]
        except Exception as e:
            print(f"Error getting available users: {str(e)}")
            return []

class DiscordBotAnalyzer(MessageAnalyzer):
    """Advanced Discord message analyzer with statistical and semantic analysis"""
    
    def __init__(self):
        """Initialize the analyzer with database connection"""
        self.db_path = 'discord_messages.db'  # Default database path
        self.pool = None
        self.base_filter = """
            channel_name NOT LIKE '%test%'
            AND channel_name NOT LIKE '%playground%'
            AND author_id != 'sesh'
            AND author_id != '1362434210895364327'
            AND author_name != 'sesh'
            AND LOWER(author_name) != 'pepe'
            AND LOWER(author_name) != 'pepino'
        """
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)

    async def initialize(self):
        """Initialize the database connection pool"""
        if self.pool is None:
            try:
                self.pool = await aiosqlite.connect(self.db_path)
                print(f"Database connection established to {self.db_path}")
            except Exception as e:
                print(f"Error connecting to database: {str(e)}")
                raise

    async def close(self):
        """Close the database connection"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("Database connection closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.pool:
            asyncio.create_task(self.close())

    def format_timestamp(self, timestamp: str) -> str:
        """Format a timestamp string to a readable format"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M')
        except:
            return timestamp

    async def update_word_frequencies(self, args: dict = None) -> str:
        """Update word frequency statistics - async wrapper"""
        try:
            await self.initialize()
            
            # Get all messages
            async with self.pool.execute(f"""
                SELECT content
                FROM messages
                WHERE content IS NOT NULL 
                AND content != ''
                AND {self.base_filter}
            """) as cursor:
                messages = await cursor.fetchall()
            
            if not messages:
                return "No messages found for word frequency analysis"
            
            # Process messages
            word_freq = {}
            for msg in messages:
                content = msg[0] if msg else ""
                if content:
                    words = content.lower().split()
                    for word in words:
                        # Clean word and filter
                        word = word.strip('.,!?";:()[]{}')
                        if len(word) > 3 and word.isalpha():
                            word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Format results
            result = "**Word Frequency Analysis**\n\n"
            result += "**Most Common Words:**\n"
            for word, freq in sorted_words[:20]:  # Top 20 words
                result += f"{word}: {freq} occurrences\n"
            
            return result
            
        except Exception as e:
            return f"Error updating word frequencies: {str(e)}"

    async def update_user_statistics(self, args: dict = None) -> str:
        """Update user activity statistics - async wrapper"""
        try:
            await self.initialize()
            
            # Get user statistics
            async with self.pool.execute(f"""
                SELECT 
                    author_id,
                    COALESCE(author_display_name, author_name, author_id) as display_name,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT channel_name) as channels_active,
                    AVG(LENGTH(content)) as avg_message_length
                FROM messages
                WHERE {self.base_filter}
                GROUP BY author_id, author_display_name, author_name
                ORDER BY message_count DESC
                LIMIT 20
            """) as cursor:
                users = await cursor.fetchall()
            
            if not users:
                return "No user statistics available"
            
            # Format results
            result = "**User Activity Statistics**\n\n"
            for user in users:
                display_name = user[1] if user[1] else user[0]
                message_count = user[2]
                channels_active = user[3]
                avg_length = user[4] if user[4] else 0
                
                result += f"**{display_name}:**\n"
                result += f"• Total Messages: {message_count}\n"
                result += f"• Active Channels: {channels_active}\n"
                result += f"• Average Message Length: {avg_length:.1f} characters\n\n"
            
            return result
            
        except Exception as e:
            return f"Error updating user statistics: {str(e)}"

    async def analyze_topics_spacy(self, args: dict = None) -> str:
        """Analyze topics in messages using advanced spaCy NLP with trend analysis"""
        try:
            await self.initialize()
            
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
                async with self.pool.execute(f"""
                    SELECT DISTINCT channel_name 
                    FROM messages 
                    WHERE {self.base_filter}
                """) as cursor:
                    channels = await cursor.fetchall()
                    channel_names = [ch[0] for ch in channels]
                
                if channel_filter not in channel_names:
                    matches = [ch for ch in channel_names if channel_filter.lower() in ch.lower()]
                    if matches:
                        channel_filter = matches[0]
                    else:
                        return f"Channel '{channel_filter}' not found. Available channels: {', '.join(channel_names[:10])}"
            
            # Get messages with timestamps for trend analysis
            if channel_filter:
                async with self.pool.execute(f"""
                    SELECT content, timestamp, author_name
                    FROM messages 
                    WHERE channel_name = ? AND {self.base_filter}
                    AND content IS NOT NULL AND content != ''
                    AND LENGTH(content) > 30
                    ORDER BY timestamp DESC 
                    LIMIT 800
                """, (channel_filter,)) as cursor:
                    messages = await cursor.fetchall()
            else:
                async with self.pool.execute(f"""
                    SELECT content, timestamp, author_name
                    FROM messages 
                    WHERE {self.base_filter}
                    AND content IS NOT NULL AND content != ''
                    AND LENGTH(content) > 30
                    ORDER BY timestamp DESC 
                    LIMIT 800
                """) as cursor:
                    messages = await cursor.fetchall()
            
            if not messages:
                return "No messages found for topic analysis."
            
            # Advanced text cleaning to remove Discord noise
            def clean_content(text):
                # Remove Discord-specific patterns
                text = re.sub(r'<@[!&]?\d+>', '', text)  # Remove mentions
                text = re.sub(r'<#\d+>', '', text)  # Remove channel references
                text = re.sub(r'<:\w+:\d+>', '', text)  # Remove custom emojis
                text = re.sub(r'https?://\S+', '', text)  # Remove URLs
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
                text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
                text = re.sub(r'[🎯🏷️💡🔗🔑📝🌎🏭]', '', text)  # Remove emoji noise
                text = re.sub(r'\b(time zone|buddy group|display name|main goal|learning topics)\b', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\b(the server|the session|the recording|the future)\b', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\b(messages?|channel|group|topic|session|meeting)\b', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\b\d+\s*(minutes?|hours?|days?|weeks?|months?)\b', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            # Process messages with enhanced analysis
            docs = []
            technical_terms = Counter()
            business_concepts = Counter()
            innovation_indicators = Counter()
            discussion_themes = defaultdict(list)
            temporal_trends = defaultdict(list)
            
            # Define domain-specific patterns
            tech_patterns = [
                r'\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework)\b',
                r'\b(python|javascript|typescript|react|node|docker|kubernetes|aws|azure)\b',
                r'\b(database|sql|nosql|analytics|visualization|dashboard|metrics)\b'
            ]
            
            business_patterns = [
                r'\b(strategy|roadmap|KPI|ROI|revenue|growth|market|customer|client)\b',
                r'\b(product|service|solution|platform|integration|deployment|scale)\b',
                r'\b(team|collaboration|workflow|process|efficiency|optimization)\b'
            ]
            
            innovation_pattern_list = [
                r'\b(innovation|transformation|disruption|breakthrough|cutting.edge)\b',
                r'\b(future|trend|emerging|next.gen|state.of.the.art|revolutionary)\b',
                r'\b(experiment|prototype|pilot|proof.of_concept|MVP|beta)\b'
            ]
            
            # Process messages with advanced spaCy analysis
            docs = []
            technical_terms = Counter()
            business_concepts = Counter()
            innovation_indicators = Counter()
            complex_phrases = Counter()
            semantic_concepts = defaultdict(list)
            discussion_themes = defaultdict(list)
            temporal_trends = defaultdict(list)
            sentence_contexts = []
            
            # Advanced spaCy patterns for complex phrase detection
            def extract_complex_concepts(doc):
                """Extract complex multi-word concepts using dependency parsing"""
                concepts = []
                
                # Extract compound subjects with their predicates
                for token in doc:
                    if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
                        # Get the full noun phrase subject
                        subject_span = doc[token.left_edge.i:token.right_edge.i+1]
                        
                        # Get the predicate (verb and its objects)
                        if token.head.pos_ == "VERB":
                            verb = token.head
                            predicate_parts = [verb.text]
                            
                            # Add direct objects, prepositional objects
                            for child in verb.children:
                                if child.dep_ in ["dobj", "pobj", "attr", "prep"]:
                                    obj_span = doc[child.left_edge.i:child.right_edge.i+1]
                                    predicate_parts.append(obj_span.text)
                            
                            if len(predicate_parts) > 1:
                                full_concept = f"{subject_span.text} {' '.join(predicate_parts)}"
                                if len(full_concept.split()) >= 3 and len(full_concept) > 15:
                                    concepts.append(full_concept.lower().strip())
                
                # Extract complex noun phrases with modifiers
                for chunk in doc.noun_chunks:
                    # Get extended noun phrases including prepositional phrases
                    extended_phrase = chunk.text
                    
                    # Look for prepositional phrases attached to this chunk
                    for token in chunk:
                        for child in token.children:
                            if child.dep_ == "prep":
                                prep_phrase = doc[child.i:child.right_edge.i+1]
                                extended_phrase += f" {prep_phrase.text}"
                    
                    if len(extended_phrase.split()) >= 3 and len(extended_phrase) > 20:
                        concepts.append(extended_phrase.lower().strip())
                
                # Extract technical compounds (noun + noun + noun patterns)
                for i, token in enumerate(doc[:-2]):
                    if (token.pos_ in ["NOUN", "PROPN"] and 
                        doc[i+1].pos_ in ["NOUN", "PROPN", "ADJ"] and 
                        doc[i+2].pos_ in ["NOUN", "PROPN"]):
                        
                        # Check if they form a meaningful technical term
                        compound = f"{token.text} {doc[i+1].text} {doc[i+2].text}"
                        
                        # Look ahead for even longer compounds
                        j = i + 3
                        while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and j < i + 6:
                            compound += f" {doc[j].text}"
                            j += 1
                        
                        if len(compound.split()) >= 3:
                            concepts.append(compound.lower())
                
                return concepts
            
            def extract_semantic_relationships(doc):
                """Extract semantic relationships using dependency parsing"""
                relationships = []
                
                # Find cause-effect relationships
                for token in doc:
                    if token.lemma_ in ["cause", "lead", "result", "enable", "drive", "impact"]:
                        # Get what causes what
                        cause = None
                        effect = None
                        
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                cause_span = doc[child.left_edge.i:child.right_edge.i+1]
                                cause = cause_span.text
                            elif child.dep_ in ["dobj", "attr"]:
                                effect_span = doc[child.left_edge.i:child.right_edge.i+1]
                                effect = effect_span.text
                        
                        if cause and effect and len(f"{cause} → {effect}") > 10:
                            relationships.append(f"{cause} → {effect}")
                
                # Find comparative relationships
                for token in doc:
                    if token.pos_ == "ADJ" and token.dep_ == "acomp":
                        # Get what's being compared
                        subject = None
                        for sibling in token.head.children:
                            if sibling.dep_ == "nsubj":
                                subj_span = doc[sibling.left_edge.i:sibling.right_edge.i+1]
                                subject = subj_span.text
                        
                        if subject:
                            comparison = f"{subject} is {token.text}"
                            if len(comparison) > 8:
                                relationships.append(comparison)
                
                return relationships
            
            # Define advanced domain patterns
            tech_patterns = [
                r'\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework)\b',
                r'\b(python|javascript|typescript|react|node|docker|kubernetes|aws|azure)\b',
                r'\b(database|sql|nosql|analytics|visualization|dashboard|metrics)\b'
            ]
            
            business_patterns = [
                r'\b(strategy|roadmap|KPI|ROI|revenue|growth|market|customer|client)\b',
                r'\b(product|service|solution|platform|integration|deployment|scale)\b',
                r'\b(team|collaboration|workflow|process|efficiency|optimization)\b'
            ]
            
            innovation_pattern_list = [
                r'\b(innovation|transformation|disruption|breakthrough|cutting.edge)\b',
                r'\b(future|trend|emerging|next.gen|state.of.the.art|revolutionary)\b',
                r'\b(experiment|prototype|pilot|proof.of_concept|MVP|beta)\b'
            ]
            
            for i, msg in enumerate(messages):
                content, timestamp, author = msg[0], msg[1], msg[2]
                
                # Skip very short or formulaic messages
                if len(content.split()) < 6:
                    continue
                    
                cleaned_content = clean_content(content)
                if len(cleaned_content.split()) < 4:
                    continue
                
                try:
                    doc = nlp(cleaned_content)
                    
                    # Store sentence-level context for later analysis
                    for sent in doc.sents:
                        if len(sent.text.split()) >= 5:
                            sentence_contexts.append({
                                'text': sent.text,
                                'author': author,
                                'timestamp': timestamp,
                                'main_concepts': [token.lemma_ for token in sent if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]
                            })
                    
                    docs.append((doc, timestamp, author, content))
                    
                    # Extract complex concepts using advanced spaCy features
                    complex_concepts = extract_complex_concepts(doc)
                    for concept in complex_concepts:
                        if len(concept.split()) >= 3:
                            complex_phrases[concept] += 1
                            
                            # Categorize by semantic content
                            if any(tech in concept for tech in ['ai', 'ml', 'data', 'algorithm', 'api', 'tech', 'system', 'platform', 'tool']):
                                semantic_concepts['Advanced Technology'].append(concept)
                            elif any(biz in concept for biz in ['business', 'strategy', 'market', 'customer', 'revenue', 'growth', 'team', 'process']):
                                semantic_concepts['Business Strategy'].append(concept)
                            elif any(inn in concept for inn in ['innovation', 'future', 'transform', 'disrupt', 'emerging', 'new', 'next']):
                                semantic_concepts['Innovation & Future'].append(concept)
                            else:
                                semantic_concepts['General Discussion'].append(concept)
                    
                    # Extract semantic relationships
                    relationships = extract_semantic_relationships(doc)
                    for rel in relationships:
                        semantic_concepts['Cause & Effect'].append(rel)
                    
                    # Extract simple patterns for baseline
                    for pattern in tech_patterns:
                        matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                        for match in matches:
                            technical_terms[match.lower()] += 1
                    
                    for pattern in business_patterns:
                        matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                        for match in matches:
                            business_concepts[match.lower()] += 1
                    
                    for pattern in innovation_pattern_list:
                        matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                        for match in matches:
                            innovation_indicators[match.lower()] += 1
                    
                    # Time-based trend analysis with concepts
                    try:
                        msg_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        week_key = msg_date.strftime('%Y-W%U')
                        
                        # Add complex concepts to temporal trends
                        temporal_trends[week_key].extend([concept.split()[0] for concept in complex_concepts if concept])
                        
                    except:
                        pass
                
                except Exception as e:
                    continue
            
            if not docs:
                return "No meaningful content found for analysis."
            
            # Advanced topic clustering using semantic similarity
            semantic_clusters = defaultdict(list)
            conversation_flows = []
            
            # Analyze conversation flows (who responds to whom about what)
            for i, (doc1, ts1, author1, content1) in enumerate(docs[:100]):
                doc1_concepts = [token.lemma_.lower() for token in doc1 
                               if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop]
                
                for j, (doc2, ts2, author2, content2) in enumerate(docs[:100]):
                    if i >= j or author1 == author2:
                        continue
                    
                    doc2_concepts = [token.lemma_.lower() for token in doc2 
                                   if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop]
                    
                    # Calculate semantic overlap
                    set1, set2 = set(doc1_concepts), set(doc2_concepts)
                    if len(set1.union(set2)) > 0:
                        similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                        if similarity > 0.4:  # Higher threshold for quality
                            common_concepts = list(set1.intersection(set2))[:3]
                            if common_concepts:
                                conversation_flows.append({
                                    'participants': [author1, author2],
                                    'concepts': common_concepts,
                                    'similarity': similarity
                                })
            
            # Trend analysis - identify emerging vs declining topics
            trend_analysis = {}
            if len(temporal_trends) >= 2:
                all_weeks = sorted(temporal_trends.keys())
                recent_weeks = all_weeks[-2:]
                older_weeks = all_weeks[:-2] if len(all_weeks) > 2 else []
                
                recent_concepts = Counter()
                older_concepts = Counter()
                
                for week in recent_weeks:
                    recent_concepts.update(temporal_trends[week])
                for week in older_weeks:
                    older_concepts.update(temporal_trends[week])
                
                # Find emerging topics (high in recent, low in older)
                emerging = []
                declining = []
                
                for concept, recent_count in recent_concepts.most_common(20):
                    older_count = older_concepts.get(concept, 0)
                    if recent_count >= 3:
                        ratio = recent_count / max(older_count, 1)
                        if ratio > 2 and older_count < recent_count:
                            emerging.append((concept, recent_count, ratio))
                        elif ratio < 0.5 and older_count > recent_count:
                            declining.append((concept, older_count, ratio))
                
                trend_analysis = {'emerging': emerging[:5], 'declining': declining[:5]}
            
            # Format sophisticated results
            result = "**🧠 Advanced Topic & Trend Analysis**\n\n"
            if channel_filter:
                result += f"**Channel: #{channel_filter}**\n\n"
            
            result += f"**📊 Analyzed {len([d for d in docs if d])} substantial messages**\n\n"
            
            # Technical Innovation Topics
            if technical_terms:
                result += "**🔬 Technical Innovation Topics:**\n"
                for term, count in technical_terms.most_common(10):
                    if count >= 3:
                        result += f"• {term.upper()} ({count} discussions)\n"
                result += "\n"
            
            # Business & Strategy Themes
            if business_concepts:
                result += "**📈 Business & Strategy Themes:**\n"
                for concept, count in business_concepts.most_common(8):
                    if count >= 2:
                        result += f"• {concept.title()} ({count} mentions)\n"
                result += "\n"
            
            # Innovation Indicators
            if innovation_indicators:
                result += "**� Innovation & Future Focus:**\n"
                for pattern, count in innovation_indicators.most_common(6):
                    if count >= 2:
                        result += f"• {pattern.title()} ({count} mentions)\n"
                result += "\n"
            
            # Complex Multi-word Concepts (NEW!)
            if complex_phrases:
                result += "**🎯 Complex Discussion Topics:**\n"
                for phrase, count in complex_phrases.most_common(12):
                    if count >= 2 and len(phrase.split()) >= 3:
                        result += f"• {phrase.title()} ({count} discussions)\n"
                result += "\n"
            
            # Semantic Concept Categories (NEW!)
            for category, concepts in semantic_concepts.items():
                if len(concepts) >= 3:
                    concept_counts = Counter(concepts)
                    top_concepts = concept_counts.most_common(6)
                    if top_concepts and any(count >= 2 for _, count in top_concepts):
                        result += f"**🔍 {category}:**\n"
                        for concept, count in top_concepts:
                            if count >= 2:
                                result += f"• {concept.title()} ({count}x)\n"
                        result += "\n"

            # Conversation Flow Analysis
            if conversation_flows:
                result += "**�️ Key Conversation Flows:**\n"
                flow_summary = defaultdict(int)
                for flow in conversation_flows:
                    concept_key = ' + '.join(flow['concepts'][:2])
                    flow_summary[concept_key] += 1
                
                for flow, count in sorted(flow_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                    if count >= 2:
                        result += f"• {flow.title()} ({count} collaborative discussions)\n"
                result += "\n"
            
            # Trend Analysis
            if trend_analysis:
                if trend_analysis['emerging']:
                    result += "**📈 Emerging Trends:**\n"
                    for concept, count, ratio in trend_analysis['emerging']:
                        result += f"• {concept.title()} ({count} mentions, {ratio:.1f}x growth)\n"
                    result += "\n"
                
                if trend_analysis['declining']:
                    result += "**� Declining Topics:**\n"
                    for concept, count, ratio in trend_analysis['declining']:
                        result += f"• {concept.title()} (was {count} mentions)\n"
                    result += "\n"
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error analyzing topics: {str(e)}"