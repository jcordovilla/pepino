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
            async with self.pool.cursor() as cursor:
                await cursor.execute("""
                    SELECT DISTINCT channel_name 
                    FROM messages 
                    WHERE channel_name NOT LIKE '%test%'
                    AND channel_name NOT LIKE '%playground%'
                    ORDER BY channel_name
                """)
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
                AND author_name != 'sesh'
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
            AND author_name != 'sesh'
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

    async def update_temporal_stats(self, args: dict = None) -> str:
        """Update temporal activity statistics - async wrapper"""
        try:
            await self.initialize()
            
            # Get temporal statistics
            async with self.pool.execute(f"""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    strftime('%w', timestamp) as day,
                    COUNT(*) as message_count
                FROM messages
                WHERE {self.base_filter}
                GROUP BY hour, day
                ORDER BY day, hour
            """) as cursor:
                stats = await cursor.fetchall()
            
            if not stats:
                return "No temporal statistics available"
            
            # Process statistics
            hourly_stats = {}
            daily_stats = {}
            
            for stat in stats:
                hour = int(stat[0])
                day = int(stat[1])
                count = stat[2]
                
                if hour not in hourly_stats:
                    hourly_stats[hour] = 0
                hourly_stats[hour] += count
                
                if day not in daily_stats:
                    daily_stats[day] = 0
                daily_stats[day] += count
            
            # Format results
            result = "**Temporal Activity Analysis**\n\n"
            
            result += "**Activity by Hour:**\n"
            for hour in sorted(hourly_stats.keys()):
                result += f"{hour:02d}:00-{hour+1:02d}:00: {hourly_stats[hour]} messages\n"
            
            result += "\n**Activity by Day:**\n"
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            for day in range(7):
                result += f"{days[day]}: {daily_stats.get(day, 0)} messages\n"
            
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

    async def generate_channel_activity_chart(self, channel_name: str) -> str:
        """Generate a chart showing channel activity over time"""
        try:
            # Get message counts by day
            cursor = self.conn.cursor()
            cursor.execute(f"""
                WITH filtered_messages AS (
                    SELECT *
                    FROM messages
                    WHERE channel_name = ? AND {self.base_filter}
                )
                SELECT 
                    date(timestamp) as date,
                    COUNT(*) as message_count
                FROM filtered_messages
                GROUP BY date
                ORDER BY date
            """, (channel_name,))
            
            data = cursor.fetchall()
            if not data:
                return None
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            # Convert dates to datetime objects
            dates = [datetime.strptime(row['date'], '%Y-%m-%d') for row in data]
            counts = [row['message_count'] for row in data]
            
            plt.plot(dates, counts, marker='o')
            plt.title(f'Message Activity in #{channel_name}')
            plt.xlabel('Date')
            plt.ylabel('Message Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Save the plot to a temporary file
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, f'channel_activity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return file_path
            
        except Exception as e:
            print(f"Error generating channel activity chart: {e}")
            return None

    async def generate_user_activity_chart(self, user_name: str, temporal_stats: list) -> str:
        """Generate a chart showing user activity over time"""
        try:
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Extract hours and counts
            hours = [int(stat[0]) for stat in temporal_stats]
            counts = [stat[1] for stat in temporal_stats]
            
            # Create the plot
            plt.bar(hours, counts)
            plt.title(f'Message Activity for {user_name}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Message Count')
            plt.xticks(range(24))
            plt.grid(True, alpha=0.3)
            
            # Save to temporary file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'temp/user_activity_{timestamp}.png'
            plt.savefig(filename)
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Error generating user activity chart: {str(e)}")
            return None

    async def run_analysis(self, task: str, args: dict = None) -> str:
        """Run the specified analysis task"""
        try:
            # Convert message_id to int if present
            if args and 'message_id' in args:
                try:
                    args['message_id'] = int(args['message_id'])
                except ValueError:
                    return "Error: message_id must be a number"
            
            # Get the analysis method
            analysis_method = self.analysis_tasks.get(task)
            if not analysis_method:
                return f"Unknown analysis task: {task}"
            
            # Run the analysis
            return await analysis_method(args)
            
        except Exception as e:
            return f"Error running analysis: {str(e)}"

    async def get_available_analyses(self) -> str:
        """Return a formatted list of available analyses"""
        result = "**Available Analyses:**\n\n"
        for task, func in self.analysis_tasks.items():
            result += f"• `{task}`: {func.__doc__ or 'No description available'}\n"
        return result

    async def analyze_topics(self, args: dict = None) -> str:
        """Analyze topics in messages using LDA"""
        try:
            await self.ensure_model_loaded()
            
            # Get channel filter if provided
            channel_filter = None
            if args and "channel_name" in args:
                channel_filter = args["channel_name"]
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
                    score = fuzz.ratio(channel.lower(), channel_filter.lower())
                    if score > best_score and score > 80:  # Require 80% similarity
                        best_score = score
                        best_match = channel
                
                if not best_match:
                    return f"Channel '{channel_filter}' not found. Available channels: {', '.join(channels)}"
                
                channel_filter = best_match
            
            # Get messages
            cursor = self.conn.cursor()
            if channel_filter:
                cursor.execute(f"""
                    WITH filtered_messages AS (
                        SELECT content, author_id, channel_name, timestamp 
                        FROM messages 
                        WHERE channel_name = ? AND {self.base_filter}
                    )
                    SELECT * FROM filtered_messages
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """, (channel_filter,))
            else:
                cursor.execute(f"""
                    WITH filtered_messages AS (
                        SELECT content, author_id, channel_name, timestamp 
                        FROM messages 
                        WHERE {self.base_filter}
                    )
                    SELECT * FROM filtered_messages
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """)
            
            messages = cursor.fetchall()
            if not messages:
                return "No messages found to analyze."
            
            # Get user display names
            user_names = {}
            for msg in messages:
                if msg[1] not in user_names:
                    cursor.execute("""
                        SELECT display_name 
                        FROM users 
                        WHERE user_id = ?
                    """, (msg[1],))
                    result = cursor.fetchone()
                    user_names[msg[1]] = result[0] if result else msg[1]
            
            # Prepare text for analysis
            texts = []
            for msg in messages:
                if msg[0] and len(msg[0].split()) > 3:  # Only use messages with more than 3 words
                    texts.append(msg[0])
            
            if not texts:
                return "No suitable messages found for topic analysis."
            
            # Get embeddings
            embeddings = []
            for text in texts:
                try:
                    embedding = await self.get_embedding(text)
                    if embedding is not None:
                        embeddings.append(embedding)
                except Exception as e:
                    print(f"Error getting embedding: {e}")
                    continue
            
            if not embeddings:
                return "Failed to generate embeddings for messages."
            
            # Convert to numpy array
            X = np.array(embeddings)
            
            # Perform LDA
            n_topics = min(5, len(texts))  # Number of topics (up to 5)
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            
            # Fit LDA
            lda.fit(X)
            
            # Get top words for each topic
            feature_names = [f"word_{i}" for i in range(X.shape[1])]
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
            
            # Get topic distribution for each message
            topic_dist = lda.transform(X)
            
            # Find most representative messages for each topic
            representative_messages = []
            for topic_idx in range(n_topics):
                # Get messages with highest probability for this topic
                topic_probs = topic_dist[:, topic_idx]
                top_msg_idx = topic_probs.argsort()[-3:][::-1]  # Top 3 messages
                
                for idx in top_msg_idx:
                    msg = messages[idx]
                    prob = topic_probs[idx]
                    if prob > 0.3:  # Only include if probability > 30%
                        representative_messages.append({
                            'topic': topic_idx + 1,
                            'content': msg[0],
                            'author': user_names[msg[1]],
                            'channel': msg[2],
                            'timestamp': self.format_timestamp(msg[3]),
                            'probability': f"{prob:.2%}"
                        })
            
            # Format results
            result = f"**Topic Analysis{' for ' + channel_filter if channel_filter else ''}**\n\n"
            
            # Add topics
            result += "**Identified Topics:**\n"
            for topic in topics:
                result += f"- {topic}\n"
            
            # Add representative messages
            if representative_messages:
                result += "\n**Representative Messages:**\n"
                for msg in representative_messages:
                    result += f"\nTopic {msg['topic']} ({msg['probability']}):\n"
                    result += f"From: {msg['author']} in {msg['channel']} at {msg['timestamp']}\n"
                    result += f"Message: {msg['content']}\n"
            
            return result
            
        except Exception as e:
            print(f"Error in topic analysis: {e}")
            return f"Error analyzing topics: {str(e)}"

    async def get_user_insights(self, user_id: str) -> str:
        """Get insights for a specific user with fuzzy matching"""
        try:
            await self.initialize()
            print(f"Getting insights for user: {user_id}")
            
            # Get all available users (both username and display name) for fuzzy matching
            async with self.pool.execute("""
                SELECT DISTINCT author_id, author_name, author_display_name
                FROM messages 
                WHERE author_name IS NOT NULL 
                AND author_name != 'sesh'
                ORDER BY author_name
            """) as cursor:
                all_users = await cursor.fetchall()
            
            if not all_users:
                return "No users found in the database."
            
            # Try exact matches first
            user = None
            
            # Try exact match on username
            for u in all_users:
                if u[1].lower() == user_id.lower():
                    user = u
                    print(f"Found exact username match: {user_id}")
                    break
            
            # Try exact match on display name if no username match
            if not user:
                for u in all_users:
                    display_name = u[2] if len(u) > 2 and u[2] else ""
                    if display_name and display_name.lower() == user_id.lower():
                        user = u
                        print(f"Found exact display name match: {user_id}")
                        break
            
            # If no exact match, use fuzzy matching on both username and display name
            if not user:
                from fuzzywuzzy import fuzz
                
                # Find the best fuzzy match across both usernames and display names
                best_match = None
                best_ratio = 0
                
                for u in all_users:
                    # Check username fuzzy match
                    username_ratio = fuzz.ratio(user_id.lower(), u[1].lower())
                    if username_ratio > best_ratio:
                        best_ratio = username_ratio
                        best_match = u
                    
                    # Check display name fuzzy match
                    display_name = u[2] if len(u) > 2 and u[2] else ""
                    if display_name:
                        display_ratio = fuzz.ratio(user_id.lower(), display_name.lower())
                        if display_ratio > best_ratio:
                            best_ratio = display_ratio
                            best_match = u
                
                # Use fuzzy match if similarity is above threshold (70%)
                if best_ratio >= 70:
                    user = best_match
                    print(f"Fuzzy match found: '{user_id}' -> '{user[1]}' (similarity: {best_ratio}%)")
            
            if not user:
                # No good match found, show suggestions
                from fuzzywuzzy import process
                
                # Create list of all searchable names (both usernames and display names)
                searchable_names = []
                for u in all_users:
                    searchable_names.append(u[1])  # username
                    display_name = u[2] if len(u) > 2 and u[2] else ""
                    if display_name and display_name != u[1]:
                        searchable_names.append(display_name)  # display name
                
                suggestions = process.extract(user_id, searchable_names, limit=5)
                suggestion_text = "\n".join([f"• {name} (similarity: {score}%)" for name, score in suggestions])
                return f"User '{user_id}' not found. Did you mean one of these?\n{suggestion_text}"
            
            actual_user_id = user[0]
            user_name = user[1]
            print(f"Found user: {user_name} (ID: {actual_user_id})")
            
            # Get user statistics
            async with self.pool.execute("""
                SELECT 
                    COUNT(*) as message_count,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM messages 
                WHERE author_id = ?
                AND channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                AND author_name != 'sesh'
            """, (actual_user_id,)) as cursor:
                stats = await cursor.fetchone()
            
            if not stats or stats[0] == 0:
                return f"No messages found for user {user_name}"
            
            # Get channel activity
            async with self.pool.execute("""
                SELECT 
                    channel_name,
                    COUNT(*) as message_count
                FROM messages 
                WHERE author_id = ?
                AND channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                AND author_name != 'sesh'
                GROUP BY channel_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (actual_user_id,)) as cursor:
                channel_stats = await cursor.fetchall()
            
            # Get recent messages for word frequency
            async with self.pool.execute("""
                SELECT content
                FROM messages 
                WHERE author_id = ?
                AND channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                AND author_name != 'sesh'
                AND content IS NOT NULL AND content != ''
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (actual_user_id,)) as cursor:
                messages = await cursor.fetchall()
            
            # Process word frequency manually
            word_freq = {}
            for msg in messages:
                content = msg[0] if msg else ""
                if content:
                    # Simple word processing
                    words = content.lower().split()
                    for word in words:
                        word = word.strip('.,!?";:()[]{}')
                        if len(word) > 3 and word.isalpha():
                            word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 10 words
            word_stats = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Format the results
            result = f"**User Analysis for {user_name}**\n\n"
            
            # Basic stats
            result += f"**Message Statistics:**\n"
            result += f"• Total Messages: {stats[0]}\n"
            result += f"• First Message: {stats[1]}\n"
            result += f"• Last Message: {stats[2]}\n\n"
            
            # Channel activity
            result += f"**Top Channels:**\n"
            for chan in channel_stats:
                result += f"• #{chan[0]}: {chan[1]} messages\n"
            result += "\n"
            
            # Word frequency
            result += f"**Most Used Words:**\n"
            for word in word_stats:
                result += f"• {word[0]}: {word[1]} times\n"
            
            return result
            
        except Exception as e:
            print(f"Error in get_user_insights: {str(e)}")
            return f"Error getting user insights: {str(e)}"