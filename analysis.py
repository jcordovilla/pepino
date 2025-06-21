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
        """Cleanup method to handle connection cleanup safely"""
        try:
            # For DiscordBotAnalyzer, we use async connections (self.pool)
            # The pool should be closed via the async close() method
            # We don't have a self.conn attribute, so we override the base class __del__
            pass
        except:
            # Ignore any errors during cleanup
            pass

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
                    COALESCE(author_display_name, author_name) as display_name,
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
                SELECT 
                    CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                    CAST(strftime('%w', timestamp) AS INTEGER) as day_of_week,
                    DATE(timestamp) as date,
                    COUNT(*) as message_count
                FROM messages
                WHERE {self.base_filter}
                AND timestamp IS NOT NULL
                GROUP BY hour, day_of_week, date
                ORDER BY date, hour
            """)
            
            stats = cursor.fetchall()
            if not stats:
                return "No temporal statistics available"
            
            # Process statistics
            hourly_stats = {}
            daily_stats = {}
            weekly_stats = {}
            
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            for stat in stats:
                hour = stat[0]
                day_of_week = stat[1]
                message_count = stat[3]
                
                # Hourly aggregation
                if hour not in hourly_stats:
                    hourly_stats[hour] = 0
                hourly_stats[hour] += message_count
                
                # Daily aggregation
                day_name = day_names[day_of_week]
                if day_name not in daily_stats:
                    daily_stats[day_name] = 0
                daily_stats[day_name] += message_count
                
                # Weekly pattern
                if day_of_week not in weekly_stats:
                    weekly_stats[day_of_week] = 0
                weekly_stats[day_of_week] += message_count
            
            # Format results
            result = "**📊 Temporal Activity Analysis**\n\n"
            
            # Peak hours
            if hourly_stats:
                sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1], reverse=True)
                result += "**🕐 Peak Activity Hours:**\n"
                for hour, count in sorted_hours[:5]:
                    time_str = f"{hour:02d}:00-{hour:02d}:59"
                    result += f"• {time_str}: {count:,} messages\n"
                result += "\n"
            
            # Daily distribution
            if daily_stats:
                result += "**📅 Activity by Day of Week:**\n"
                for day in day_names:
                    if day in daily_stats:
                        count = daily_stats[day]
                        result += f"• {day}: {count:,} messages\n"
                result += "\n"
            
            # Activity patterns
            if hourly_stats:
                result += "**⏰ Activity Patterns:**\n"
                
                # Morning (6-11)
                morning = sum(hourly_stats.get(h, 0) for h in range(6, 12))
                # Afternoon (12-17)
                afternoon = sum(hourly_stats.get(h, 0) for h in range(12, 18))
                # Evening (18-23)
                evening = sum(hourly_stats.get(h, 0) for h in range(18, 24))
                # Night (0-5)
                night = sum(hourly_stats.get(h, 0) for h in range(0, 6))
                
                total = morning + afternoon + evening + night
                if total > 0:
                    result += f"• Morning (06-11): {morning:,} messages ({morning/total*100:.1f}%)\n"
                    result += f"• Afternoon (12-17): {afternoon:,} messages ({afternoon/total*100:.1f}%)\n"
                    result += f"• Evening (18-23): {evening:,} messages ({evening/total*100:.1f}%)\n"
                    result += f"• Night (00-05): {night:,} messages ({night/total*100:.1f}%)\n"
            
            return result
            
        except Exception as e:
            return f"Error updating temporal statistics: {str(e)}"

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
        """Get list of available channels in the database (ordered by activity)"""
        try:
            await self.initialize()
            async with self.pool.execute("""
                SELECT 
                    channel_name,
                    COUNT(*) as message_count
                FROM messages 
                WHERE channel_name IS NOT NULL 
                AND channel_name != ''
                AND channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                AND channel_name NOT LIKE '%pg%'
                GROUP BY channel_name
                ORDER BY message_count DESC, channel_name
            """) as cursor:
                channels = await cursor.fetchall()
                return [channel[0] for channel in channels if channel[0]]
        except Exception as e:
            print(f"Error getting available channels: {str(e)}")
            return []

    async def get_available_users(self) -> List[str]:
        """Get list of available users in the database (display names preferred)"""
        try:
            await self.initialize()
            async with self.pool.execute("""
                SELECT DISTINCT 
                    COALESCE(author_display_name, author_name) as display_name,
                    COUNT(*) as message_count
                FROM messages 
                WHERE author_name IS NOT NULL 
                AND author_id != 'sesh'
                AND author_id != '1362434210895364327'
                AND author_name != 'sesh'
                AND LOWER(author_name) != 'pepe'
                AND LOWER(author_name) != 'pepino'
                AND COALESCE(author_display_name, author_name) IS NOT NULL
                GROUP BY COALESCE(author_display_name, author_name)
                ORDER BY message_count DESC, display_name
            """) as cursor:
                users = await cursor.fetchall()
                return [user[0] for user in users if user[0]]
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
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt')
        try:
            import nltk
            nltk.data.find('corpora/stopwords')
        except LookupError:
            import nltk
            nltk.download('stopwords')
        
        # Create temp directory if it doesn't exist
        import os
        os.makedirs('temp', exist_ok=True)

    async def initialize(self):
        """Initialize the database connection pool"""
        if self.pool is None:
            try:
                import aiosqlite
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

    def format_timestamp(self, timestamp: str) -> str:
        """Format a timestamp string to a readable format"""
        try:
            from datetime import datetime
            # Handle different timestamp formats
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M')
        except:
            return timestamp

    async def get_channel_name_mapping(self, bot_guilds=None) -> Dict[str, str]:
        """Create a mapping of old database channel names to current Discord channel names"""
        try:
            await self.initialize()
            
            # Get all channel names from database
            async with self.pool.execute("""
                SELECT DISTINCT channel_name
                FROM messages 
                WHERE channel_name IS NOT NULL 
                AND channel_name != ''
                ORDER BY channel_name
            """) as cursor:
                db_channels = await cursor.fetchall()
                db_channel_names = [ch[0] for ch in db_channels if ch[0]]
            
            # If bot guilds are provided, get current Discord channel names
            current_channels = {}
            if bot_guilds:
                for guild in bot_guilds:
                    for channel in guild.channels:
                        if hasattr(channel, 'name'):
                            current_channels[channel.name] = channel.name
                            # Also map with common prefixes that might be used
                            current_channels[f"#{channel.name}"] = channel.name
                            current_channels[f"🏛{channel.name}"] = channel.name
                            current_channels[f"🦾{channel.name}"] = channel.name
                            current_channels[f"🏘{channel.name}"] = channel.name
            
            # Create mapping: old_name -> new_name
            channel_mapping = {}
            for db_name in db_channel_names:
                # First, try exact match
                if db_name in current_channels:
                    channel_mapping[db_name] = current_channels[db_name]
                    continue
                
                # Try without emoji prefixes
                clean_db_name = db_name
                for prefix in ['🏛', '🦾', '🏘', '#']:
                    if clean_db_name.startswith(prefix):
                        clean_db_name = clean_db_name[len(prefix):]
                        break
                
                # Look for matches in current channels
                best_match = None
                for current_name in current_channels.values():
                    if clean_db_name == current_name:
                        best_match = current_name
                        break
                    elif clean_db_name.lower() == current_name.lower():
                        best_match = current_name
                        break
                    elif clean_db_name in current_name or current_name in clean_db_name:
                        best_match = current_name
                
                if best_match:
                    channel_mapping[db_name] = best_match
                else:
                    # Keep original name if no match found
                    channel_mapping[db_name] = db_name
            
            return channel_mapping
            
        except Exception as e:
            print(f"Error creating channel name mapping: {str(e)}")
            return {}

    async def get_available_channels_with_mapping(self, bot_guilds=None) -> List[str]:
        """Get available channels with current Discord names when possible"""
        try:
            await self.initialize()
            
            # Get channel mapping
            channel_mapping = await self.get_channel_name_mapping(bot_guilds)
            
            # Get channels ordered by activity from database
            async with self.pool.execute("""
                SELECT 
                    channel_name,
                    COUNT(*) as message_count
                FROM messages 
                WHERE channel_name IS NOT NULL 
                AND channel_name != ''
                AND channel_name NOT LIKE '%test%'
                AND channel_name NOT LIKE '%playground%'
                AND channel_name NOT LIKE '%pg%'
                GROUP BY channel_name
                ORDER BY message_count DESC, channel_name
            """) as cursor:
                channels = await cursor.fetchall()
            
            # Map to current names and deduplicate
            mapped_channels = []
            seen = set()
            
            for channel, count in channels:
                if channel:
                    # Use mapped name if available, otherwise original
                    current_name = channel_mapping.get(channel, channel)
                    if current_name not in seen:
                        mapped_channels.append(current_name)
                        seen.add(current_name)
            
            return mapped_channels
            
        except Exception as e:
            print(f"Error getting available channels with mapping: {str(e)}")
            # Fallback to original method
            return await self.get_available_channels()

    async def analyze_topics_spacy(self, args: dict = None) -> str:
        """Simplified topic analysis with clean, actionable insights"""
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
            
            # Get messages with minimum character filter
            if channel_filter:
                async with self.pool.execute(f"""
                    SELECT content, timestamp, author_name
                    FROM messages 
                    WHERE channel_name = ? AND {self.base_filter}
                    AND content IS NOT NULL AND content != ''
                    AND LENGTH(content) > 50
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
                    AND LENGTH(content) > 50
                    ORDER BY timestamp DESC 
                    LIMIT 800
                """) as cursor:
                    messages = await cursor.fetchall()
            
            if not messages:
                return "No messages found for topic analysis."

            # Get comprehensive statistics for overview section
            if channel_filter:
                # Get channel-specific stats
                async with self.pool.execute(f"""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                        COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                        COUNT(CASE WHEN has_reactions = 1 THEN 1 END) as posts_with_reactions,
                        COUNT(DISTINCT author_id) as active_contributors,
                        AVG(LENGTH(content)) as avg_msg_length,
                        MIN(timestamp) as earliest_msg,
                        MAX(timestamp) as latest_msg
                    FROM messages
                    WHERE channel_name = ? AND {self.base_filter}
                    AND content IS NOT NULL AND LENGTH(content) > 50
                """, (channel_filter,)) as cursor:
                    stats = await cursor.fetchone()
                
                # Get total channel members
                async with self.pool.execute(f"""
                    SELECT COUNT(DISTINCT user_id) as total_members
                    FROM channel_members
                    WHERE channel_name = ?
                """, (channel_filter,)) as cursor:
                    member_result = await cursor.fetchone()
                    total_members = member_result[0] if member_result else 0
            else:
                # Global stats
                async with self.pool.execute(f"""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                        COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                        COUNT(CASE WHEN has_reactions = 1 THEN 1 END) as posts_with_reactions,
                        COUNT(DISTINCT author_id) as active_contributors,
                        AVG(LENGTH(content)) as avg_msg_length,
                        MIN(timestamp) as earliest_msg,
                        MAX(timestamp) as latest_msg
                    FROM messages
                    WHERE {self.base_filter}
                    AND content IS NOT NULL AND LENGTH(content) > 50
                """) as cursor:
                    stats = await cursor.fetchone()
                total_members = 0
            
            # Advanced text cleaning
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
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            # Process messages for topics
            technical_terms = Counter()
            business_concepts = Counter()
            key_discussions = Counter()
            temporal_trends = defaultdict(list)
            
            # Define focused patterns
            tech_patterns = [
                r'\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework|metrics)\b'
            ]
            
            business_patterns = [
                r'\b(team|collaboration|workflow|process|efficiency|optimization|integration|strategy|growth|solution|deployment)\b'
            ]
            
            for msg in messages:
                content, timestamp, author = msg[0], msg[1], msg[2]
                cleaned_content = clean_content(content)
                
                if len(cleaned_content.split()) < 5:
                    continue
                
                try:
                    doc = nlp(cleaned_content)
                    
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
                    
                    # Extract multi-word discussion themes
                    for chunk in doc.noun_chunks:
                        if (len(chunk.text.split()) >= 3 and 
                            len(chunk.text) > 15 and
                            chunk.text.lower() not in ['the conversational leaders', 'the community coordinators']):
                            
                            clean_chunk = re.sub(r'^(the|a|an)\s+', '', chunk.text.lower()).strip()
                            if len(clean_chunk.split()) >= 2:
                                key_discussions[clean_chunk.title()] += 1
                    
                    # Time-based trends
                    try:
                        msg_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        week_key = msg_date.strftime('%Y-W%U')
                        
                        # Add significant concepts to trends
                        main_concepts = [token.lemma_.lower() for token in doc 
                                       if token.pos_ in ['NOUN', 'PROPN'] and 
                                       not token.is_stop and len(token.text) > 3]
                        temporal_trends[week_key].extend(main_concepts[:3])
                        
                    except:
                        pass
                
                except Exception:
                    continue
            
            # Calculate trend analysis
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
            
            # Format simplified results
            result = f"🧠 Topic Analysis: #{channel_filter}\n\n"
            
            # Enhanced overview section
            if stats:
                total_msgs = stats[0] if stats[0] else 0
                total_replies = stats[1] if stats[1] else 0
                original_posts = stats[2] if stats[2] else 1
                posts_with_reactions = stats[3] if stats[3] else 0
                active_contributors = stats[4] if stats[4] else 0
                avg_length = int(stats[5]) if stats[5] else 0
                earliest = stats[6] if stats[6] else None
                latest = stats[7] if stats[7] else None
                
                replies_per_post = total_replies / original_posts if original_posts > 0 else 0
                reaction_rate = (posts_with_reactions / total_msgs * 100) if total_msgs > 0 else 0
                participation_rate = (active_contributors / total_members * 100) if total_members > 0 else 0
                
                result += "📊 Channel Overview:\n"
                result += f"• {total_msgs} substantial messages analyzed (50+ character minimum)\n"
                result += f"• {replies_per_post:.2f} average replies per post\n"
                result += f"• {reaction_rate:.1f}% reaction rate (messages with reactions)\n"
                if total_members > 0:
                    result += f"• {total_members} total members, {active_contributors} active contributors ({participation_rate:.1f}% participation)\n"
                result += f"• Average message length: {avg_length} characters\n"
                
                if earliest and latest:
                    try:
                        start_date = datetime.fromisoformat(earliest.replace('Z', '+00:00')).strftime('%b %Y')
                        end_date = datetime.fromisoformat(latest.replace('Z', '+00:00')).strftime('%b %Y')
                        if start_date == end_date:
                            result += f"• Activity timeframe: {start_date}\n"
                        else:
                            result += f"• Activity timeframe: {start_date} - {end_date}\n"
                    except:
                        pass
                
                result += "\n"
            
            # Technical Topics
            if technical_terms:
                result += "🔧 Technical Topics:\n"
                for term, count in technical_terms.most_common(8):
                    if count >= 3:
                        result += f"• {term} ({count} discussions)\n"
                result += "\n"
            
            # Business Topics  
            if business_concepts:
                result += "� Business Topics:\n"
                for concept, count in business_concepts.most_common(6):
                    if count >= 3:
                        result += f"• {concept.title()} ({count} mentions)\n"
                result += "\n"
            
            # Key Discussion Threads
            if key_discussions:
                result += "🎯 Key Discussion Threads:\n"
                for discussion, count in key_discussions.most_common(8):
                    if count >= 2:
                        result += f"• {discussion} ({count} discussions)\n"
                result += "\n"
            
            # Trending Topics
            if emerging_topics:
                result += "📈 Trending: "
                trending_items = []
                for concept, count, ratio in emerging_topics[:3]:
                    trending_items.append(f"{concept.title()} ({ratio:.1f}x growth)")
                result += ", ".join(trending_items) + "\n"
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error analyzing topics: {str(e)}"

    async def extract_concepts_from_content(self, messages) -> List[str]:
        """Extract most relevant and frequent topics from user messages"""
        try:
            from collections import Counter
            import re
            
            # Combine all message content
            all_text = " ".join([msg[0] for msg in messages if msg[0] and len(msg[0]) > 15])
            if not all_text.strip():
                return []
            
            # Clean text but preserve important terms
            text = all_text.lower()
            # Remove URLs but keep other content
            text = re.sub(r'https?://[^\s]+', '', text)
            text = re.sub(r'<@[!&]?\d+>', '', text)  # Remove Discord mentions
            
            concepts = []
            
            # 1. Extract meaningful compound phrases (2-3 words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
            # Generate 2-word phrases
            two_word_phrases = []
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 8:  # Reasonable length
                    two_word_phrases.append(phrase)
            
            # Generate 3-word phrases  
            three_word_phrases = []
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 12:  # Reasonable length
                    three_word_phrases.append(phrase)
            
            # Count phrase frequencies
            phrase_counter = Counter(two_word_phrases + three_word_phrases)
            
            # Add frequent phrases
            for phrase, count in phrase_counter.most_common(30):
                if count >= 2:  # Appears at least twice
                    concepts.append(phrase)
            
            # 2. Extract important single words that are content-rich
            important_single_words = []
            for word in words:
                if (len(word) >= 5 and 
                    word not in ['discord', 'everyone', 'message', 'channel', 'meeting', 'group', 'thanks', 'please', 'hello', 'morning', 'evening', 'today', 'think', 'really', 'great', 'would', 'could', 'should', 'about', 'after', 'before', 'which', 'where', 'there', 'other', 'first', 'through', 'welcome', 'question', 'discussion', 'conversation']):
                    important_single_words.append(word)
            
            # Count single words and add frequent ones
            word_counter = Counter(important_single_words)
            for word, count in word_counter.most_common(20):
                if count >= 3:  # Appears at least 3 times
                    concepts.append(word)
            
            # 3. Domain-specific terms that are actually relevant
            domain_terms = []
            
            # Look for actual topics being discussed
            activity_patterns = [
                r'\b(?:content curation|content curator|conversational leader|buddy group|cohort|workshop|session|sync|onboarding|mentorship|coordination|feedback|analytics|automation|leadership)\b',
                r'\b(?:agent ops|discord tool|channel analysis|reaction counter|watermarking|genai|social media|linkedin|google|anthropic|chatgpt)\b',
                r'\b(?:net arch|forum discussion|group chat|zoom meeting|recording|guidelines|submission|deployment|implementation)\b'
            ]
            
            for pattern in activity_patterns:
                matches = re.findall(pattern, text)
                domain_terms.extend(matches)
            
            concepts.extend(domain_terms)
            
            # Count all concepts and filter
            concept_counter = Counter(concepts)
            
            # Final filtering for most relevant topics
            final_concepts = []
            for concept, count in concept_counter.most_common(25):
                # Prioritize actual discussion topics over generic terms
                concept_clean = concept.strip()
                if (len(concept_clean) >= 5 and
                    count >= 2 and
                    concept_clean not in ['meeting today', 'thank you', 'good morning', 'everyone here', 'really good', 'really think', 'would like', 'think this', 'this really', 'really great', 'thank everyone', 'everyone thank', 'great work', 'good work', 'work with', 'working with']):
                    final_concepts.append(concept_clean)
                    
                if len(final_concepts) >= 5:  # Limit to top 5
                    break
            
            return final_concepts
            
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            return []

    async def update_user_statistics(self, args: dict = None) -> str:
        """Enhanced user activity statistics with concept analysis - overrides base class method"""
        try:
            await self.initialize()
            
            # Get user statistics with most active channel
            async with self.pool.execute(f"""
                WITH user_stats AS (
                    SELECT 
                        author_id,
                        COALESCE(author_display_name, author_name) as display_name,
                        COUNT(*) as message_count,
                        COUNT(DISTINCT channel_name) as channels_active,
                        AVG(LENGTH(content)) as avg_message_length,
                        MIN(DATE(timestamp)) as first_message_date,
                        MAX(DATE(timestamp)) as last_message_date
                    FROM messages
                    WHERE {self.base_filter}
                    GROUP BY author_id, author_display_name, author_name
                ),
                user_top_channels AS (
                    SELECT 
                        author_id,
                        channel_name as top_channel,
                        COUNT(*) as channel_messages,
                        ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY COUNT(*) DESC) as rn
                    FROM messages
                    WHERE {self.base_filter}
                    GROUP BY author_id, channel_name
                )
                SELECT 
                    u.author_id,
                    u.display_name,
                    u.message_count,
                    u.channels_active,
                    u.avg_message_length,
                    u.first_message_date,
                    u.last_message_date,
                    c.top_channel,
                    c.channel_messages
                FROM user_stats u
                LEFT JOIN user_top_channels c ON u.author_id = c.author_id AND c.rn = 1
                ORDER BY u.message_count DESC
                LIMIT 10
            """) as cursor:
                users = await cursor.fetchall()
            
            if not users:
                return "No user statistics available"
            
            # Format results with enhanced information
            result = "**📊 Top 10 User Activity Statistics**\n\n"
            
            for i, user in enumerate(users, 1):
                display_name = user[1] if user[1] else "Unknown"
                message_count = user[2]
                channels_active = user[3]
                avg_length = user[4] if user[4] else 0
                first_date = user[5] if user[5] else "Unknown"
                last_date = user[6] if user[6] else "Unknown"
                top_channel = user[7] if user[7] else "Unknown"
                channel_messages = user[8] if user[8] else 0
                
                # Format activity period as simple date range
                if first_date != "Unknown" and last_date != "Unknown":
                    if first_date == last_date:
                        activity_period = first_date
                    else:
                        activity_period = f"{first_date} → {last_date}"
                else:
                    activity_period = "Unknown"
                
                result += f"**{i}. {display_name}**\n"
                result += f"• Messages: {message_count:,} • Channels: {channels_active} • Avg Length: {avg_length:.0f} chars\n"
                result += f"• Most Active: #{top_channel} ({channel_messages} messages)\n"
                result += f"• Active Period: {activity_period}\n"
                
                # Get user's main topics with improved concept extraction
                try:
                    async with self.pool.execute(f"""
                        SELECT content
                        FROM messages
                        WHERE author_id = ? AND {self.base_filter}
                        AND content IS NOT NULL 
                        AND LENGTH(content) > 50
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """, (user[0],)) as cursor:
                        user_messages = await cursor.fetchall()
                    
                    if user_messages:
                        user_concepts = await self.extract_concepts_from_content(user_messages)
                        if user_concepts:
                            # Filter for meaningful multi-word concepts
                            meaningful_concepts = [concept for concept in user_concepts 
                                                 if len(concept.split()) >= 2 or len(concept) > 6]
                            if meaningful_concepts:
                                result += f"• Main Topics: {', '.join(meaningful_concepts[:3])}\n"
                            elif user_concepts:
                                result += f"• Main Topics: {', '.join(user_concepts[:3])}\n"
                
                except Exception as e:
                    print(f"Error getting concepts for user {display_name}: {e}")
                
                result += "\n"
            
            # Add overall statistics
            total_messages = sum(user[2] for user in users)
            avg_channels_per_user = sum(user[3] for user in users) / len(users) if users else 0
            
            result += f"**📈 Summary:**\n"
            result += f"• Top 10 users: {total_messages:,} messages • Avg channels per user: {avg_channels_per_user:.1f}\n"
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error updating user statistics: {str(e)}"

    async def resolve_channel_name(self, user_input: str, bot_guilds=None) -> str:
        """Resolve user input to the actual database channel name"""
        try:
            await self.initialize()
            
            # Get channel mapping (current -> old)
            channel_mapping = await self.get_channel_name_mapping(bot_guilds)
            
            # Create reverse mapping (current -> database_name)
            reverse_mapping = {}
            for db_name, current_name in channel_mapping.items():
                reverse_mapping[current_name] = db_name
            
            # Try to find the database channel name
            # 1. Direct match with database name
            async with self.pool.execute("""
                SELECT DISTINCT channel_name
                FROM messages 
                WHERE channel_name = ?
                LIMIT 1
            """, (user_input,)) as cursor:
                exact_match = await cursor.fetchone()
                if exact_match:
                    return user_input
            
            # 2. Try reverse mapping (current name -> database name)
            if user_input in reverse_mapping:
                return reverse_mapping[user_input]
            
            # 3. Try fuzzy matching with database names
            async with self.pool.execute("""
                SELECT DISTINCT channel_name
                FROM messages 
                WHERE channel_name IS NOT NULL
            """) as cursor:
                all_channels = await cursor.fetchall()
                db_channel_names = [ch[0] for ch in all_channels if ch[0]]
            
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
    
    async def get_user_insights(self, user_name: str) -> str:
        """Get comprehensive insights for a specific user matching the original format"""
        try:
            await self.initialize()
            
            # Find the user by name (case-insensitive)
            async with self.pool.execute(f"""
                SELECT DISTINCT author_id, author_display_name, author_name
                FROM messages
                WHERE {self.base_filter}
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
                if user_name.lower() == current_name.lower():
                    best_match = user
                    break
                elif user_name.lower() in current_name.lower():
                    best_match = user
            
            if not best_match:
                best_match = matching_users[0]
            
            author_id, display_name, author_name = best_match
            display_name = display_name or author_name
            
            # Get basic user statistics
            async with self.pool.execute(f"""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT channel_name) as channels_active,
                    AVG(LENGTH(content)) as avg_message_length,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM messages
                WHERE author_id = ? AND {self.base_filter}
                AND content IS NOT NULL
            """, (author_id,)) as cursor:
                stats = await cursor.fetchone()
            
            if not stats or stats[0] == 0:
                return f"❌ No messages found for user '{display_name}'"
            
            total_messages, channels_active, avg_length, active_days, first_msg, last_msg = stats
            
            # Get detailed channel activity with average message lengths
            async with self.pool.execute(f"""
                SELECT 
                    channel_name, 
                    COUNT(*) as message_count,
                    AVG(LENGTH(content)) as avg_chars_per_message
                FROM messages
                WHERE author_id = ? AND {self.base_filter}
                AND content IS NOT NULL
                GROUP BY channel_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (author_id,)) as cursor:
                channel_activity = await cursor.fetchall()
            
            # Get activity by time of day (grouped into periods)
            async with self.pool.execute(f"""
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
                WHERE author_id = ? AND {self.base_filter}
                AND timestamp IS NOT NULL
                GROUP BY time_period
                ORDER BY messages DESC
            """, (author_id,)) as cursor:
                time_activity = await cursor.fetchall()
            
            # Get recent messages for content analysis
            async with self.pool.execute(f"""
                SELECT content
                FROM messages
                WHERE author_id = ? AND {self.base_filter}
                AND content IS NOT NULL 
                AND LENGTH(content) > 20
                ORDER BY timestamp DESC
                LIMIT 200
            """, (author_id,)) as cursor:
                recent_messages = await cursor.fetchall()
            
            # Extract key concepts using advanced spaCy analysis (same as channel analysis)
            user_concepts = []
            technical_terms = []
            business_concepts = []
            innovation_indicators = []
            complex_phrases = []
            
            if recent_messages:
                try:
                    import spacy
                    from collections import Counter
                    import re
                    
                    # Load spaCy model
                    try:
                        nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        print("spaCy model not found. Will use basic concept extraction.")
                        user_concepts = await self.extract_concepts_from_content(recent_messages)
                    else:
                        # Advanced text cleaning (same as channel analysis)
                        def clean_content(text):
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
                        
                        # Advanced concept extraction using spaCy (same patterns as channel analysis)
                        def extract_complex_concepts(doc):
                            concepts = []
                            
                            # Extract compound subjects with their predicates
                            for token in doc:
                                if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
                                    subject_span = doc[token.left_edge.i:token.right_edge.i+1]
                                    if token.head.pos_ == "VERB":
                                        verb = token.head
                                        predicate_parts = [verb.text]
                                        for child in verb.children:
                                            if child.dep_ in ["dobj", "pobj", "attr", "prep"]:
                                                obj_span = doc[child.left_edge.i:child.right_edge.i+1]
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
                                            prep_phrase = doc[child.i:child.right_edge.i+1]
                                            extended_phrase += f" {prep_phrase.text}"
                                if len(extended_phrase.split()) >= 3 and len(extended_phrase) > 20:
                                    concepts.append(extended_phrase.lower().strip())
                            
                            # Extract technical compounds
                            for i, token in enumerate(doc[:-2]):
                                if (token.pos_ in ["NOUN", "PROPN"] and 
                                    doc[i+1].pos_ in ["NOUN", "PROPN", "ADJ"] and 
                                    doc[i+2].pos_ in ["NOUN", "PROPN"]):
                                    compound = f"{token.text} {doc[i+1].text} {doc[i+2].text}"
                                    j = i + 3
                                    while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and j < i + 6:
                                        compound += f" {doc[j].text}"
                                        j += 1
                                    if len(compound.split()) >= 3:
                                        concepts.append(compound.lower())
                            
                            return concepts
                        
                        # Domain-specific patterns (same as channel analysis)
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
                        
                        innovation_patterns = [
                            r'\b(innovation|transformation|disruption|breakthrough|cutting.edge)\b',
                            r'\b(future|trend|emerging|next.gen|state.of.the.art|revolutionary)\b',
                            r'\b(experiment|prototype|pilot|proof.of_concept|MVP|beta)\b'
                        ]
                        
                        # Process user's messages
                        all_complex_concepts = []
                        tech_counter = Counter()
                        business_counter = Counter()
                        innovation_counter = Counter()
                        
                        # Combine recent messages
                        combined_text = " ".join([msg[0] for msg in recent_messages if msg[0]])
                        cleaned_content = clean_content(combined_text)
                        
                        if cleaned_content and len(cleaned_content.split()) >= 10:
                            try:
                                doc = nlp(cleaned_content[:500000])  # Process user's content
                                
                                # Extract complex concepts
                                complex_concepts = extract_complex_concepts(doc)
                                all_complex_concepts.extend(complex_concepts)
                                
                                # Extract simple patterns
                                for pattern in tech_patterns:
                                    matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                                    for match in matches:
                                        tech_counter[match.lower()] += 1
                                
                                for pattern in business_patterns:
                                    matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                                    for match in matches:
                                        business_counter[match.lower()] += 1
                                
                                for pattern in innovation_patterns:
                                    matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
                                    for match in matches:
                                        innovation_counter[match.lower()] += 1
                                
                            except Exception as e:
                                print(f"Error in spaCy processing for {display_name}: {e}")
                        
                        # Compile results
                        complex_phrases = [concept for concept, count in Counter(all_complex_concepts).most_common(8) if count >= 1]
                        technical_terms = [term for term, count in tech_counter.most_common(5) if count >= 1]
                        business_concepts = [concept for concept, count in business_counter.most_common(5) if count >= 1]
                        innovation_indicators = [pattern for pattern, count in innovation_counter.most_common(5) if count >= 1]
                        
                        # Combine all concepts for the traditional display
                        user_concepts = (complex_phrases[:4] + technical_terms[:2] + 
                                       business_concepts[:2] + innovation_indicators[:2])[:8]
                        
                except Exception as e:
                    print(f"Error extracting advanced concepts for {display_name}: {e}")
                    # Fallback to basic extraction
                    try:
                        user_concepts = await self.extract_concepts_from_content(recent_messages)
                    except:
                        user_concepts = []
            
            # Generate user activity chart for past 30 days
            chart_path = None
            try:
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                from datetime import datetime, timedelta
                import os
                
                # Get daily message counts for past 30 days for this user
                async with self.pool.execute(f"""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as messages
                    FROM messages 
                    WHERE author_id = ? AND {self.base_filter}
                    AND timestamp IS NOT NULL
                    AND DATE(timestamp) >= DATE('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date ASC
                """, (author_id,)) as cursor:
                    daily_activity = await cursor.fetchall()
                
                if daily_activity and len(daily_activity) > 1:
                    # Prepare data for plotting
                    dates = []
                    message_counts = []
                    
                    for date_str, count in daily_activity:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            dates.append(date_obj)
                            message_counts.append(count)
                        except:
                            continue
                    
                    if dates and message_counts:
                        # Create the plot
                        plt.figure(figsize=(12, 6))
                        plt.bar(dates, message_counts, color='#5865F2', alpha=0.7, edgecolor='#4752C4', linewidth=1)
                        
                        # Clean user name for chart title (remove emojis and special chars)
                        clean_user_name = re.sub(r'[^\w\s-]', '', display_name)
                        
                        # Formatting
                        plt.title(f'Daily Message Activity - {clean_user_name}', fontsize=16, fontweight='bold', pad=20)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Number of Messages', fontsize=12)
                        
                        # Format x-axis
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                        plt.xticks(rotation=45)
                        
                        # Add grid for better readability
                        plt.grid(True, alpha=0.3, axis='y')
                        
                        # Add some statistics to the plot
                        avg_messages = sum(message_counts) / len(message_counts)
                        max_messages = max(message_counts)
                        
                        plt.axhline(y=avg_messages, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Average: {avg_messages:.1f} msg/day')
                        
                        plt.legend()
                        plt.tight_layout()
                        
                        # Save the chart with sanitized filename
                        safe_user_name = re.sub(r'[^\w\s-]', '', display_name).replace(' ', '_').strip('_')
                        if not safe_user_name:
                            safe_user_name = "unknown_user"
                        
                        chart_filename = f"user_activity_{safe_user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        chart_path = os.path.join('temp', chart_filename)
                        
                        # Ensure temp directory exists
                        os.makedirs('temp', exist_ok=True)
                        
                        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
            except Exception as e:
                print(f"Error generating user activity chart: {e}")
            
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
                    result += f"• First Message: {self.format_timestamp(first_msg)}\n"
                    result += f"• Last Message: {self.format_timestamp(last_msg)}\n"
            
            result += "\n"
            
            # Channel Activity
            if channel_activity:
                result += f"**📍 Channel Activity:**\n"
                for channel, count, avg_chars in channel_activity:
                    result += f"• #{channel}: {count} messages (avg {avg_chars:.0f} chars)\n"
                result += "\n"
            
            # Activity by Time of Day
            if time_activity:
                result += f"**� Activity by Time of Day:**\n"
                for period, count in time_activity:
                    result += f"• {period}: {count} messages\n"
                result += "\n"
            
            # Key Topics & Concepts
            if user_concepts:
                result += f"**🧠 Key Topics & Concepts:**\n"
                # Format concepts with title case and bullet points
                formatted_concepts = []
                for concept in user_concepts[:8]:
                    # Title case each word and clean up
                    formatted_concept = ' '.join(word.capitalize() for word in concept.split())
                    formatted_concepts.append(f"• {formatted_concept}")
                
                result += '\n'.join(formatted_concepts) + "\n"
            
            # Return both text and chart path if chart was generated
            if chart_path and os.path.exists(chart_path):
                return (result, chart_path)
            else:
                return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error getting user insights: {str(e)}"
    
    async def get_channel_insights(self, channel_name: str) -> str:
        """Get comprehensive channel statistics and insights"""
        try:
            await self.initialize()
            
            # Resolve the channel name to the actual database name
            resolved_channel = await self.resolve_channel_name(channel_name)
            
            # Get basic channel statistics
            async with self.pool.execute(f"""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT author_id) as unique_users,
                    AVG(LENGTH(content)) as avg_message_length,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND content IS NOT NULL
            """, (resolved_channel,)) as cursor:
                stats = await cursor.fetchone()
            
            if not stats or stats[0] == 0:
                # Try to find similar channel names
                async with self.pool.execute(f"""
                    SELECT DISTINCT channel_name, COUNT(*) as msg_count
                    FROM messages 
                    WHERE {self.base_filter}
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
            
            total_messages, unique_users, avg_length, first_msg, last_msg = stats
            
            # Get engagement metrics
            async with self.pool.execute(f"""
                SELECT 
                    COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                    COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                    COUNT(CASE WHEN has_reactions = 1 THEN 1 END) as posts_with_reactions
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
            """, (resolved_channel,)) as cursor:
                engagement = await cursor.fetchone()
            
            total_replies, original_posts, posts_with_reactions = engagement
            replies_per_post = total_replies / original_posts if original_posts > 0 else 0
            reaction_rate = (posts_with_reactions / total_messages * 100) if total_messages > 0 else 0
            
            # Get top contributors
            async with self.pool.execute(f"""
                SELECT 
                    COALESCE(author_display_name, author_name) as display_name,
                    COUNT(*) as message_count,
                    AVG(LENGTH(content)) as avg_chars
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND content IS NOT NULL
                GROUP BY author_id, author_display_name, author_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (resolved_channel,)) as cursor:
                contributors = await cursor.fetchall()
            
            # Get peak activity hours
            async with self.pool.execute(f"""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as messages
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND timestamp IS NOT NULL
                GROUP BY strftime('%H', timestamp)
                ORDER BY messages DESC
                LIMIT 3
            """, (resolved_channel,)) as cursor:
                peak_hours = await cursor.fetchall()
            
            # Get activity by day of week
            async with self.pool.execute(f"""
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
                WHERE channel_name = ? AND {self.base_filter}
                AND timestamp IS NOT NULL
                GROUP BY strftime('%w', timestamp)
                ORDER BY messages DESC
                LIMIT 3
            """, (resolved_channel,)) as cursor:
                day_activity = await cursor.fetchall()
            
            # Get recent activity (last 7 days)
            async with self.pool.execute(f"""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as messages
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND timestamp IS NOT NULL
                AND DATE(timestamp) >= DATE('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 7
            """, (resolved_channel,)) as cursor:
                recent_activity = await cursor.fetchall()
            
            # Get channel health metrics (activity in last week)
            async with self.pool.execute(f"""
                SELECT COUNT(DISTINCT author_id) as weekly_active
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND timestamp >= datetime('now', '-7 days')
            """, (resolved_channel,)) as cursor:
                weekly_active_result = await cursor.fetchone()
            
            weekly_active = weekly_active_result[0] if weekly_active_result else 0
            channel_amp = (weekly_active / unique_users * 100) if unique_users > 0 else 0
            
            # Get inactive users (users who posted before but not in last 7 days)
            async with self.pool.execute(f"""
                SELECT COUNT(DISTINCT author_id) as inactive_users
                FROM messages 
                WHERE channel_name = ? AND {self.base_filter}
                AND timestamp < datetime('now', '-7 days')
                AND author_id NOT IN (
                    SELECT DISTINCT author_id 
                    FROM messages 
                    WHERE channel_name = ? AND {self.base_filter}
                    AND timestamp >= datetime('now', '-7 days')
                )
            """, (resolved_channel, resolved_channel)) as cursor:
                inactive_result = await cursor.fetchone()
            
            inactive_users = inactive_result[0] if inactive_result else 0
            inactive_percentage = (inactive_users / unique_users * 100) if unique_users > 0 else 0
            
            # Get total channel members (from new channel_members table)
            total_channel_members = 0
            lurkers = 0
            participation_rate = 0
            
            try:
                async with self.pool.execute(f"""
                    SELECT COUNT(DISTINCT user_id) as total_members
                    FROM channel_members 
                    WHERE channel_name = ?
                """, (resolved_channel,)) as cursor:
                    member_result = await cursor.fetchone()
                
                if member_result and member_result[0]:
                    total_channel_members = member_result[0]
                    lurkers = total_channel_members - unique_users
                    participation_rate = (unique_users / total_channel_members * 100) if total_channel_members > 0 else 0
                    
            except Exception as e:
                # If channel_members table doesn't exist or has no data, continue without it
                print(f"Note: Channel membership data not available: {e}")
            
            # Extract top topics using advanced spaCy analysis
            top_topics = []
            try:
                import spacy
                from collections import Counter
                import re
                
                # Get content for topic analysis
                async with self.pool.execute(f"""
                    SELECT content
                    FROM messages 
                    WHERE channel_name = ? AND {self.base_filter}
                    AND content IS NOT NULL 
                    AND LENGTH(content) > 30
                    ORDER BY timestamp DESC
                    LIMIT 200
                """, (resolved_channel,)) as cursor:
                    topic_messages = await cursor.fetchall()
                
                if topic_messages:
                    try:
                        nlp = spacy.load("en_core_web_sm")
                        
                        # Advanced text cleaning (same as channel analysis)
                        def clean_content(text):
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
                            text = re.sub(r'\s+', ' ', text).strip()
                            return text
                        
                        # Extract complex concepts using spaCy
                        def extract_complex_topics(doc):
                            topics = []
                            
                            # Extract compound subjects with their predicates
                            for token in doc:
                                if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
                                    subject_span = doc[token.left_edge.i:token.right_edge.i+1]
                                    if token.head.pos_ == "VERB":
                                        verb = token.head
                                        predicate_parts = [verb.text]
                                        for child in verb.children:
                                            if child.dep_ in ["dobj", "pobj", "attr"]:
                                                obj_span = doc[child.left_edge.i:child.right_edge.i+1]
                                                predicate_parts.append(obj_span.text)
                                        if len(predicate_parts) > 1:
                                            full_topic = f"{subject_span.text} {' '.join(predicate_parts)}"
                                            if len(full_topic.split()) >= 2 and len(full_topic) > 10:
                                                topics.append(full_topic.lower().strip())
                            
                            # Extract extended noun phrases (technical terms, concepts)
                            for chunk in doc.noun_chunks:
                                if len(chunk.text.split()) >= 2 and len(chunk.text) > 8:
                                    # Filter out common stopwords and generic terms
                                    if not any(word in chunk.text.lower() for word in ['this', 'that', 'some', 'any', 'the', 'these', 'those']):
                                        topics.append(chunk.text.lower().strip())
                            
                            # Extract technical compounds and domain-specific terms
                            for i, token in enumerate(doc[:-1]):
                                if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                                    doc[i+1].pos_ in ["NOUN", "PROPN"]):
                                    compound = f"{token.text} {doc[i+1].text}"
                                    # Look ahead for longer compounds
                                    j = i + 2
                                    while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and j < i + 4:
                                        compound += f" {doc[j].text}"
                                        j += 1
                                    if len(compound.split()) >= 2 and len(compound) > 6:
                                        topics.append(compound.lower())
                            
                            return topics
                        
                        # Process messages and extract topics
                        all_topics = []
                        combined_text = " ".join([msg[0] for msg in topic_messages if msg[0]])
                        cleaned_content = clean_content(combined_text)
                        
                        if cleaned_content and len(cleaned_content.split()) >= 20:
                            doc = nlp(cleaned_content[:500000])  # Limit text length
                            complex_topics = extract_complex_topics(doc)
                            all_topics.extend(complex_topics)
                        
                        # Count and filter topics
                        topic_counter = Counter(all_topics)
                        
                        # Filter for meaningful topics (avoid very short, very long, or common words)
                        filtered_topics = []
                        for topic, count in topic_counter.most_common(50):
                            if (6 <= len(topic) <= 60 and 
                                count >= 2 and 
                                len(topic.split()) >= 2 and
                                topic not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'will', 'can', 'would', 'could'] and
                                not topic.startswith(('i ', 'you ', 'we ', 'they '))):
                                # Title case the topic for display
                                formatted_topic = ' '.join(word.capitalize() for word in topic.split())
                                filtered_topics.append(formatted_topic)
                        
                        top_topics = filtered_topics[:10]
                        
                    except OSError:
                        print("spaCy model not found for topic extraction.")
                    except Exception as e:
                        print(f"Error in topic extraction: {e}")
                        
            except Exception as e:
                print(f"Error extracting topics: {e}")
            
            # Generate activity chart for past 30 days
            chart_path = None
            try:
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                from datetime import datetime, timedelta
                import os
                
                # Get daily message counts for past 30 days
                async with self.pool.execute(f"""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as messages
                    FROM messages 
                    WHERE channel_name = ? AND {self.base_filter}
                    AND timestamp IS NOT NULL
                    AND DATE(timestamp) >= DATE('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date ASC
                """, (resolved_channel,)) as cursor:
                    daily_activity = await cursor.fetchall()
                
                if daily_activity and len(daily_activity) > 1:
                    # Prepare data for plotting
                    dates = []
                    message_counts = []
                    
                    for date_str, count in daily_activity:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            dates.append(date_obj)
                            message_counts.append(count)
                        except:
                            continue
                    
                    if dates and message_counts:
                        # Create the plot
                        plt.figure(figsize=(12, 6))
                        plt.bar(dates, message_counts, color='#5865F2', alpha=0.7, edgecolor='#4752C4', linewidth=1)
                        
                        # Clean channel name for chart title (remove emojis)
                        clean_channel_name = re.sub(r'[^\w\s-]', '', resolved_channel)
                        
                        # Formatting
                        plt.title(f'Daily Message Activity - {clean_channel_name}', fontsize=16, fontweight='bold', pad=20)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Number of Messages', fontsize=12)
                        
                        # Format x-axis
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                        plt.xticks(rotation=45)
                        
                        # Add grid for better readability
                        plt.grid(True, alpha=0.3, axis='y')
                        
                        # Add some statistics to the plot
                        avg_messages = sum(message_counts) / len(message_counts)
                        max_messages = max(message_counts)
                        
                        plt.axhline(y=avg_messages, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Average: {avg_messages:.1f} msg/day')
                        
                        plt.legend()
                        plt.tight_layout()
                        
                        # Save the chart with sanitized filename
                        safe_channel_name = re.sub(r'[^\w\s-]', '', resolved_channel).replace(' ', '_').strip('_')
                        if not safe_channel_name:
                            safe_channel_name = "unknown_channel"
                        
                        chart_filename = f"channel_activity_{safe_channel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        chart_path = os.path.join('temp', chart_filename)
                        
                        # Ensure temp directory exists
                        os.makedirs('temp', exist_ok=True)
                        
                        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
            except Exception as e:
                print(f"Error generating activity chart: {e}")
            
            # Format results
            result = f"**Channel Analysis: #{resolved_channel}**\n\n"
            
            # Basic Statistics
            result += f"**Basic Statistics:**\n"
            result += f"• Total Messages: {total_messages}\n"
            result += f"• Unique Users: {unique_users}\n"
            result += f"• Average Message Length: {avg_length:.1f} characters\n"
            
            if first_msg and last_msg:
                try:
                    from datetime import datetime
                    first_dt = datetime.fromisoformat(first_msg.replace('Z', '+00:00'))
                    last_dt = datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                    result += f"• First Message: {first_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    result += f"• Last Message: {last_dt.strftime('%Y-%m-%d %H:%M')}\n"
                except:
                    result += f"• First Message: {self.format_timestamp(first_msg)}\n"
                    result += f"• Last Message: {self.format_timestamp(last_msg)}\n"
            
            result += "\n"
            
            # Engagement Metrics
            result += f"**📈 Engagement Metrics:**\n"
            result += f"• Average Replies per Original Post: {replies_per_post:.2f}\n"
            result += f"• Posts with Reactions: {reaction_rate:.1f}% ({posts_with_reactions}/{total_messages})\n"
            result += f"• Total Replies: {total_replies} | Original Posts: {original_posts}\n\n"
            
            # Top Contributors
            if contributors:
                result += f"**Top Contributors:**\n"
                for name, count, avg_chars in contributors:
                    result += f"• {name}: {count} messages (avg {avg_chars:.0f} chars)\n"
                result += "\n"
            
            # Peak Activity Hours
            if peak_hours:
                result += f"**Peak Activity Hours:**\n"
                for hour, count in peak_hours:
                    result += f"• {hour}:00-{hour}:59: {count} messages\n"
                result += "\n"
            
            # Activity by Day
            if day_activity:
                result += f"**Activity by Day:**\n"
                for day, count in day_activity:
                    result += f"• {day}: {count} messages\n"
                result += "\n"
            
            # Recent Activity
            if recent_activity:
                result += f"**Recent Activity (Last 7 Days):**\n"
                for date, count in recent_activity:
                    result += f"• {date}: {count} messages\n"
                result += "\n"
            
            # Channel Health Metrics
            result += f"**📈 Channel Health Metrics:**\n"
            
            if total_channel_members > 0:
                # Enhanced metrics with full membership data
                result += f"• Total Channel Members: {total_channel_members}\n"
                result += f"• Members Who Ever Posted: {unique_users} ({participation_rate:.1f}%)\n"
                result += f"• Weekly Active Members: {weekly_active} ({(weekly_active/total_channel_members*100):.1f}% of total)\n"
                result += f"• Recently Inactive Members: {inactive_users} ({inactive_percentage:.1f}% of posters)\n"
                result += f"• Lurkers (Never Posted): {lurkers} ({(lurkers/total_channel_members*100):.1f}%)\n"
                result += f"• Participation Rate: {participation_rate:.1f}% (members who have posted)\n"
                result += f"• Activity Ratio: {weekly_active} active / {inactive_users} inactive / {lurkers} lurkers\n\n"
            else:
                # Fallback to message-based metrics
                result += f"• Total Members Ever Active: {unique_users}\n"
                result += f"• Weekly Active Members: {weekly_active} ({channel_amp:.1f}%)\n"
                result += f"• Recently Inactive Members: {inactive_users} ({inactive_percentage:.1f}%)\n"
                result += f"• Activity Ratio: {weekly_active} active / {inactive_users} inactive\n"
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
            import traceback
            traceback.print_exc()
            return f"Error getting channel insights: {str(e)}"