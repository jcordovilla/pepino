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

class MessageAnalyzer:
    def __init__(self, db_path: str = 'discord_messages.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Initialize the sentence transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize LDA model
        self.lda = LatentDirichletAllocation(
            n_components=10,
            max_iter=10,
            learning_method='online',
            random_state=42
        )

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

    def find_similar_messages(self, message_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar messages using embeddings"""
        self.cursor.execute('SELECT embedding FROM message_embeddings WHERE message_id = ?', (message_id,))
        result = self.cursor.fetchone()
        if not result:
            return []
        
        query_embedding = np.frombuffer(result['embedding'])
        
        self.cursor.execute('SELECT message_id, embedding FROM message_embeddings')
        all_embeddings = self.cursor.fetchall()
        
        similarities = []
        for row in all_embeddings:
            if row['message_id'] == message_id:
                continue
            embedding = np.frombuffer(row['embedding'])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((row['message_id'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        similar_messages = []
        for msg_id, similarity in similarities[:limit]:
            self.cursor.execute('''
                SELECT m.*, a.name as author_name
                FROM messages m
                LEFT JOIN messages a ON m.author_id = a.id
                WHERE m.id = ?
            ''', (msg_id,))
            msg = self.cursor.fetchone()
            if msg:
                similar_messages.append({
                    'message': dict(msg),
                    'similarity': float(similarity)
                })
        
        return similar_messages

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

    def update_word_frequencies(self) -> None:
        """Update word frequency statistics"""
        self.cursor.execute('SELECT channel_id, content FROM messages WHERE content IS NOT NULL')
        messages = self.cursor.fetchall()
        
        channel_word_counts = defaultdict(Counter)
        for msg in messages:
            words = word_tokenize(self.preprocess_text(msg['content']))
            # Remove stopwords
            words = [w for w in words if w not in stopwords.words('english')]
            channel_word_counts[msg['channel_id']].update(words)
        
        # Update database
        for channel_id, word_counts in channel_word_counts.items():
            for word, count in word_counts.items():
                self.cursor.execute('''
                    INSERT OR REPLACE INTO word_frequencies (word, channel_id, frequency, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (word, channel_id, count, datetime.utcnow().isoformat()))
        
        self.conn.commit()

    def update_user_statistics(self) -> None:
        """Update user activity statistics"""
        self.cursor.execute('''
            SELECT 
                author_id,
                channel_id,
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_length,
                GROUP_CONCAT(strftime('%H', timestamp)) as hours
            FROM messages
            GROUP BY author_id, channel_id
        ''')
        
        stats = self.cursor.fetchall()
        for stat in stats:
            hours = stat['hours'].split(',')
            active_hours = json.dumps(Counter(hours))
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO user_statistics 
                (user_id, channel_id, message_count, avg_message_length, active_hours, last_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                stat['author_id'],
                stat['channel_id'],
                stat['message_count'],
                stat['avg_length'],
                active_hours,
                datetime.utcnow().isoformat()
            ))
        
        self.conn.commit()

    def update_conversation_chains(self) -> None:
        """Update conversation chain analysis"""
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
        
        # Update database
        for root_id, chain in chains.items():
            last_id = chain[-1]
            self.cursor.execute('''
                INSERT OR REPLACE INTO conversation_chains 
                (root_message_id, last_message_id, message_count, created_at)
                VALUES (?, ?, ?, ?)
            ''', (root_id, last_id, len(chain), datetime.utcnow().isoformat()))
        
        self.conn.commit()

    def update_temporal_stats(self) -> None:
        """Update temporal statistics"""
        self.cursor.execute('''
            SELECT 
                channel_id,
                date(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as active_users,
                AVG(LENGTH(content)) as avg_length
            FROM messages
            GROUP BY channel_id, date(timestamp)
        ''')
        
        stats = self.cursor.fetchall()
        for stat in stats:
            self.cursor.execute('''
                INSERT OR REPLACE INTO message_temporal_stats 
                (channel_id, date, message_count, active_users, avg_message_length)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                stat['channel_id'],
                stat['date'],
                stat['message_count'],
                stat['active_users'],
                stat['avg_length']
            ))
        
        self.conn.commit()

    def get_channel_insights(self, channel_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a channel"""
        insights = {}
        
        # Basic statistics
        self.cursor.execute('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_message_length,
                COUNT(CASE WHEN has_reactions THEN 1 END) as messages_with_reactions
            FROM messages
            WHERE channel_id = ?
        ''', (channel_id,))
        insights['basic_stats'] = dict(self.cursor.fetchone())
        
        # Top users
        self.cursor.execute('''
            SELECT 
                author_id,
                author_name,
                COUNT(*) as message_count
            FROM messages
            WHERE channel_id = ?
            GROUP BY author_id
            ORDER BY message_count DESC
            LIMIT 10
        ''', (channel_id,))
        insights['top_users'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Word frequencies
        self.cursor.execute('''
            SELECT word, frequency
            FROM word_frequencies
            WHERE channel_id = ?
            ORDER BY frequency DESC
            LIMIT 20
        ''', (channel_id,))
        insights['top_words'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Temporal patterns
        self.cursor.execute('''
            SELECT date, message_count, active_users
            FROM message_temporal_stats
            WHERE channel_id = ?
            ORDER BY date DESC
            LIMIT 30
        ''', (channel_id,))
        insights['temporal_patterns'] = [dict(row) for row in self.cursor.fetchall()]
        
        return insights

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a user"""
        insights = {}
        
        # Basic statistics
        self.cursor.execute('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT channel_id) as active_channels,
                AVG(LENGTH(content)) as avg_message_length,
                COUNT(CASE WHEN has_reactions THEN 1 END) as messages_with_reactions
            FROM messages
            WHERE author_id = ?
        ''', (user_id,))
        insights['basic_stats'] = dict(self.cursor.fetchone())
        
        # Channel activity
        self.cursor.execute('''
            SELECT 
                channel_id,
                channel_name,
                COUNT(*) as message_count
            FROM messages
            WHERE author_id = ?
            GROUP BY channel_id
            ORDER BY message_count DESC
        ''', (user_id,))
        insights['channel_activity'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Temporal patterns
        self.cursor.execute('''
            SELECT active_hours
            FROM user_statistics
            WHERE user_id = ?
        ''', (user_id,))
        result = self.cursor.fetchone()
        if result:
            insights['active_hours'] = json.loads(result['active_hours'])
        
        return insights

    def run_all_analyses(self) -> None:
        """Run all analysis updates"""
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
        topics = self.perform_topic_modeling()
        print("\nTopics found:")
        for topic, words in topics.items():
            print(f"\n{topic}: {', '.join(words)}")

if __name__ == '__main__':
    analyzer = MessageAnalyzer()
    analyzer.run_all_analyses() 