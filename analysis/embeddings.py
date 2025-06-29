"""
Embedding operations for Discord message analysis
"""
import asyncio
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.text_processing import preprocess_text


class EmbeddingManager:
    """Manages embedding model and operations"""
    
    def __init__(self):
        self.embedding_model = None
        self.model_loaded = False
        self.model_load_error = None
        self._model_loading = False

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

    def generate_message_embeddings(self, cursor, conn, batch_size: int = 100) -> None:
        """Generate and store embeddings for messages"""
        cursor.execute('SELECT id, content FROM messages WHERE content IS NOT NULL')
        messages = cursor.fetchall()
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            texts = [preprocess_text(msg['content']) for msg in batch]
            embeddings = self.embedding_model.encode(texts)
            
            for msg, embedding in zip(batch, embeddings):
                cursor.execute('''
                    INSERT OR REPLACE INTO message_embeddings (message_id, embedding)
                    VALUES (?, ?)
                ''', (msg['id'], embedding.tobytes()))
            
            conn.commit()
            print(f"Processed {i + len(batch)} messages")

    async def find_similar_messages_data(self, cursor, conn, message_id: int) -> Tuple[Dict, List[Tuple]]:
        """Find messages similar to a given message - returns data for processing"""
        try:
            # Get the target message
            cursor = conn.execute("""
                SELECT content, author_id, channel_name, timestamp
                FROM messages 
                WHERE id = ?
            """, (message_id,))
            
            target_msg = cursor.fetchone()
            if not target_msg:
                raise ValueError(f"Message with ID {message_id} not found")
            
            # Get other messages for comparison
            cursor = conn.execute("""
                SELECT id, content, author_id, channel_name, timestamp
                FROM messages 
                WHERE id != ? AND content IS NOT NULL AND content != ''
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (message_id,))
            
            messages = cursor.fetchall()
            
            if not messages:
                raise ValueError("No messages found for comparison")
            
            # Get embeddings
            target_embedding = self.get_embedding(target_msg['content'])
            if target_embedding is None:
                raise ValueError("Could not generate embedding for target message")
            
            message_embeddings = []
            for msg in messages:
                embedding = self.get_embedding(msg['content'])
                if embedding is not None:
                    message_embeddings.append((msg, embedding))
            
            if not message_embeddings:
                raise ValueError("Could not generate embeddings for comparison messages")
            
            # Calculate similarities
            similarities = []
            for msg, embedding in message_embeddings:
                similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                similarities.append((msg, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return target_msg, similarities
            
        except Exception as e:
            raise e


# Global embedding manager instance
embedding_manager = EmbeddingManager()


# Convenience functions that use the global manager
async def ensure_model_loaded():
    """Ensure the embedding model is loaded"""
    return await embedding_manager.ensure_model_loaded()


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for text if model is loaded"""
    return embedding_manager.get_embedding(text)


def generate_message_embeddings(cursor, conn, batch_size: int = 100) -> None:
    """Generate and store embeddings for messages"""
    return embedding_manager.generate_message_embeddings(cursor, conn, batch_size)


async def find_similar_messages_data(cursor, conn, message_id: int) -> Tuple[Dict, List[Tuple]]:
    """Find messages similar to a given message - returns data for processing"""
    return await embedding_manager.find_similar_messages_data(cursor, conn, message_id)
