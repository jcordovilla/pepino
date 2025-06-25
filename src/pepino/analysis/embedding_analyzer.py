"""
Embedding service for semantic search and content analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pepino.data.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories.embedding_repository import EmbeddingRepository

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing message embeddings."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.model = None
        self.device = None
        self.model_loaded = False
        self.embedding_dim = None

    def initialize(self):
        """Initialize the embedding model."""
        if self.model_loaded:
            return

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Initialize device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load model
            model_name = self.settings.embedding.model_name
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_loaded = True
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info("SentenceTransformer model loaded successfully")

        except ImportError as e:
            logger.error(f"Required dependencies not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model_loaded:
            self.initialize()

        if not text or not text.strip():
            return np.zeros(self.embedding_dim)

        try:
            # Generate embedding directly
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim)

    def batch_process_messages(
        self,
        db_path: str,
        channel_filter: Optional[str] = None,
        batch_size: int = 100,
        max_messages: Optional[int] = None,
    ) -> int:
        """Process messages in batches to generate embeddings."""
        if not self.model_loaded:
            self.initialize()

        try:
            # Initialize database
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)
            message_repo = MessageRepository(db_manager)

            # Get messages without embeddings
            messages = message_repo.get_messages_without_embeddings(
                channel_filter, max_messages
            )

            if not messages:
                logger.info("No messages found without embeddings")
                return 0

            total_processed = 0
            for i in range(0, len(messages), batch_size):
                batch = messages[i : i + batch_size]
                batch_texts = [msg["content"] for msg in batch]

                logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} messages)")

                # Generate embeddings for batch directly
                try:
                    embeddings = self.model.encode(
                        batch_texts, convert_to_numpy=True, show_progress_bar=False
                    )

                    # Store embeddings
                    for msg, embedding in zip(batch, embeddings):
                        embedding_repo.store_embedding(
                            msg["message_id"], embedding.tolist()
                        )

                    total_processed += len(batch)
                    logger.info(f"Processed {total_processed}/{len(messages)} messages")

                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue

            db_manager.close()
            logger.info(f"Completed processing {total_processed} messages")
            return total_processed

        except Exception as e:
            logger.error(f"Failed to batch process messages: {e}")
            return 0

    def find_similar_messages(
        self,
        db_path: str,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.7,
        channel_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find messages similar to a query text."""
        if not self.model_loaded:
            self.initialize()

        try:
            # Generate query embedding directly
            query_embedding = self.model.encode(query_text, convert_to_numpy=True)

            # Initialize database
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            # Get all embeddings
            stored_embeddings = embedding_repo.get_all_embeddings(channel_filter)

            if not stored_embeddings:
                logger.warning("No embeddings found in database")
                return []

            # Calculate similarities
            similarities = []
            for record in stored_embeddings:
                try:
                    stored_embedding = np.array(record["embedding"])
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(stored_embedding)
                    )

                    if similarity >= threshold:
                        similarities.append(
                            {
                                "message_id": record["message_id"],
                                "content": record["content"],
                                "author": record["author"],
                                "channel": record["channel"],
                                "timestamp": record["timestamp"],
                                "similarity": float(similarity),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")
                    continue

            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            db_manager.close()

            return similarities[:limit]

        except Exception as e:
            logger.error(f"Failed to find similar messages: {e}")
            return []

    def get_embedding_statistics(self, db_path: str) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        try:
            # Initialize database
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            stats = embedding_repo.get_embedding_statistics()
            db_manager.close()

            return stats

        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            return {}

    def cleanup_embeddings(self, db_path: str) -> int:
        """Remove embeddings for messages that no longer exist."""
        try:
            # Initialize database
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            deleted_count = embedding_repo.cleanup_orphaned_embeddings()
            db_manager.close()

            logger.info(f"Cleaned up {deleted_count} orphaned embeddings")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup embeddings: {e}")
            return 0
