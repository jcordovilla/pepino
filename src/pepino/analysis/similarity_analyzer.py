"""
Similarity service for semantic search and content analysis.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories import EmbeddingRepository, MessageRepository
from pepino.logging_config import get_logger

from .embedding_analyzer import EmbeddingService

logger = get_logger(__name__)


class SimilarityService:
    """Service for semantic search and content similarity analysis."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)

    def find_similar_messages(
        self, query_text: str, db_path: str, limit: int = 10, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find messages semantically similar to the query text."""
        try:
            self.embedding_service.initialize()

            # Use the embedding service to find similar messages
            similar_messages = self.embedding_service.find_similar_messages(
                db_path, query_text, limit, threshold
            )

            return similar_messages

        except Exception as e:
            logger.error(f"Failed to find similar messages: {e}")
            raise

    def detect_duplicates(
        self,
        db_path: str,
        channel_id: Optional[str] = None,
        similarity_threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Detect duplicate or near-duplicate messages."""
        try:
            self.embedding_service.initialize()

            # Initialize database and repository
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            # Get messages with embeddings using repository
            messages = embedding_repo.get_messages_with_embeddings(
                channel_id, limit=1000
            )

            if len(messages) < 2:
                db_manager.close()
                return []

            # Convert embeddings to numpy arrays
            embeddings = []
            message_data = []

            for msg in messages:
                embedding = np.array(msg["embedding"])
                embeddings.append(embedding)
                message_data.append(
                    {
                        "id": msg["message_id"],
                        "content": msg["content"],
                        "author": msg["author"],
                        "channel": msg["channel"],
                        "timestamp": msg["timestamp"],
                    }
                )

            # Calculate similarity matrix
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)

            # Find duplicate pairs
            duplicate_groups = []
            processed_pairs = set()

            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarity = similarity_matrix[i][j]

                    if similarity >= similarity_threshold:
                        pair_key = tuple(sorted([i, j]))
                        if pair_key not in processed_pairs:
                            duplicate_groups.append(
                                {
                                    "message1": message_data[i],
                                    "message2": message_data[j],
                                    "similarity": round(similarity, 3),
                                }
                            )
                            processed_pairs.add(pair_key)

            # Sort by similarity
            duplicate_groups.sort(key=lambda x: x["similarity"], reverse=True)

            db_manager.close()
            return duplicate_groups

        except Exception as e:
            logger.error(f"Failed to detect duplicates: {e}")
            raise

    def cluster_content(
        self,
        db_path: str,
        channel_id: Optional[str] = None,
        min_cluster_size: int = 3,
        eps: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Cluster messages by content similarity."""
        try:
            self.embedding_service.initialize()

            # Initialize database and repository
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            # Get messages with embeddings using repository
            messages = embedding_repo.get_messages_with_embeddings(
                channel_id, limit=2000
            )

            if len(messages) < min_cluster_size:
                db_manager.close()
                return []

            # Convert embeddings to numpy arrays
            embeddings = []
            message_data = []

            for msg in messages:
                embedding = np.array(msg["embedding"])
                embeddings.append(embedding)
                message_data.append(
                    {
                        "id": msg["message_id"],
                        "content": msg["content"],
                        "author": msg["author"],
                        "channel": msg["channel"],
                        "timestamp": msg["timestamp"],
                    }
                )

            # Perform clustering
            embeddings_array = np.array(embeddings)
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(
                embeddings_array
            )

            # Group messages by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                if label >= 0:  # Skip noise points (label = -1)
                    clusters[label].append(message_data[i])

            # Convert to list format
            cluster_results = []
            for cluster_id, cluster_messages in clusters.items():
                if len(cluster_messages) >= min_cluster_size:
                    # Calculate cluster statistics
                    authors = [msg["author"] for msg in cluster_messages]
                    channels = [msg["channel"] for msg in cluster_messages]

                    cluster_results.append(
                        {
                            "cluster_id": cluster_id,
                            "message_count": len(cluster_messages),
                            "unique_authors": len(set(authors)),
                            "unique_channels": len(set(channels)),
                            "messages": cluster_messages,
                            "top_authors": self._get_top_items(authors, 5),
                            "top_channels": self._get_top_items(channels, 3),
                        }
                    )

            # Sort by cluster size
            cluster_results.sort(key=lambda x: x["message_count"], reverse=True)

            db_manager.close()
            return cluster_results

        except Exception as e:
            logger.error(f"Failed to cluster content: {e}")
            raise

    def find_semantic_groups(
        self,
        db_path: str,
        channel_id: Optional[str] = None,
        query_terms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find semantic groups in messages."""
        try:
            if query_terms:
                return self._find_groups_by_queries(db_path, channel_id, query_terms)
            else:
                return self._discover_semantic_groups(db_path, channel_id)

        except Exception as e:
            logger.error(f"Failed to find semantic groups: {e}")
            raise

    def _find_groups_by_queries(
        self, db_path: str, channel_id: Optional[str], query_terms: List[str]
    ) -> List[Dict[str, Any]]:
        """Find groups based on specific query terms."""
        try:
            self.embedding_service.initialize()

            groups = []
            for query in query_terms:
                similar_messages = self.embedding_service.find_similar_messages(
                    db_path, query, limit=20, threshold=0.6, channel_filter=channel_id
                )

                if similar_messages:
                    groups.append(
                        {
                            "query": query,
                            "message_count": len(similar_messages),
                            "messages": similar_messages,
                            "avg_similarity": sum(
                                msg["similarity"] for msg in similar_messages
                            )
                            / len(similar_messages),
                        }
                    )

            # Sort by message count
            groups.sort(key=lambda x: x["message_count"], reverse=True)
            return groups

        except Exception as e:
            logger.error(f"Failed to find groups by queries: {e}")
            raise

    def _discover_semantic_groups(
        self, db_path: str, channel_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Discover semantic groups automatically."""
        try:
            # Use clustering to discover groups
            clusters = self.cluster_content(
                db_path, channel_id, min_cluster_size=5, eps=0.25
            )

            # Convert clusters to semantic groups
            semantic_groups = []
            for cluster in clusters:
                representative_content = self._extract_representative_content(
                    cluster["messages"]
                )

                semantic_groups.append(
                    {
                        "group_id": cluster["cluster_id"],
                        "theme": representative_content,
                        "message_count": cluster["message_count"],
                        "unique_authors": cluster["unique_authors"],
                        "messages": cluster["messages"][:10],  # Limit for readability
                        "top_authors": cluster["top_authors"],
                    }
                )

            return semantic_groups

        except Exception as e:
            logger.error(f"Failed to discover semantic groups: {e}")
            raise

    def _extract_representative_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract representative content from a group of messages."""
        if not messages:
            return ""

        # Simple approach: use the shortest message as representative
        shortest_msg = min(messages, key=lambda x: len(x["content"]))
        return (
            shortest_msg["content"][:100] + "..."
            if len(shortest_msg["content"]) > 100
            else shortest_msg["content"]
        )

    def _get_top_items(self, items: List[str], limit: int) -> List[Tuple[str, int]]:
        """Get top items by frequency."""
        from collections import Counter

        counter = Counter(items)
        return counter.most_common(limit)

    def calculate_similarity_matrix(
        self, db_path: str, message_ids: List[str]
    ) -> Dict[str, Any]:
        """Calculate similarity matrix for specific messages."""
        try:
            self.embedding_service.initialize()

            # Initialize database and repository
            db_manager = DatabaseManager(db_path)
            db_manager.initialize()
            embedding_repo = EmbeddingRepository(db_manager)

            # Get embeddings for specified messages
            embeddings_data = []
            for message_id in message_ids:
                embedding = embedding_repo.get_embedding(message_id)
                if embedding:
                    embeddings_data.append(
                        {"message_id": message_id, "embedding": np.array(embedding)}
                    )

            if len(embeddings_data) < 2:
                db_manager.close()
                return {"error": "Need at least 2 messages with embeddings"}

            # Calculate similarity matrix
            embeddings = [item["embedding"] for item in embeddings_data]
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)

            db_manager.close()

            return {
                "message_ids": [item["message_id"] for item in embeddings_data],
                "similarity_matrix": similarity_matrix.tolist(),
            }

        except Exception as e:
            logger.error(f"Failed to calculate similarity matrix: {e}")
            raise
