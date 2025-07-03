"""
Embedding repository for data access operations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config import settings
from ..database.manager import DatabaseManager


class EmbeddingRepository:
    """Repository for embedding data access."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        # Use unified settings for base filter
        self.base_filter = settings.base_filter.strip()

    def store_embedding(self, message_id: str, embedding: List[float]):
        """Store an embedding for a message."""
        query = "INSERT OR REPLACE INTO embeddings (message_id, embedding) VALUES (?, ?)"
        # Convert list to JSON string for storage
        import json
        embedding_json = json.dumps(embedding)
        self.db_manager.execute_query(query, (message_id, embedding_json))

    def get_embedding(self, message_id: str) -> Optional[List[float]]:
        """Get embedding for a specific message."""
        query = "SELECT embedding FROM embeddings WHERE message_id = ?"
        row = self.db_manager.execute_query(query, (message_id,), fetch_one=True)
        if row:
            import json
            return json.loads(row["embedding"])
        return None

    def get_all_embeddings(self, channel_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all embeddings with message metadata."""
        if channel_filter:
            query = f"""
                SELECT m.id as message_id, m.content, m.author_name as author, 
                       m.channel_name as channel, m.timestamp, e.embedding
                FROM messages m
                JOIN embeddings e ON m.id = e.message_id
                WHERE m.channel_name = ? AND {self.base_filter}
                ORDER BY m.timestamp DESC
            """
            params = (channel_filter,)
        else:
            query = f"""
                SELECT m.id as message_id, m.content, m.author_name as author, 
                       m.channel_name as channel, m.timestamp, e.embedding
                FROM messages m
                JOIN embeddings e ON m.id = e.message_id
                WHERE {self.base_filter}
                ORDER BY m.timestamp DESC
            """
            params = ()

        results = self.db_manager.execute_query(query, params)
        embeddings = []
        
        for row in results:
            import json
            embedding = json.loads(row["embedding"])
            embeddings.append({
                "message_id": row["message_id"],
                "content": row["content"],
                "author": row["author"],
                "channel": row["channel"],
                "timestamp": row["timestamp"],
                "embedding": embedding,
            })
        
        return embeddings

    def get_messages_with_embeddings(
        self, channel_id: Optional[str] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get messages that have embeddings."""
        channel_filter = f"AND m.channel_name = '{channel_id}'" if channel_id else ""

        query = f"""
            SELECT m.id as message_id, m.content, m.author_name as author, 
                   m.channel_name as channel, m.timestamp, e.embedding
            FROM messages m
            JOIN embeddings e ON m.id = e.message_id
            WHERE m.content IS NOT NULL AND m.content != '' AND {self.base_filter}
            {channel_filter}
            ORDER BY m.timestamp DESC 
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (limit,))
        embeddings = []
        
        for row in results:
            import json
            embedding = json.loads(row["embedding"])
            embeddings.append({
                "message_id": row["message_id"],
                "content": row["content"],
                "author": row["author"],
                "channel": row["channel"],
                "timestamp": row["timestamp"],
                "embedding": embedding,
            })
        
        return embeddings

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        # Total messages
        total_query = f"SELECT COUNT(*) FROM messages WHERE {self.base_filter}"
        total_result = self.db_manager.execute_query(total_query, fetch_one=True)
        total_messages = total_result[0] if total_result else 0

        # Messages with embeddings
        embedded_query = "SELECT COUNT(*) FROM embeddings"
        embedded_result = self.db_manager.execute_query(embedded_query, fetch_one=True)
        embedded_messages = embedded_result[0] if embedded_result else 0

        return {
            "total_messages": total_messages,
            "embedded_messages": embedded_messages,
            "coverage_percentage": round(embedded_messages / total_messages * 100, 2)
            if total_messages > 0
            else 0,
        }

    def cleanup_orphaned_embeddings(self) -> int:
        """Clean up embeddings for messages that no longer exist."""
        query = """
            DELETE FROM embeddings
            WHERE message_id NOT IN (
                SELECT id FROM messages
            )
        """

        # Get count before deletion
        count_query = """
            SELECT COUNT(*) FROM embeddings
            WHERE message_id NOT IN (
                SELECT id FROM messages
            )
        """
        
        count_result = self.db_manager.execute_query(count_query, fetch_one=True)
        orphaned_count = count_result[0] if count_result else 0
        
        # Perform deletion
        self.db_manager.execute_query(query)
        
        return orphaned_count
