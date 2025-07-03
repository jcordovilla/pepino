"""
User Analyzer 

Synchronous user analysis using the data facade pattern for repository management.
Provides comprehensive user activity analysis with proper separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from .models import UserAnalysisResponse, LocalUserStatistics, ChannelActivity
from .models import EnhancedUserAnalysisResponse, EnhancedUserStatistics, EnhancedChannelActivity, TimeOfDayActivity, SemanticAnalysisResult

logger = logging.getLogger(__name__)


class UserAnalyzer:
    """
    Synchronous User Analyzer 
    
    Analyzes user activity patterns, channel engagement, and messaging behavior
    using the data facade pattern for centralized repository management.
    
    Features:
    - Message count analysis through data facade layer
    - Channel activity distribution
    - Time-based activity patterns
    - User engagement metrics
    - Cross-channel behavior analysis
    
    All database operations are abstracted through the data facade for proper
    separation of concerns and dependency injection support.
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize user analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("UserAnalyzer initialized with data facade pattern")
    
    def analyze(
        self, 
        username: str,
        days: Optional[int] = None,
        include_patterns: bool = True
    ) -> Optional[UserAnalysisResponse]:
        """
        Analyze a user comprehensively using repository layer.
        
        Args:
            username: Name of the user to analyze
            days: Number of days to look back (None = all time)
            include_patterns: Whether to include time patterns analysis
            
        Returns:
            UserAnalysisResponse object with comprehensive results
        """
        
        try:
            logger.info(f"Starting user analysis for: {username}")
            
            # Get basic statistics through repository
            statistics = self._get_user_statistics_via_repository(username, days)
            if not statistics or statistics.total_messages == 0:
                logger.warning(f"No messages found for user: {username}")
                return None
            
            # Get channel activity through repository
            channel_activity = self._get_user_channel_activity_via_repository(username, days)
            
            # Get time patterns if requested
            time_patterns = {}
            if include_patterns:
                time_patterns = self._analyze_time_patterns_via_repository(username, days)
            
            # Create summary
            summary = self._create_user_summary(statistics, channel_activity, time_patterns)
            
            # Build result
            from .models import UserInfo
            
            analysis = UserAnalysisResponse(
                user_info=UserInfo(
                    author_id=username,  # Use username as ID for now
                    display_name=username
                ),
                statistics=self._convert_to_user_statistics_model(statistics, username)
            )
            
            logger.info(f"User analysis completed for: {username}")
            return analysis
            
        except Exception as e:
            logger.error(f"User analysis failed for {username}: {e}")
            return None
    
    def analyze_enhanced(
        self, 
        username: str,
        days: Optional[int] = None,
        include_semantic: bool = True
    ) -> Optional[EnhancedUserAnalysisResponse]:
        """
        Perform enhanced comprehensive user analysis.
        
        Args:
            username: Name of the user to analyze
            days: Number of days to look back (None = all time)
            include_semantic: Whether to include semantic analysis
            
        Returns:
            EnhancedUserAnalysisResponse object with comprehensive results
        """
        
        try:
            logger.info(f"Starting enhanced user analysis for: {username}")
            
            # Get enhanced statistics through repository
            statistics = self._get_enhanced_user_statistics_via_repository(username, days)
            if not statistics or statistics.message_count == 0:
                logger.warning(f"No messages found for user: {username}")
                return None
            
            # Get enhanced channel activity
            channel_activity = self._get_enhanced_channel_activity_via_repository(username, days)
            
            # Get time of day patterns
            time_patterns = self._get_time_patterns_via_repository(username, days)
            
            # Get semantic analysis if requested
            semantic_analysis = None
            if include_semantic:
                semantic_analysis = self._get_semantic_analysis_via_repository(username, days)
            
            # Get user's top topics
            top_topics = self._get_user_topics_via_repository(username, days)
            
            # Build result
            from .models import UserInfo
            
            analysis = EnhancedUserAnalysisResponse(
                user_info=UserInfo(
                    author_id=username,  # Use username as ID for now
                    display_name=statistics.display_name or username
                ),
                statistics=statistics,
                channel_activity=channel_activity,
                time_patterns=time_patterns,
                semantic_analysis=semantic_analysis,
                top_topics=top_topics
            )
            
            logger.info(f"Enhanced user analysis completed for: {username}")
            return analysis
            
        except Exception as e:
            logger.error(f"Enhanced user analysis failed for {username}: {e}")
            return None
    
    def _convert_to_user_statistics_model(self, stats: LocalUserStatistics, username: str):
        """Convert our LocalUserStatistics to the model expected by UserAnalysisResponse."""
        from .models import UserStatistics as ModelUserStatistics
        
        return ModelUserStatistics(
            author_id=username,
            author_name=username,
            message_count=stats.total_messages,
            channels_active=stats.unique_channels,
            avg_message_length=stats.avg_message_length,
            first_message_date=stats.first_message_date,
            last_message_date=stats.last_message_date
        )
    
    def _get_user_statistics_via_repository(self, username: str, days: Optional[int]) -> Optional[LocalUserStatistics]:
        """Get basic user statistics using data facade pattern."""
        
        try:
            # Use data facade for repository access
            stats_data = self.data_facade.user_repository.get_user_message_statistics(username, days)
            
            if not stats_data or stats_data.get('total_messages', 0) == 0:
                return None
            
            # Calculate messages per day
            total_messages = stats_data['total_messages']
            if total_messages > 0 and stats_data['first_message'] and stats_data['last_message']:
                first_date = datetime.fromisoformat(stats_data['first_message'].replace('Z', '+00:00'))
                last_date = datetime.fromisoformat(stats_data['last_message'].replace('Z', '+00:00'))
                time_span = (last_date - first_date).days + 1
                messages_per_day = total_messages / max(time_span, 1)
            else:
                messages_per_day = 0
            
            return LocalUserStatistics(
                total_messages=stats_data['total_messages'],
                unique_channels=stats_data['unique_channels'],
                messages_per_day=round(messages_per_day, 2),
                avg_message_length=stats_data.get('avg_message_length', 0.0),
                first_message_date=stats_data['first_message'],
                last_message_date=stats_data['last_message'],
                analysis_period_days=days
            )
            
        except Exception as e:
            logger.error(f"Failed to get user statistics via repository: {e}")
            return None
    
    def _get_user_channel_activity_via_repository(
        self, 
        username: str, 
        days: Optional[int],
        limit: int = 20
    ) -> List[ChannelActivity]:
        """Get user's activity distribution across channels using data facade."""
        
        try:
            # Use data facade for repository access
            activity_data = self.data_facade.user_repository.get_user_channel_activity(username, days, limit)
            
            channel_activities = []
            for data in activity_data:
                activity = ChannelActivity(
                    channel_name=data['channel_name'],
                    message_count=data['message_count'],
                    first_message=data['first_message'],
                    last_message=data['last_message']
                )
                channel_activities.append(activity)
            
            return channel_activities
            
        except Exception as e:
            logger.error(f"Failed to get user channel activity via repository: {e}")
            return []

    def _analyze_time_patterns_via_repository(self, username: str, days: Optional[int]) -> Dict[str, any]:
        """Analyze user time patterns using data facade."""
        
        try:
            # Get hourly patterns through data facade
            hourly_data = self.data_facade.user_repository.get_user_hourly_patterns(username, days)
            
            # Process hourly data
            hourly_activity = {}
            peak_hour = 0
            max_messages = 0
            
            for entry in hourly_data:
                hour = entry['hour']
                count = entry['message_count']
                hourly_activity[hour] = count
                
                if count > max_messages:
                    max_messages = count
                    peak_hour = hour
            
            # Determine activity period
            if peak_hour >= 6 and peak_hour < 12:
                activity_period = "morning"
            elif peak_hour >= 12 and peak_hour < 18:
                activity_period = "afternoon"
            elif peak_hour >= 18 and peak_hour < 22:
                activity_period = "evening"
            else:
                activity_period = "night"
            
            return {
                'hourly_activity': hourly_activity,
                'peak_hour': peak_hour,
                'activity_period': activity_period,
                'total_active_hours': len(hourly_activity)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze time patterns via repository: {e}")
            return {}
    
    def _create_user_summary(
        self,
        statistics: LocalUserStatistics,
        channel_activity: List[ChannelActivity],
        time_patterns: Dict[str, any]
    ) -> Dict[str, any]:
        """Create summary of user analysis results."""
        
        # Channel diversity
        if statistics.unique_channels > 5:
            channel_diversity = "high"
        elif statistics.unique_channels > 2:
            channel_diversity = "medium"
        else:
            channel_diversity = "low"
        
        # Activity level
        if statistics.messages_per_day > 10:
            activity_level = "very active"
        elif statistics.messages_per_day > 5:
            activity_level = "active"
        elif statistics.messages_per_day > 1:
            activity_level = "moderate"
        else:
            activity_level = "low"
        
        # Most active channel
        most_active_channel = None
        if channel_activity:
            most_active_channel = channel_activity[0].channel_name
        
        # Time preference
        time_preference = time_patterns.get('activity_period', 'unknown')
        
        return {
            'activity_level': activity_level,
            'channel_diversity': channel_diversity,
            'most_active_channel': most_active_channel,
            'time_preference': time_preference,
            'total_channels': statistics.unique_channels,
            'avg_daily_messages': statistics.messages_per_day
        }

    def get_available_users(self) -> List[str]:
        """Get list of available users using data facade."""
        try:
            return self.data_facade.user_repository.get_available_users()
        except Exception as e:
            logger.error(f"Failed to get available users: {e}")
            return []

    def get_top_users(self, limit: int = 10, days: Optional[int] = None) -> List[Dict[str, any]]:
        """Get top users by message count using data facade."""
        try:
            return self.data_facade.user_repository.get_top_users_by_message_count(limit, days)
        except Exception as e:
            logger.error(f"Failed to get top users: {e}")
            return []

    def get_user_health(self, username: str) -> Dict[str, any]:
        """Get user engagement health metrics using repository."""
        
        try:
            # Get basic stats
            recent_stats = self.user_repository.get_user_message_statistics(username, days=7)
            overall_stats = self.user_repository.get_user_message_statistics(username, days=None)
            
            if not overall_stats or overall_stats.get('total_messages', 0) == 0:
                return {
                    'status': 'inactive',
                    'health_score': 0,
                    'recommendations': ['User has no message history']
                }
            
            # Calculate health metrics
            recent_messages = recent_stats.get('total_messages', 0)
            total_messages = overall_stats.get('total_messages', 0)
            channels = overall_stats.get('unique_channels', 0)
            
            # Health score calculation
            health_score = 0
            
            # Recent activity (30% weight)
            if recent_messages > 10:
                health_score += 30
            elif recent_messages > 5:
                health_score += 20
            elif recent_messages > 0:
                health_score += 10
            
            # Channel diversity (30% weight)
            if channels > 5:
                health_score += 30
            elif channels > 2:
                health_score += 20
            elif channels > 0:
                health_score += 10
            
            # Overall engagement (40% weight)
            if total_messages > 1000:
                health_score += 40
            elif total_messages > 100:
                health_score += 30
            elif total_messages > 10:
                health_score += 20
            else:
                health_score += 10
            
            # Determine status
            if health_score >= 80:
                status = 'excellent'
            elif health_score >= 60:
                status = 'good'
            elif health_score >= 40:
                status = 'fair'
            else:
                status = 'needs_attention'
            
            # Generate recommendations
            recommendations = []
            if recent_messages == 0:
                recommendations.append('Consider encouraging recent participation')
            if channels <= 2:
                recommendations.append('User could benefit from engaging in more channels')
            if total_messages < 50:
                recommendations.append('New user - consider welcoming initiatives')
            
            return {
                'status': status,
                'health_score': health_score,
                'recent_activity': recent_messages,
                'total_activity': total_messages,
                'channel_diversity': channels,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to get user health for {username}: {e}")
            return {
                'status': 'error',
                'health_score': 0,
                'recommendations': [f'Error analyzing user: {e}']
            }

    def get_user_channel_comparison(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> Dict[str, any]:
        """Compare user's activity across different channels using repository."""
        
        try:
            channel_activity = self.user_repository.get_user_channel_activity(username, days, limit=50)
            
            if not channel_activity:
                return {
                    'total_channels': 0,
                    'channels': [],
                    'primary_channel': None,
                    'diversity_score': 0
                }
            
            total_messages = sum(activity['message_count'] for activity in channel_activity)
            
            # Calculate percentages and create comparison
            channels = []
            for activity in channel_activity:
                percentage = (activity['message_count'] / total_messages * 100) if total_messages > 0 else 0
                channels.append({
                    'name': activity['channel_name'],
                    'messages': activity['message_count'],
                    'percentage': round(percentage, 1),
                    'first_message': activity['first_message'],
                    'last_message': activity['last_message']
                })
            
            # Primary channel (highest activity)
            primary_channel = channels[0]['name'] if channels else None
            
            # Diversity score (1 - Herfindahl index)
            if total_messages > 0:
                herfindahl = sum((activity['message_count'] / total_messages) ** 2 for activity in channel_activity)
                diversity_score = round((1 - herfindahl) * 100, 1)
            else:
                diversity_score = 0
            
            return {
                'total_channels': len(channels),
                'channels': channels,
                'primary_channel': primary_channel,
                'diversity_score': diversity_score,
                'total_messages': total_messages
            }
            
        except Exception as e:
            logger.error(f"Failed to get user channel comparison for {username}: {e}")
            return {
                'total_channels': 0,
                'channels': [],
                'primary_channel': None,
                'diversity_score': 0
            }

    def _get_enhanced_user_statistics_via_repository(self, username: str, days: Optional[int]) -> Optional[EnhancedUserStatistics]:
        """Get enhanced user statistics using data facade pattern."""
        
        try:
            # Use data facade for repository access
            stats_data = self.data_facade.user_repository.get_user_enhanced_statistics(username, days)
            
            if not stats_data or stats_data.get('total_messages', 0) == 0:
                return None
            
            # Get display name from a sample message with better query
            display_name = None
            try:
                sample_query = """
                SELECT author_display_name 
                FROM messages 
                WHERE author_name = ? 
                AND author_display_name IS NOT NULL 
                AND author_display_name != ''
                AND author_display_name != author_name
                ORDER BY timestamp DESC
                LIMIT 1
                """
                result = self.data_facade.user_repository.db_manager.execute_query(
                    sample_query, (username,), fetch_one=True
                )
                if result and result['author_display_name']:
                    display_name = result['author_display_name']
                else:
                    # Fallback: get any display name for this user
                    fallback_query = """
                    SELECT author_display_name 
                    FROM messages 
                    WHERE author_name = ? 
                    AND author_display_name IS NOT NULL 
                    AND author_display_name != ''
                    LIMIT 1
                    """
                    fallback_result = self.data_facade.user_repository.db_manager.execute_query(
                        fallback_query, (username,), fetch_one=True
                    )
                    if fallback_result and fallback_result['author_display_name']:
                        display_name = fallback_result['author_display_name']
            except Exception as e:
                logger.warning(f"Could not retrieve display name for {username}: {e}")
            
            # Use display name if found, otherwise use username
            final_display_name = display_name if display_name else username
            
            return EnhancedUserStatistics(
                author_id=username,
                author_name=username,
                display_name=final_display_name,
                message_count=stats_data['total_messages'],
                channels_active=stats_data['unique_channels'],
                active_days=stats_data['active_days'],
                avg_message_length=stats_data.get('avg_message_length', 0.0),
                first_message_date=stats_data['first_message'],
                last_message_date=stats_data['last_message']
            )
            
        except Exception as e:
            logger.error(f"Failed to get enhanced user statistics via repository: {e}")
            return None

    def _get_enhanced_channel_activity_via_repository(
        self, 
        username: str, 
        days: Optional[int],
        limit: int = 10
    ) -> List[EnhancedChannelActivity]:
        """Get enhanced user's activity distribution across channels using data facade."""
        
        try:
            # Use data facade for repository access
            activity_data = self.data_facade.user_repository.get_user_channel_activity_enhanced(username, days, limit)
            
            channel_activities = []
            for data in activity_data:
                activity = EnhancedChannelActivity(
                    channel_name=data['channel_name'],
                    message_count=data['message_count'],
                    avg_message_length=data.get('avg_message_length', 0.0),
                    first_message_date=data['first_message'],
                    last_message_date=data['last_message']
                )
                channel_activities.append(activity)
            
            return channel_activities
            
        except Exception as e:
            logger.error(f"Failed to get enhanced channel activity via repository: {e}")
            return []

    def _get_time_patterns_via_repository(self, username: str, days: Optional[int]) -> List[TimeOfDayActivity]:
        """Get time of day patterns using data facade."""
        
        try:
            # Use data facade for repository access
            time_data = self.data_facade.user_repository.get_user_time_of_day_patterns(username, days)
            
            time_patterns = []
            for period, count in time_data.items():
                pattern = TimeOfDayActivity(
                    period=period,
                    message_count=count
                )
                time_patterns.append(pattern)
            
            # Sort by message count (most active first)
            time_patterns.sort(key=lambda x: x.message_count, reverse=True)
            
            return time_patterns
            
        except Exception as e:
            logger.error(f"Failed to get time patterns via repository: {e}")
            return []

    def _get_semantic_analysis_via_repository(self, username: str, days: Optional[int]) -> Optional[SemanticAnalysisResult]:
        """Get semantic analysis of user content using NLP analyzer."""
        
        try:
            # Temporarily disable semantic analysis to avoid method conflicts
            # TODO: Fix method signature conflict between sync and async get_user_content_sample
            logger.info("Semantic analysis temporarily disabled")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get semantic analysis via repository: {e}")
            return None

    def _fallback_semantic_analysis(self, content: str) -> SemanticAnalysisResult:
        """Fallback semantic analysis using simple keyword extraction."""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', content.lower())
        word_freq = Counter(words)
        
        # Filter common words
        stop_words = {'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this', 'have', 'from', 'not', 'but', 'can', 'all', 'any', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say'}
        
        # Common tech terms
        tech_terms = []
        tech_keywords = {'ai', 'api', 'data', 'model', 'algorithm', 'code', 'system', 'tech', 'development', 'software', 'programming', 'machine', 'learning', 'neural', 'network'}
        
        for word, count in word_freq.most_common(100):
            if word in tech_keywords and count >= 2:
                tech_terms.append(word.upper())
        
        # Get most common meaningful words as concepts
        concepts = []
        for word, count in word_freq.most_common(50):
            if word not in stop_words and len(word) > 3 and count >= 2:
                concepts.append(word.title())
        
        return SemanticAnalysisResult(
            key_entities=[],  # No entity extraction in fallback
            technology_terms=tech_terms[:15],
            key_concepts=concepts[:20]
        )

    def _get_user_topics_via_repository(self, username: str, days: Optional[int]) -> List['TopicItem']:
        """Get user's most common topics using TopicAnalyzer."""
        
        try:
            from .models import TopicItem
            from .topic_analyzer import TopicAnalyzer
            
            # Get user's message content (using direct query to avoid method conflict)
            user_messages = self._get_user_messages_for_topics(username, days, 500)
            
            if not user_messages or len(user_messages) < 5:
                logger.info(f"Not enough messages for topic analysis for user {username}")
                return []
            
            # Initialize topic analyzer
            topic_analyzer = TopicAnalyzer()
            topic_analyzer.data_facade = self.data_facade  # Pass data facade for consistency
            
            # Extract topics from user messages
            logger.info(f"Extracting topics from {len(user_messages)} messages for user {username}")
            bertopic_results, domain_analysis = topic_analyzer.extract_topics(
                user_messages, 
                min_topic_size=2  # Allow smaller clusters for individual users
            )
            
            # Convert to TopicItem models
            topic_items = []
            for topic_data in bertopic_results[:10]:  # Top 10 topics
                topic_item = TopicItem(
                    topic=topic_data["topic"],
                    frequency=topic_data["frequency"],
                    relevance_score=topic_data["relevance_score"]
                )
                topic_items.append(topic_item)
            
            logger.info(f"Found {len(topic_items)} topics for user {username}")
            return topic_items
            
        except Exception as e:
            logger.error(f"Failed to get user topics for {username}: {e}")
            # Fallback to simple keyword-based topics
            return self._fallback_user_topics(username, days)

    def _fallback_user_topics(self, username: str, days: Optional[int]) -> List['TopicItem']:
        """Fallback topic analysis using simple keyword extraction."""
        
        try:
            from .models import TopicItem
            from collections import Counter
            import re
            
            # Get user's message content (using direct query to avoid method conflict)
            user_messages = self._get_user_messages_for_topics(username, days, 200)
            
            if not user_messages:
                return []
            
            # Combine all messages into one text
            all_text = " ".join(user_messages).lower()
            
            # Extract meaningful words (3+ characters, not common words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            
            # Filter out common stop words
            stop_words = {
                'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this', 
                'have', 'from', 'not', 'but', 'can', 'all', 'any', 'had', 'her', 'one', 'our', 
                'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say', 'its', 
                'two', 'more', 'very', 'what', 'know', 'just', 'first', 'also', 'after', 
                'back', 'other', 'many', 'than', 'then', 'them', 'these', 'some', 'her', 
                'would', 'make', 'like', 'into', 'him', 'has', 'more', 'go', 'no', 'so', 
                'up', 'out', 'if', 'about', 'who', 'oil', 'sit', 'set'
            }
            
            # Count word frequencies
            word_freq = Counter(word for word in words if word not in stop_words and len(word) > 3)
            
            # Convert to TopicItem models
            topic_items = []
            total_words = sum(word_freq.values())
            
            for word, count in word_freq.most_common(10):
                if count >= 2:  # Must appear at least twice
                    relevance_score = count / total_words if total_words > 0 else 0
                    topic_item = TopicItem(
                        topic=word.title(),
                        frequency=count,
                        relevance_score=round(relevance_score, 3)
                    )
                    topic_items.append(topic_item)
            
            logger.info(f"Fallback topic analysis found {len(topic_items)} topics for user {username}")
            return topic_items
            
        except Exception as e:
            logger.error(f"Fallback topic analysis failed for {username}: {e}")
            return []

    def _get_user_messages_for_topics(self, username: str, days: Optional[int], limit: int) -> List[str]:
        """Get user messages for topic analysis (avoiding method name conflict)."""
        try:
            query = """
            SELECT content
            FROM messages 
            WHERE author_name = ?
            """
            
            params = [username]
            
            if days:
                query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
                params.append(days)
            
            query += """
            AND content IS NOT NULL
            AND LENGTH(content) > 10
            ORDER BY RANDOM()
            LIMIT ?
            """
            params.append(limit)
            
            results = self.data_facade.user_repository.db_manager.execute_query(query, tuple(params))
            
            return [row['content'] for row in results] if results else []
            
        except Exception as e:
            logger.error(f"Failed to get user messages for topics: {e}")
            return [] 