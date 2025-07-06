"""
Message Analyzer for weekly channel analysis.

Provides focused analysis of message patterns, engagement, and content.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict

from pepino.analysis.helpers.data_facade import AnalysisDataFacade
from pepino.logging_config import get_logger

logger = get_logger(__name__)

def to_utc(dt):
    if isinstance(dt, str):
        if dt.endswith("Z"):
            dt = dt.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

class MessageAnalyzer:
    """Analyzer for message-specific patterns and metrics."""
    
    def __init__(self, data_facade: AnalysisDataFacade):
        self.data_facade = data_facade
    
    def analyze_messages(self, channel_name: Optional[str] = None, days_back: int = 7, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze messages for the specified week period, for a single channel or all channels.
        
        Args:
            channel_name: Channel to analyze (None for all channels)
            days_back: Number of days to look back (default 7 for weekly)
            end_date: End date for analysis (default: current date)
        Returns:
            Dictionary with message analysis results
        """
        try:
            # Use provided end date or default to current date
            if end_date is None:
                end_date = to_utc(datetime.now())
            else:
                end_date = to_utc(end_date)
            start_date = end_date - timedelta(days=days_back)
            # Debug output
            logger.info(f"MessageAnalyzer: channel={channel_name}, end_date={end_date}, start_date={start_date}")
            # Get messages for the period using date range
            messages = self.data_facade.message_repository.get_messages_by_date_range(
                channel_name, start_date, end_date, limit=10000
            )
            if not messages:
                return self._get_empty_message_analysis(start_date, end_date, days_back)
            # Analyze message patterns
            message_stats = self._analyze_message_statistics(messages)
            activity_patterns = self._analyze_activity_patterns(messages)
            top_terms = self._extract_top_terms(messages)
            top_reaction_messages = self._get_top_reaction_messages(messages)
            if channel_name is None:
                # All channels: use the all-channels version for top commented messages
                top_commented_messages = self._get_top_commented_messages_all_channels(messages, start_date, end_date)
            else:
                top_commented_messages = self._get_top_commented_messages(messages)
            return {
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'days_back': days_back
                },
                'message_stats': message_stats,
                'activity_patterns': activity_patterns,
                'top_terms': top_terms,
                'top_reaction_messages': top_reaction_messages,
                'top_commented_messages': top_commented_messages,
                'total_messages': len(messages)
            }
        except Exception as e:
            logger.error(f"Error analyzing weekly messages for {channel_name if channel_name else 'all channels'}: {e}")
            return self._get_empty_message_analysis(start_date, end_date, days_back)
    
    def _analyze_message_statistics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze basic message statistics."""
        if not messages:
            return {}
        
        human_messages = [m for m in messages if not m.get('author_is_bot', False)]
        bot_messages = [m for m in messages if m.get('author_is_bot', False)]
        
        total_messages = len(messages)
        human_count = len(human_messages)
        bot_count = len(bot_messages)
        
        # Calculate activity rate (messages per day)
        if messages:
            timestamps = [to_utc(m['timestamp']) for m in messages]
            first_msg = min(timestamps)
            last_msg = max(timestamps)
            days_span = (last_msg - first_msg).days + 1
            activity_rate = total_messages / days_span if days_span > 0 else 0
        else:
            activity_rate = 0
        
        return {
            'total_messages': total_messages,
            'human_messages': human_count,
            'bot_messages': bot_count,
            'human_percentage': (human_count / total_messages * 100) if total_messages > 0 else 0,
            'bot_percentage': (bot_count / total_messages * 100) if total_messages > 0 else 0,
            'activity_rate': activity_rate,
            'avg_message_length': sum(len(m.get('content', '')) for m in messages) / total_messages if total_messages > 0 else 0
        }
    
    def _analyze_activity_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze activity patterns by hour and day."""
        if not messages:
            return {'peak_hours': [], 'peak_days': []}
        
        # Group by hour
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for message in messages:
            timestamp = to_utc(message['timestamp'])
            hour = timestamp.hour
            day = timestamp.strftime('%A')  # Monday, Tuesday, etc.
            
            hourly_counts[hour] += 1
            daily_counts[day] += 1
        
        # Get top 3 peak hours
        peak_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hours_formatted = [
            {
                'hour_range': f"{hour:02d}:00-{hour:02d}:59",
                'day': self._get_most_active_day_for_hour(messages, hour),
                'count': count
            }
            for hour, count in peak_hours
        ]
        
        # Get top 3 peak days
        peak_days = sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days_formatted = [
            {
                'day': day,
                'count': count
            }
            for day, count in peak_days
        ]
        
        return {
            'peak_hours': peak_hours_formatted,
            'peak_days': peak_days_formatted
        }
    
    def _get_most_active_day_for_hour(self, messages: List[Dict], target_hour: int) -> str:
        """Get the most active day for a specific hour."""
        day_counts = defaultdict(int)
        
        for message in messages:
            timestamp = to_utc(message['timestamp'])
            if timestamp.hour == target_hour:
                day = timestamp.strftime('%A')
                day_counts[day] += 1
        
        if day_counts:
            return max(day_counts.items(), key=lambda x: x[1])[0]
        return "Unknown"
    
    def _extract_top_terms(self, messages: List[Dict], top_n: int = 5) -> List[Dict]:
        """Extract top discussion terms from messages using NLP concepts."""
        if not messages:
            return []
        
        # Import NLP service
        from pepino.analysis.helpers.nlp_analyzer import NLPService
        
        try:
            nlp_service = NLPService()
            nlp_service.initialize()
            
            # Extract concepts from all messages
            all_concepts = []
            for message in messages:
                content = message.get('content', '')
                if content and content.strip():
                    concepts = nlp_service.extract_concepts(content)
                    all_concepts.extend(concepts)
            
            # Count concepts and get top N
            concept_counts = Counter(all_concepts)
            top_concepts = concept_counts.most_common(top_n)
            
            return [
                {
                    'term': concept,
                    'count': count
                }
                for concept, count in top_concepts
            ]
            
        except Exception as e:
            # Fallback to simple term extraction if NLP fails
            logger = get_logger(__name__)
            logger.warning(f"NLP concept extraction failed, falling back to simple terms: {e}")
            
            all_terms = []
            for message in messages:
                content = message.get('content', '').lower()
                # Split by common delimiters and filter out common words
                terms = [term.strip() for term in content.split() 
                        if len(term.strip()) > 3 and not term.startswith(('http', '@', '#', '<'))]
                all_terms.extend(terms)
            
            # Count terms and get top N
            term_counts = Counter(all_terms)
            top_terms = term_counts.most_common(top_n)
            
            return [
                {
                    'term': term,
                    'count': count
                }
                for term, count in top_terms
            ]
    
    def _get_top_reaction_messages(self, messages: List[Dict], top_n: int = 3) -> List[Dict]:
        """Get messages with the most reactions."""
        if not messages:
            return []
        
        # Filter messages with reactions
        messages_with_reactions = []
        for msg in messages:
            if msg.get('has_reactions', False):
                try:
                    # Try to get actual reaction count from reactions data
                    reactions = msg.get('reactions')
                    reaction_count = 0
                    
                    if reactions:
                        # Parse reactions JSON if it's a string
                        if isinstance(reactions, str):
                            import json
                            try:
                                reactions = json.loads(reactions)
                            except:
                                reactions = []
                        
                        # Calculate total reaction count
                        if reactions and isinstance(reactions, list):
                            reaction_count = sum(reaction.get('count', 0) for reaction in reactions)
                    
                    # If no reaction count available, use has_reactions as proxy (count as 1)
                    if reaction_count == 0 and msg.get('has_reactions', False):
                        reaction_count = 1
                    
                    if reaction_count > 0:
                        messages_with_reactions.append({
                            'message_id': msg.get('id'),
                            'content': msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', ''),
                            'author': msg.get('author_name', 'Unknown'),
                            'reaction_count': reaction_count,
                            'timestamp': msg.get('timestamp')
                        })
                except Exception as e:
                    logger.warning(f"Error parsing reactions for message {msg.get('id')}: {e}")
                    continue
        
        # Sort by reaction count and get top N
        top_messages = sorted(messages_with_reactions, key=lambda x: x['reaction_count'], reverse=True)[:top_n]
        
        return top_messages
    
    def _get_top_commented_messages(self, messages: List[Dict], top_n: int = 3) -> List[Dict]:
        """Get messages with the most replies/comments using the repository method for accuracy."""
        if not messages:
            return []
        # Use the date range from the messages
        timestamps = [to_utc(m['timestamp']) for m in messages]
        start_date = min(timestamps)
        end_date = max(timestamps)
        channel_name = messages[0].get('channel_name') if messages else None
        # Always use the current channel to ensure links are valid
        return self.data_facade.message_repository.get_top_commented_messages(
            channel_name, start_date, end_date, top_n=top_n
        )
    
    def _get_top_commented_messages_all_channels(self, messages: List[Dict], start_date: datetime, end_date: datetime, top_n: int = 3) -> List[Dict]:
        """Get messages with the most replies/comments across all channels."""
        if not messages:
            return []
        # Use None for channel_name to get top commented messages across all channels
        return self.data_facade.message_repository.get_top_commented_messages(
            None, start_date, end_date, top_n=top_n
        )
    
    def _get_empty_message_analysis(self, start_date: datetime, end_date: datetime, days_back: int) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'days_back': days_back
            },
            'message_stats': {
                'total_messages': 0,
                'human_messages': 0,
                'bot_messages': 0,
                'human_percentage': 0,
                'bot_percentage': 0,
                'activity_rate': 0,
                'avg_message_length': 0
            },
            'activity_patterns': {
                'peak_hours': [],
                'peak_days': []
            },
            'top_terms': [],
            'top_reaction_messages': [],
            'top_commented_messages': [],
            'total_messages': 0
        } 