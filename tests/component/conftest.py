import os
import shutil
import pytest
import sqlite3
from datetime import datetime, timedelta
from src.pepino.data.database.schema import init_database

# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_artifacts(request):
#     """Cleanup test artifacts after all component tests run."""
#     def remove_artifacts():
#         # Remove generated test databases and data
#         for fname in ["test_database.db", "test_data.json", "analysis_validation_results.json", "discord_messages.db", "comprehensive_test.db"]:
#             try:
#                 os.remove(os.path.join(os.path.dirname(__file__), "..", fname))
#             except FileNotFoundError:
#                 pass
#         # Remove logs
#         logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
#         if os.path.exists(logs_dir):
#             for f in os.listdir(logs_dir):
#                 try:
#                     os.remove(os.path.join(logs_dir, f))
#                 except Exception:
#                     pass
#         # Remove __pycache__
#         pycache_dir = os.path.join(os.path.dirname(__file__), "..", "__pycache__")
#         if os.path.exists(pycache_dir):
#             shutil.rmtree(pycache_dir, ignore_errors=True)
#     request.addfinalizer(remove_artifacts)


@pytest.fixture
def comprehensive_test_db():
    """Create a comprehensive test database with known quantities for precise validation."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "comprehensive_test.db")
    
    print(f"DEBUG: Creating database at {db_path}")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Use the real schema initializer
    init_database(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"DEBUG: Database connection created successfully")
    
    # Insert known test data with precise quantities
    # Use recent UTC timestamps so messages fall within the 30-day analysis window
    base_time = datetime.utcnow() - timedelta(days=1)
    
    # Test channels with exact message counts to match expected output
    test_channels = [
        ("🏘old-general-chat", 166),  # 166 messages (146 + 20 recent)
        ("🦾agent-ops", 70),          # 70 messages
        ("🏛netarch-general", 46),    # 46 messages
        ("🛝discord-pg", 27),         # 27 messages
        ("jose-test", 22),            # 22 messages
    ]
    
    # Test users with known message counts
    test_users = [
        ("oscarsan.chez", 80),        # 80 messages
        ("jose.chez", 60),            # 60 messages
        ("alice.test", 45),           # 45 messages
        ("bob.test", 35),             # 35 messages
        ("charlie.test", 20),         # 20 messages
        ("diana.test", 15),           # 15 messages
        ("eve.test", 10),             # 10 messages
    ]
    
    # Create messages with exact counts per channel
    message_id = 1
    for channel_name, total_messages in test_channels:
        # Distribute messages among users proportionally
        total_user_messages = sum(u[1] for u in test_users)
        messages_created = 0
        
        for user_name, user_messages in test_users:
            if messages_created >= total_messages:
                break
                
            # Calculate how many messages this user should have in this channel
            user_ratio = user_messages / total_user_messages
            channel_messages_for_user = int(total_messages * user_ratio)
            
            # Ensure we don't exceed the total for this channel
            remaining_messages = total_messages - messages_created
            channel_messages_for_user = min(channel_messages_for_user, remaining_messages)
            
            # Ensure at least 1 message if user has activity
            if channel_messages_for_user == 0 and user_messages > 0 and messages_created < total_messages:
                channel_messages_for_user = 1
            
            for i in range(channel_messages_for_user):
                message_time = base_time + timedelta(hours=message_id)
                cursor.execute("""
                    INSERT INTO messages (
                        id, content, timestamp, edited_timestamp, jump_url,
                        author_id, author_name, author_discriminator, author_display_name, author_is_bot, author_avatar_url, author_accent_color, author_banner_url, author_color, author_created_at, author_default_avatar_url, author_public_flags, author_system, author_verified,
                        author_status, author_activity, author_desktop_status, author_mobile_status, author_web_status,
                        guild_id, guild_name, guild_member_count, guild_description, guild_icon_url, guild_banner_url, guild_splash_url, guild_discovery_splash_url, guild_features, guild_verification_level, guild_explicit_content_filter, guild_mfa_level, guild_premium_tier, guild_premium_subscription_count,
                        channel_id, channel_name, channel_type, channel_topic, channel_nsfw, channel_position, channel_slowmode_delay, channel_category_id, channel_overwrites,
                        thread_id, thread_name, thread_archived, thread_auto_archive_duration, thread_locked, thread_member_count, thread_message_count, thread_owner_id, thread_parent_id, thread_slowmode_delay,
                        mentions, mention_everyone, mention_roles, mention_channels, referenced_message_id, referenced_message,
                        attachments, embeds,
                        reactions, emoji_stats,
                        pinned, flags, nonce, type, is_system, mentions_everyone, has_reactions,
                        components, interaction,
                        stickers, role_subscription_data,
                        application_id, application,
                        activity,
                        position, role_subscription_listing_id, webhook_id, tts, suppress_embeds, allowed_mentions, message_reference,
                        has_attachments, has_embeds, has_stickers, has_mentions, has_reference, is_webhook,
                        created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?,
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?,
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                """, (
                    f"msg_{message_id}",
                    f"Test message {message_id} by {user_name} in {channel_name}",
                    message_time.isoformat(),
                    None,  # edited_timestamp
                    None,  # jump_url
                    f"user_{user_name}",
                    user_name,
                    None,  # author_discriminator
                    user_name,
                    0,  # author_is_bot
                    None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None,
                    "guild1",
                    "Test Guild",
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    channel_name,               # channel_id - use real name, no prefix
                    channel_name,               # channel_name - real name, no prefix
                    "text",
                    None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None,
                    None, None,
                    None, None,
                    None, None, None, None, None, None, None,
                    None, None,
                    None, 
                    None, None, None, None, None, 
                    None, None, None, None, None, None,
                    0, 0, 0, 0, 0, 0, message_time.isoformat()
                ))
                message_id += 1
                messages_created += 1

    conn.commit()
    
    # Debug: Check if messages were actually inserted
    cursor.execute("SELECT COUNT(*) FROM messages")
    message_count = cursor.fetchone()[0]
    print(f"DEBUG: Messages inserted: {message_count}")
    
    # Debug: Check a few sample messages
    cursor.execute("SELECT id, content, channel_name FROM messages LIMIT 3")
    sample_messages = cursor.fetchall()
    print(f"DEBUG: Sample messages: {sample_messages}")
    
    conn.close()
    
    print(f"DEBUG: Database created successfully, size: {os.path.getsize(db_path)} bytes")
    
    yield db_path
    
    # Cleanup - COMMENTED OUT FOR DEBUGGING
    # if os.path.exists(db_path):
    #     os.remove(db_path)


@pytest.fixture
def expected_counts():
    """Expected counts for comprehensive test data validation."""
    return {
        'channels': {
            '🏘old-general-chat': 166,  # Exact count to match expected output
            '🦾agent-ops': 70,
            '🏛netarch-general': 46,
            '🛝discord-pg': 27,
            'jose-test': 22,
        },
        'users': {
            'oscarsan.chez': 80,
            'jose.chez': 60,
            'alice.test': 45,
            'bob.test': 35,
            'charlie.test': 20,
            'diana.test': 15,
            'eve.test': 10,
        },
        'total_messages': 331,  # Sum of all channel messages (166+70+46+27+22)
        'total_users': 7,
        'total_channels': 5,
    } 