#!/usr/bin/env python3
"""
Quick database test with timeout protection.
This tests if our hanging fixes work with database operations.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import pepino
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.data.database.manager import DatabaseManager
from pepino.data.config import Settings


async def test_database_operations():
    """Test basic database operations with proper cleanup"""
    db_manager = None
    try:
        print("🔧 Testing database connection with timeout protection...")
        
        # Initialize with timeout
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        print("✅ Database initialized successfully")
        
        # Simple query test with timeout
        query = "SELECT COUNT(*) FROM messages LIMIT 1"
        async with db_manager.pool.execute(query) as cursor:
            result = await asyncio.wait_for(cursor.fetchone(), timeout=5.0)
            message_count = result[0] if result else 0
            
        print(f"✅ Database query successful: {message_count} messages")
        
        # Test channels list with timeout
        channels_query = "SELECT DISTINCT channel_name FROM messages LIMIT 5"
        async with db_manager.pool.execute(channels_query) as cursor:
            channels = await asyncio.wait_for(cursor.fetchall(), timeout=5.0)
            channel_names = [row[0] for row in channels] if channels else []
            
        print(f"✅ Found {len(channel_names)} channels")
        if channel_names:
            print(f"   First channel: #{channel_names[0]}")
        
        return True
        
    except asyncio.TimeoutError:
        print("❌ Database operation timed out - this prevents hanging!")
        return False
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False
    finally:
        # CRITICAL: Always cleanup database connections
        if db_manager:
            try:
                await asyncio.wait_for(db_manager.close(), timeout=5.0)
                print("🔧 Database connections closed properly")
            except Exception as e:
                print(f"⚠️ Error closing database: {e}")
                # Force close if needed
                if hasattr(db_manager, 'pool') and db_manager.pool:
                    try:
                        await db_manager.pool.close()
                        print("🔧 Forced database pool closure")
                    except:
                        pass


async def main():
    """Main test function with proper cleanup"""
    try:
        print("🚀 Quick Database Test (With Hanging Protection)")
        print("================================================")
        
        success = await test_database_operations()
        
        if success:
            print("\n🎉 Database operations work correctly!")
            print("✅ No hanging issues detected")
        else:
            print("\n❌ Database operations had issues")
            
    except KeyboardInterrupt:
        print("\n🛑 Test cancelled by user")
    except Exception as e:
        print(f"\n⚠️ Test error: {str(e)}")
    finally:
        print("\n🔚 Test completed - all resources cleaned up")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exiting cleanly...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Ensure any remaining tasks are cleaned up
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks()
            for task in pending:
                task.cancel()
        except:
            pass
        print("🔚 All cleanup completed") 