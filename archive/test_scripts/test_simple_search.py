#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import DiscordBotAnalyzer

async def test_simple_search():
    analyzer = DiscordBotAnalyzer()
    
    try:
        await analyzer.initialize()
        print("Analyzer initialized")
        
        # Very simple search - just find by display name
        print("Searching for 'Arturo Cuevas' by display name...")
        
        async with analyzer.pool.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE author_display_name = ?
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Search result: {user}")
        
        if user:
            user_id, username = user
            print(f"Found user: {username} (ID: {user_id})")
            
            # Get basic message count
            async with analyzer.pool.execute("""
                SELECT COUNT(*) 
                FROM messages 
                WHERE author_id = ?
            """, (user_id,)) as cursor:
                count_result = await cursor.fetchone()
                message_count = count_result[0] if count_result else 0
                print(f"Message count: {message_count}")
                
            return f"SUCCESS: Found {username} with {message_count} messages"
        else:
            return "FAILED: User not found"
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"
    finally:
        await analyzer.close()

if __name__ == "__main__":
    result = asyncio.run(test_simple_search())
    print(f"Final result: {result}")
