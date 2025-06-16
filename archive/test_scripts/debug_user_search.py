#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import DiscordBotAnalyzer

async def test_user_search():
    analyzer = DiscordBotAnalyzer()
    
    try:
        await analyzer.initialize()
        print("Analyzer initialized successfully")
        
        # Test exact display name query manually
        print("\nTesting exact display name query manually:")
        async with analyzer.pool.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE author_display_name = ?
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Manual query result: {user}")
        
        # Test the actual method
        print("\nTesting full method:")
        result = await analyzer.get_user_insights("Arturo Cuevas")
        print("Result:")
        print(result[:500] + "..." if len(result) > 500 else result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(test_user_search())
