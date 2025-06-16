#!/usr/bin/env python3
import asyncio
import aiosqlite

async def test_minimal_search():
    """Minimal test to isolate the display name search issue"""
    
    # Direct aiosqlite test
    print("=== Testing direct aiosqlite ===")
    async with aiosqlite.connect('discord_messages.db') as db:
        async with db.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE author_display_name = ?
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Direct aiosqlite result: {user}")
    
    # Test the analyzer initialization
    print("\n=== Testing analyzer pool ===")
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from analysis import DiscordBotAnalyzer
    
    analyzer = DiscordBotAnalyzer()
    await analyzer.initialize()
    
    # Test using analyzer pool
    try:
        async with analyzer.pool.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE author_display_name = ?
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Analyzer pool result: {user}")
    except Exception as e:
        print(f"Error with analyzer pool: {e}")
        import traceback
        traceback.print_exc()
    
    await analyzer.close()

if __name__ == "__main__":
    asyncio.run(test_minimal_search())
