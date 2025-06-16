#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import DiscordBotAnalyzer

async def test_format_timestamp():
    analyzer = DiscordBotAnalyzer()
    
    try:
        await analyzer.initialize()
        print("Analyzer initialized")
        
        # Get a sample timestamp from the database
        async with analyzer.pool.execute("""
            SELECT timestamp 
            FROM messages 
            WHERE author_display_name = 'Arturo Cuevas'
            LIMIT 1
        """, ) as cursor:
            result = await cursor.fetchone()
            if result:
                timestamp = result[0]
                print(f"Sample timestamp: {timestamp}")
                
                # Test format_timestamp
                try:
                    formatted = analyzer.format_timestamp(timestamp)
                    print(f"Formatted timestamp: {formatted}")
                except Exception as e:
                    print(f"Error formatting timestamp: {e}")
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(test_format_timestamp())
