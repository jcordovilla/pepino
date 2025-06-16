#!/usr/bin/env python3
import asyncio
import aiosqlite

async def test_direct_queries():
    async with aiosqlite.connect('discord_messages.db') as db:
        # Test exact display name match
        print("Testing exact display name match:")
        async with db.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE author_display_name = ?
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Result: {user}")
        
        # Test case-insensitive display name match
        print("\nTesting case-insensitive display name match:")
        async with db.execute("""
            SELECT DISTINCT author_id, author_name 
            FROM messages 
            WHERE LOWER(author_display_name) = LOWER(?)
            LIMIT 1
        """, ("Arturo Cuevas",)) as cursor:
            user = await cursor.fetchone()
            print(f"Result: {user}")
            
        # Get all display names that contain Arturo
        print("\nAll users with 'Arturo' in display name:")
        async with db.execute("""
            SELECT DISTINCT author_name, author_display_name
            FROM messages 
            WHERE author_display_name LIKE '%Arturo%'
        """) as cursor:
            users = await cursor.fetchall()
            for user in users:
                print(f"  {user[0]} -> {user[1]}")

if __name__ == "__main__":
    asyncio.run(test_direct_queries())
