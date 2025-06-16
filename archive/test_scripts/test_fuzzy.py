#!/usr/bin/env python3
import sqlite3
from fuzzywuzzy import process

def test_fuzzy_matching():
    # Connect to database
    conn = sqlite3.connect('discord_messages.db')
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute("""
        SELECT DISTINCT author_name, author_display_name 
        FROM messages 
        WHERE author_name IS NOT NULL 
        ORDER BY author_name
    """)
    all_users = cursor.fetchall()
    
    # Create name mappings
    name_to_username = {}
    user_names = []
    
    for username, display_name in all_users:
        user_names.append(username)
        name_to_username[username] = username
        
        if display_name and display_name != username:
            user_names.append(display_name)
            name_to_username[display_name] = username
    
    # Test fuzzy matching
    search_term = "Arturo Cuevas"
    print(f"Searching for: '{search_term}'")
    print(f"Total names in database: {len(user_names)}")
    
    # Check if exact match exists
    if search_term in user_names:
        print(f"✓ Exact match found: {search_term} -> {name_to_username[search_term]}")
    else:
        print("✗ No exact match found")
    
    # Test fuzzy matching
    best_matches = process.extract(search_term, user_names, limit=10)
    print(f"\nTop 10 fuzzy matches:")
    for i, (match, score) in enumerate(best_matches, 1):
        username = name_to_username.get(match, "UNKNOWN")
        print(f"{i}. {match} -> {username} (score: {score}%)")
    
    # Check specific match
    if best_matches and best_matches[0][1] >= 70:
        best_match_name = best_matches[0][0]
        matched_username = name_to_username[best_match_name]
        print(f"\n✓ Best match selected: '{best_match_name}' -> '{matched_username}' (score: {best_matches[0][1]}%)")
    else:
        print(f"\n✗ No good match found (best score: {best_matches[0][1] if best_matches else 0}%)")
    
    conn.close()

if __name__ == "__main__":
    test_fuzzy_matching()
