import os
import sqlite3
import json
from datetime import datetime
import shutil

def backup_database(db_path='discord_messages.db'):
    """Create a backup of the existing database"""
    if not os.path.exists(db_path):
        print("No existing database found to backup")
        return None
    
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path

def migrate_database():
    """Migrate the database to the new schema"""
    # Create backup
    backup_path = backup_database()
    print(f"Created backup at: {backup_path}")
    
    try:
        # Create new database with updated schema
        new_db_path = 'discord_messages_new.db'
        from fetch_messages import init_database
        init_database(new_db_path)
        
        # Connect to both databases
        old_conn = sqlite3.connect('discord_messages.db')
        new_conn = sqlite3.connect(new_db_path)
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all messages from old database
        old_cursor.execute('SELECT * FROM messages')
        messages = old_cursor.fetchall()
        
        # Get column names from old database
        old_cursor.execute('PRAGMA table_info(messages)')
        old_columns = [col[1] for col in old_cursor.fetchall()]
        
        # Get column names from new database
        new_cursor.execute('PRAGMA table_info(messages)')
        new_columns = [col[1] for col in new_cursor.fetchall()]
        
        # Create a mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}
        
        # Prepare the INSERT statement for the new database
        columns_str = ', '.join(column_mapping.keys())
        placeholders = ', '.join(['?' for _ in column_mapping])
        insert_sql = f'INSERT INTO messages ({columns_str}) VALUES ({placeholders})'
        
        # Migrate messages
        for message in messages:
            # Create a dictionary of column values
            message_dict = dict(zip(old_columns, message))
            
            # Filter values to only include columns that exist in the new schema
            values = [message_dict[col] for col in column_mapping.keys()]
            
            # Insert into new database
            new_cursor.execute(insert_sql, values)
        
        # Commit changes
        new_conn.commit()
        
        # Verify migration
        old_count = old_cursor.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
        new_count = new_cursor.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
        
        if old_count != new_count:
            raise Exception(f"Message count mismatch: old={old_count}, new={new_count}")
        
        # Close connections
        old_conn.close()
        new_conn.close()
        
        # Replace old database with new one
        os.replace(new_db_path, 'discord_messages.db')
        
        print("Migration completed successfully!")
        print(f"Migrated {new_count} messages")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        print("Restoring from backup...")
        restore_backup(backup_path)
        print("Backup restored. Please check the error and try again.")
        raise

if __name__ == '__main__':
    migrate_database() 