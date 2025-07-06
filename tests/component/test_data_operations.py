from pepino.data_operations.service import DataOperationsService

def test_data_operations_service(populated_test_db):
    service = DataOperationsService(db_path=populated_test_db)
    channels = service.get_available_channels()
    expected_channels = ['🏘old-general-chat', '🦾agent-ops', 'jose-test', '🏛netarch-general', '🛝discord-pg']
    for expected in expected_channels:
        assert expected in channels
    tables = service.get_available_tables()
    expected_tables = ['messages', 'channel_members']
    for expected in expected_tables:
        assert expected in tables
    schema = service.get_table_schema('messages')
    assert 'columns' in schema
    service.close() 