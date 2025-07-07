#!/usr/bin/env python3

# Extract the VALUES tuple from the conftest.py file and count the values
values_tuple = (
    f"msg_{1}",
    f"Test message {1} by test_user in test_channel",
    "2025-07-05T15:00:00",
    None,  # edited_timestamp
    None,  # jump_url
    f"user_test_user",
    "test_user",
    None,  # author_discriminator
    "test_user",
    0,  # author_is_bot
    None, None, None, None, None, None, None, None, None,
    None, None, None, None, None,
    "guild1",
    "Test Guild",
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    "test_channel",               # channel_id - use real name, no prefix
    "test_channel",               # channel_name - real name, no prefix
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
    0, 0, 0, 0, 0, 0, "2025-07-05T15:00:00"
)

print(f"Number of values: {len(values_tuple)}")
print(f"Values: {values_tuple}") 