# hello_claude.py
# A simple script that sends a message to Claude and prints the response.

# Step 1: Import the necessary libraries
#     - 'os' to read environment variable
#.    - 'dotenv' to load our .env file
#     - 'anthropic' to talk to the API

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Step 2: Load the .env file so our API key is available

load_dotenv()

# Step 3: Create an Anthropic client (this uses the API key automatically

client = Anthropic()

# Step 4: Send a message to Claude
#.    - Pick a model
#.    - Set max_tokens (how long the response can be)
#.    - Provide a messages list with one user message

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Hello, Claude! How are you doing today?"}]
)

# Step 5: Print the response

print(response.content[0].text)