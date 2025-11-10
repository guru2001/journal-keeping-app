# Journal Chat App

A Streamlit-based journaling application with an AI chat interface that allows users to add journal entries and query them using natural language.

![App Screenshot](images/image1.png)

## Features

### Adding Journal Entries
Users can add entries through natural language prompts:
- "Remind me to buy eggs next time I'm at the supermarket"
- "Alice says 'I should check out Kritunga for their awesome biryani'"
- "Note: Meeting with John tomorrow at 3 PM"

![Adding Entries](images/image2.png)

### Querying Journal Entries
Users can query their journal with questions like:
- "What is my shopping list?"
- "I'm at the supermarket. What should I buy?"
- "What did I note about Alice?"
- "Show me all my reminders"

## Implementation

### Technology Stack
- **Framework**: Streamlit
- **AI Model**: OpenAI GPT-4o-mini with function calling
- **Agent Framework**: LangChain with OpenAI Tools Agent
- **Embeddings**: OpenAI text-embedding-3-small for semantic search
- **Storage**: In-memory (Streamlit session state)

### Key Features

**Function Calling**: Uses OpenAI's function calling to add entries (`add_journal_entry`) and query entries (`query_journal`).

**Hybrid Search**: Combines vector search (70%) with keyword matching (30%) for optimal query results. Each entry is automatically embedded for semantic search.

**Query Categorization**: The `query_journal` function intelligently filters by category based on user intent:
- Shopping queries → `category_filter="shopping"` (returns aggregated shopping items)
- Reminder queries → `category_filter="reminder"` (returns all reminders)
- General queries → no filter (searches across all entries)

**Shopping List Extraction**: Automatically extracts and aggregates shopping items from entries. Handles incremental additions (e.g., "also add meat" only adds "meat").

**Hallucination Safeguards**: System prompt limits the assistant to journal-related queries only. Declines non-journal queries (e.g., "What is 2+2?") with appropriate responses.

**Chat History Management**: Implements sliding window approach - keeps the most recent 10 message pairs to fit within prompt limits while maintaining full history in the UI.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Architecture

- **`app.py`**: Main Streamlit application with chat interface and LangChain agent
- **`journal_manager.py`**: Journal entry management with vector search capabilities

### How It Works

1. User sends a message through the chat interface
2. LangChain agent processes the message using function calling
3. Agent determines whether to add an entry, query entries, or decline non-journal queries
4. For queries, agent determines appropriate category filter and uses hybrid search
5. Returns formatted response to the user

## Notes

- Uses GPT-4o-mini by default for cost efficiency (can be changed to GPT-4 in `app.py`)
- Journal entries are categorized automatically (shopping, reminder, quote, note, general)
- Chat history and entries stored in Streamlit session state (lost on page refresh)
- Chat history truncated to most recent 10 message pairs to fit within prompt limits
