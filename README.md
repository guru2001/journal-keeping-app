# Journal Chat App

A Streamlit-based journaling application with an AI chat interface that allows users to add journal entries and query them using natural language.

## Problem Statement

Build a chat-like interface for a Journal app using Streamlit (Python alternative to Vercel Generative UI SDK). The app enables users to add journal entries through natural language and query their journal using conversational prompts.

## User Journeys

### 1. Adding Journal Entries

Users can add new entries to their journal by writing prompts like:
- "Remind me to buy eggs next time I'm at the supermarket"
- "Alice says 'I should check out Kritunga for their awesome biryani'"
- "Note: Meeting with John tomorrow at 3 PM"

### 2. Querying Journal Entries

Users can query their journal asking:
- "What is my shopping list?"
- "I'm at the supermarket. What should I buy?" (both return shopping list entries)
- "What did I note about Alice?"
- "Show me all my reminders"

## Implementation Details

### Technology Stack

1. **Framework**: Streamlit (Python alternative to Vercel Generative UI SDK)
2. **AI Model**: OpenAI GPT-4o-mini with function calling
3. **Agent Framework**: LangChain with OpenAI Tools Agent
4. **Embeddings**: OpenAI text-embedding-3-small for semantic search
5. **Storage**: In-memory (Streamlit session state) - no backend API required

### Key Features

1. **Function Calling**: Uses OpenAI's function calling feature to:
   - Add entries to the journal (`add_journal_entry`)
   - Query existing entries (`query_journal`)

2. **Vector Search**: Semantic search for better query results:
   - Each entry is automatically embedded using OpenAI's embedding model
   - Queries are converted to embeddings and matched using cosine similarity
   - Combines vector search (70%) with keyword matching (30%) for optimal results
   - Understands semantic meaning, not just exact keyword matches

3. **Safeguards Against Hallucinations**: 
   - System prompt explicitly limits the assistant to journal-related queries only
   - Detects non-journal queries (e.g., "What is 2+2?")
   - Responds appropriately: "I'm only a journaling app. I can't do mathematical calculations"
   - Keeps conversations focused on journal-related tasks

4. **Shopping List Extraction**: 
   - Automatically extracts shopping items from entries
   - Organizes them into a queryable shopping list
   - Handles incremental additions (e.g., "also add meat" only adds "meat")

5. **Query Journal Categorization**:
   - The `query_journal` function intelligently categorizes queries based on user intent
   - Uses a `category_filter` parameter to filter results by entry category:
     - **"shopping"**: Returns aggregated shopping items from all shopping entries
     - **"reminder"**: Returns all entries categorized as reminders/tasks
     - **"" (empty)**: Performs general semantic search across all categories
   - The AI agent automatically determines the appropriate category filter based on the user's query:
     - Shopping-related queries (e.g., "What is my shopping list?", "What should I buy?") → `category_filter="shopping"`
     - Task/reminder queries (e.g., "Show me my reminders", "What are my tasks?") → `category_filter="reminder"`
     - General queries (e.g., "What did I note about Alice?") → `category_filter=""` (no filter)
   - When a category filter is applied, the search is scoped to entries of that category, improving relevance
   - For general queries without a category filter, the hybrid search (vector + keyword) searches across all entries

6. **Chat History Management**:
   - Implements sliding window approach to handle long conversations
   - Keeps the most recent 10 message pairs to avoid exceeding prompt limits
   - Maintains full chat history in UI while truncating what's sent to the model
   - Ensures chat history fits within a single prompt as per requirements

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`

## Restrictions & Considerations

- **Memory**: Entire chat history fits into memory (Streamlit session state)
- **Prompt Limits**: Chat history is truncated to the most recent 10 message pairs to ensure it fits within a single prompt
- **No Backend**: Uses server memory (Streamlit session state) for persistence - no database or backend API required
- **Session-based**: Journal entries are stored in session memory and will be lost when you refresh the page (as per requirements)

## Deliverables

- ✅ **Code**: Complete implementation in `app.py` and `journal_manager.py`
- ✅ **README**: This documentation file

## Technical Architecture

### Components

1. **`app.py`**: Main Streamlit application with chat interface and LangChain agent
2. **`journal_manager.py`**: Journal entry management with vector search capabilities

### How It Works

1. User sends a message through the chat interface
2. LangChain agent processes the message using function calling
3. Agent determines whether to:
   - Add an entry (`add_journal_entry` tool)
   - Query entries (`query_journal` tool)
   - Decline non-journal queries (via system prompt)
4. For queries, uses hybrid search (vector + keyword) to find relevant entries
5. Returns formatted response to the user

### Chat History Management

The app implements a sliding window approach:
- Keeps the initial assistant greeting
- Maintains the most recent 10 user-assistant message pairs
- Older messages are automatically truncated to prevent exceeding prompt limits
- Full conversation history remains visible in the UI

## Notes

- The app uses GPT-4o-mini by default for cost efficiency, but can be changed to GPT-4 in `app.py`
- All chat history is maintained in Streamlit session state
- Journal entries are categorized automatically (shopping, reminder, quote, note, general)
- Shopping items are extracted and aggregated across all entries
