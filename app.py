"""
Journal Chat App - A Streamlit-based journaling application with AI chat interface using LangChain
"""
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from journal_manager import JournalManager
from typing import List, Optional, Dict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize OpenAI client (for embeddings)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state
if "journal_manager" not in st.session_state:
    st.session_state.journal_manager = JournalManager(openai_client=client)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your journaling assistant. I can help you add entries to your journal or answer questions about your journal entries. How can I help you today?"
        }
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# System prompt to keep the assistant focused on journaling
SYSTEM_PROMPT = """You are a helpful journaling assistant. Your role is to:
1. Help users add entries to their journal
2. Answer questions about their journal entries
3. Extract and organize information from journal entries (like shopping lists, reminders, quotes, etc.)

IMPORTANT RULES:
- You MUST only respond to journal-related queries
- If a user asks about something unrelated to journaling (like math, general knowledge, etc.), politely decline and remind them you're only a journaling app. For non-journal queries, respond directly WITHOUT using any tools.
- For journal-related queries, ALWAYS use the available tools (add_journal_entry or query_journal) - never respond directly without using a tool
- Be friendly, helpful, and concise
- When adding items to a shopping list, ONLY mention the NEW items you just added in your response. Do NOT list all existing items unless the user explicitly asks to see the full list.

For add_journal_entry:
- Intelligently categorize entries: "shopping" (for buying/purchasing), "reminder" (for tasks/reminders), "quote" (for quotes), "note" (for general notes), "general" (for everything else)
- When extracting shopping items, extract ONLY the items mentioned in the current user message (e.g., if user says "also add meat", extract only ["meat"], not all previous items)
- When user says "also add X" or "add X too", extract only X, not previous items
- Always extract individual items as separate strings in the extracted_items array

For query_journal:
- Understand the user's intent: Are they asking for shopping items? Reminders? General entries?
- Use category_filter appropriately: "shopping" for shopping-related queries, "reminder" for tasks/to-do lists, or "" for general queries
- The query parameter should capture what the user is looking for - be intelligent about matching intent"""

# Initialize LangChain tools
def create_tools(journal_manager: JournalManager) -> List[StructuredTool]:
    """Create LangChain tools for journal operations"""
    
    def add_journal_entry(content: str, category: str, extracted_items: Optional[List[str]] = None) -> str:
        """Add a new entry to the journal. Use this when the user wants to record something in their journal, like reminders, quotes, notes, or any information they want to store.
        
        Args:
            content: The full content of the journal entry as provided by the user
            category: The category of the entry (e.g., 'reminder', 'quote', 'shopping', 'note', 'general')
            extracted_items: If the entry contains a shopping list or items to buy, extract ONLY the items mentioned in the current user message. For example, if user says 'buy eggs', extract ['eggs']. If user says 'also add meat', extract ONLY ['meat'], not previous items. Always extract individual items as separate strings. If no shopping items, leave empty.
        """
        entry_id = journal_manager.add_entry(content, category, extracted_items)
        if category == "shopping" or extracted_items:
            items_str = ", ".join(extracted_items) if extracted_items else "items from your entry"
            # Use singular/plural based on number of items
            if extracted_items and len(extracted_items) == 1:
                return f"✓ I've added {items_str} to your shopping list."
            else:
                return f"✓ I've added {items_str} to your shopping list."
        else:
            return f"✓ Entry added to your journal successfully!"
    
    def query_journal(query: str, category_filter: str = "") -> str:
        """Query the journal to find relevant entries. Use this when the user asks questions about their journal entries, like 'What is my shopping list?', 'What is my to-do list?', 'Show me my tasks', 'What did I note about X?', etc.
        
        Args:
            query: The search query to find relevant journal entries
            category_filter: Use this to filter by category based on user intent. Examples: 'shopping' for shopping lists/items to buy, 'reminder' for tasks/to-do lists/reminders, 'quote' for quotes, 'note' for notes, or '' for general queries. Be intelligent about understanding what the user is asking for.
        """
        # If category_filter is "shopping", return shopping items
        if category_filter == "shopping":
            all_items = journal_manager.get_shopping_items()
            if all_items:
                items_list = "\n".join([f"• {item}" for item in all_items])
                return f"Here's your shopping list:\n\n{items_list}"
            else:
                return "I couldn't find any shopping items in your journal."
        
        # If category_filter is "reminder", return all reminders
        if category_filter == "reminder":
            reminder_entries = journal_manager.get_entries_by_category("reminder")
            if reminder_entries:
                formatted_results = []
                for entry in reminder_entries[:10]:  # Limit to 10 entries
                    formatted_results.append(f"• {entry['content']}")
                return f"Here are your reminders and tasks:\n\n" + "\n".join(formatted_results)
            else:
                return "I couldn't find any reminders or tasks in your journal."
        
        # General query - let the query matching handle it
        results = journal_manager.query_entries(query, category_filter if category_filter else None)
        
        if not results:
            return "I couldn't find any matching entries in your journal."
        
        # Format results
        formatted_results = []
        for entry in results[:10]:  # Limit to 10 entries
            formatted_results.append(f"• {entry['content']}")
        
        return f"I found {len(results)} matching entry/entries:\n\n" + "\n".join(formatted_results)
    
    # Create LangChain tools
    tools = [
        StructuredTool.from_function(
            func=add_journal_entry,
            name="add_journal_entry",
            description="Add a new entry to the journal. Use this when the user wants to record something in their journal, like reminders, quotes, notes, or any information they want to store."
        ),
        StructuredTool.from_function(
            func=query_journal,
            name="query_journal",
            description="Query the journal to find relevant entries. Use this when the user asks questions about their journal entries, like 'What is my shopping list?', 'What is my to-do list?', 'Show me my tasks', 'What did I note about X?', etc."
        )
    ]
    
    return tools

def truncate_chat_history(messages: List[Dict], max_message_pairs: int = 10) -> List[Dict]:
    """
    Truncate chat history to keep only the most recent message pairs.
    This ensures the chat history fits within prompt limits.
    
    Args:
        messages: List of message dictionaries
        max_message_pairs: Maximum number of user-assistant message pairs to keep
    
    Returns:
        Truncated list of messages, keeping the initial assistant greeting and recent pairs
    """
    if len(messages) <= max_message_pairs * 2 + 1:  # +1 for initial greeting
        return messages
    
    # Keep the initial assistant greeting if it exists
    truncated = []
    if messages and messages[0].get("role") == "assistant":
        truncated.append(messages[0])
        start_idx = 1
    else:
        start_idx = 0
    
    # Keep only the most recent message pairs
    recent_messages = messages[start_idx:]
    if len(recent_messages) > max_message_pairs * 2:
        # Take the last max_message_pairs * 2 messages
        recent_messages = recent_messages[-(max_message_pairs * 2):]
    
    truncated.extend(recent_messages)
    return truncated

# Initialize LangChain agent
def get_agent_executor(journal_manager: JournalManager):
    """Create and return a LangChain agent executor"""
    if "agent_executor" not in st.session_state:
        # Create tools
        tools = create_tools(journal_manager)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        st.session_state.agent_executor = agent_executor
    
    return st.session_state.agent_executor

def main():
    st.set_page_config(
        page_title="Journal Chat",
        page_icon="📔",
        layout="wide"
    )
    
    # Sidebar for utilities
    with st.sidebar:
        st.header("📔 Journal Info")
        journal_manager = st.session_state.journal_manager
        total_entries = len(journal_manager.get_all_entries())
        st.metric("Total Entries", total_entries)
        
        if st.button("🗑️ Clear All Entries", type="secondary"):
            journal_manager.clear_all()
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Journal cleared! How can I help you today?"
                }
            ]
            st.rerun()
        
        if st.button("🔄 Update Embeddings", type="secondary", help="Generate embeddings for entries that don't have them"):
            with st.spinner("Updating embeddings..."):
                journal_manager.update_embeddings()
            st.success("Embeddings updated!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### All Entries")
        if total_entries > 0:
            for entry in journal_manager.get_all_entries():
                with st.expander(f"Entry #{entry['id']} - {entry['category']}"):
                    st.write(entry['content'])
                    if entry.get('extracted_items'):
                        st.write("**Items:**", ", ".join(entry['extracted_items']))
        else:
            st.info("No entries yet. Start chatting to add entries!")
    
    st.title("📔 Journal Chat")
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat and immediately show it
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()
    
    # Check if there's a pending user message (last message is user, no assistant response yet)
    # This means we need to process it and generate a response
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant")):
        
        # Get the last user message
        user_message = st.session_state.messages[-1]["content"]
        
        # Get agent executor
        journal_manager = st.session_state.journal_manager
        agent_executor = get_agent_executor(journal_manager)
        
        # Prepare chat history for LangChain (exclude the current user message)
        # Truncate to keep only recent messages to avoid exceeding prompt limits
        truncated_messages = truncate_chat_history(st.session_state.messages[:-1], max_message_pairs=10)
        chat_history = []
        for msg in truncated_messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        try:
            # Invoke agent with user input
            response = agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history
            })
            
            # Extract the response
            response_text = response.get("output", "I'm here to help with your journal. How can I assist you?")
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error: {error_msg}")
            import traceback
            st.exception(e)
            response_text = f"I'm sorry, I encountered an error: {error_msg}. Please try again."
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()

if __name__ == "__main__":
    main()

