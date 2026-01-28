"""
Streamlit UI for the File System Agent.

Provides an interactive chat interface for communicating with the AI agent
to perform file system operations through natural language.
"""

import os

# Disable telemetry and external connections
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import json
from pathlib import Path
from agent import FileSystemAgent


# Page configuration
st.set_page_config(
    page_title="File System Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #49657a;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #7c4e4e;
        border-left: 4px solid #4caf50;
    }
    .tool-call {
        background-color: #cca76c;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "safe_zones" not in st.session_state:
        # Default safe zone: user's home directory
        st.session_state.safe_zones = [str(Path.home())]
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o"
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""


def create_agent():
    """Create or recreate the agent with current settings."""
    try:
        st.session_state.agent = FileSystemAgent(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            safe_zones=st.session_state.safe_zones,
            openai_api_key=st.session_state.openai_api_key if st.session_state.openai_api_key else None,
        )
        return True
    except Exception as e:
        st.error(f"Failed to create agent: {str(e)}")
        return False


def format_tool_result(result: dict) -> str:
    """Format tool result for display."""
    if not isinstance(result, dict):
        return str(result)
    
    # Pretty print JSON
    return json.dumps(result, indent=2)


def display_message(message):
    """Display a chat message with appropriate styling."""
    if hasattr(message, "type"):
        msg_type = message.type
        content = message.content
        
        if msg_type == "human":
            st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{content}</div>', 
                       unsafe_allow_html=True)
        
        elif msg_type == "ai":
            st.markdown(f'<div class="chat-message assistant-message">🤖 <strong>Assistant:</strong><br>{content}</div>', 
                       unsafe_allow_html=True)
            
            # Display tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                with st.expander("🔧 Tool Calls", expanded=False):
                    for tool_call in message.tool_calls:
                        st.markdown(f'<div class="tool-call">📌 {tool_call["name"]}</div>', 
                                   unsafe_allow_html=True)
                        st.json(tool_call.get("args", {}))
        
        elif msg_type == "tool":
            with st.expander(f"📊 Tool Result: {message.name}", expanded=False):
                try:
                    result = json.loads(content) if isinstance(content, str) else content
                    st.json(result)
                except:
                    st.code(content)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 File System Agent</div>', unsafe_allow_html=True)
    st.markdown("*Interact with your file system using natural language powered by Llama 3.2*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        
        # OpenAI model selection
        model_name = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
            index=0 if st.session_state.model_name == "gpt-4o" else 
                  1 if st.session_state.model_name == "gpt-4o-mini" else
                  2 if st.session_state.model_name == "gpt-4-turbo" else 0,
            help="Select OpenAI model to use"
        )
        
        # OpenAI API key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key (starts with sk-)"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Safe zones configuration
        st.subheader("🔒 Safe Zones")
        st.markdown("*Directories where the agent is allowed to operate*")
        
        safe_zones_text = st.text_area(
            "Safe Zones (one per line)",
            value="\n".join(st.session_state.safe_zones),
            height=100,
            help="Enter allowed directory paths, one per line"
        )
        
        # Update settings button
        if st.button("💾 Apply Settings"):
            # Validate OpenAI API key
            if not openai_api_key:
                st.error("❌ OpenAI API key is required")
            elif not openai_api_key.startswith("sk-"):
                st.error("❌ Invalid OpenAI API key format. Key should start with 'sk-'")
            else:
                # Key looks valid, proceed
                st.session_state.model_name = model_name
                st.session_state.temperature = temperature
                st.session_state.openai_api_key = openai_api_key
                st.session_state.safe_zones = [
                    line.strip() for line in safe_zones_text.split("\n") 
                    if line.strip()
                ]
                
                if create_agent():
                    st.success("✅ Settings applied successfully!")
                else:
                    st.error("❌ Failed to create agent. Check your API key and try again.")
        
        
        # Initialize agent if not exists
        if st.session_state.agent is None:
            with st.spinner("Initializing agent..."):
                create_agent()
        
        st.divider()
        
        # System info
        st.subheader("📊 System Info")
        if st.session_state.agent:
            st.success("🟢 Agent Ready")
            
            # Show OpenAI model info
            st.info(f"**Provider:** 🌐 OpenAI")
            st.info(f"**Model:** {st.session_state.model_name}")
            st.info(f"**Temperature:** {st.session_state.temperature}")
            st.info(f"**Safe Zones:** {len(st.session_state.safe_zones)}")
        else:
            st.error("🔴 Agent Not Initialized")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Example prompts
        st.subheader("💡 Example Prompts")
        examples = [
            "List all Python files in my home directory",
            "Show me disk usage information",
            "Find all files larger than 10MB",
            "Create a new directory called 'test_folder'",
            "Get system information",
            "Search for files containing 'TODO'",
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                st.session_state.example_prompt = example
    
    # Main chat interface
    st.divider()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(message)
    
    # Chat input
    st.divider()
    
    # Use example prompt if set
    default_prompt = ""
    if hasattr(st.session_state, "example_prompt"):
        default_prompt = st.session_state.example_prompt
        delattr(st.session_state, "example_prompt")
    
    prompt = st.chat_input(
        "Ask me to perform file system operations...",
        key="chat_input"
    )
    
    # Handle example prompt
    if default_prompt:
        prompt = default_prompt
    
    # Process user input
    if prompt:
        if st.session_state.agent is None:
            st.error("⚠️ Agent not initialized. Please check your settings.")
            return
        
        # Add user message to history
        from langchain_core.messages import HumanMessage
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with chat_container:
            display_message(user_message)
        
        # Show processing indicator
        with st.spinner("🤔 Agent is thinking..."):
            try:
                # Run agent
                result = st.session_state.agent.run(prompt)
                
                # Add all messages from the result to history
                new_messages = result.get("messages", [])[1:]  # Skip the user message we already added
                st.session_state.messages.extend(new_messages)
                
                # Display new messages
                with chat_container:
                    for message in new_messages:
                        display_message(message)
                
                # Show success metrics
                st.success(f"✅ Completed in {result.get('total_steps', 0)} steps")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
        
        # Rerun to update the display
        st.rerun()


if __name__ == "__main__":
    main()