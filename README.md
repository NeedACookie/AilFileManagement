# File System Agent 🤖

An intelligent file system agent powered by LangGraph and Llama 3.2 (via Ollama) that allows you to interact with your file system using natural language.

## Features

- 🤖 **Local LLM**: Uses Ollama with Llama 3.2 for privacy and offline operation
- 🔧 **30+ Tools**: Comprehensive file system, system info, and security tools
- 🔒 **Security**: Safe zones, path validation, and protected directory enforcement
- 💬 **Interactive UI**: Beautiful Streamlit interface with chat-based interaction
- 📊 **Tool Visualization**: See exactly what operations the agent performs

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally
- Llama 3.2 model pulled: `ollama pull llama3.2`

## Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit UI

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Configuration

In the sidebar, you can configure:

- **Model Settings**: Choose Ollama model and temperature
- **Safe Zones**: Define allowed directories for operations
- **System Info**: View agent status and settings

### Example Prompts

Try these natural language commands:

- "List all Python files in my home directory"
- "Show me disk usage information"
- "Find all files larger than 10MB in the current directory"
- "Create a new directory called 'test_folder'"
- "Get system information including CPU and memory"
- "Search for files containing 'TODO' in my Documents folder"
- "What are the permissions for /etc/hosts?"
- "Calculate SHA256 hash of requirements.txt"

## Project Structure

```
AilFileManagement/
├── app.py                 # Streamlit UI
├── agent.py              # LangGraph agent logic
├── tools/
│   ├── __init__.py       # Tool exports
│   ├── filesystem.py     # File system operations (11 tools)
│   ├── system.py         # System information (12 tools)
│   └── security.py       # Security & validation (10 tools)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Available Tools

### File System (11 tools)
- `list_directory` - List directory contents with filtering
- `search_files` - Multi-criteria file search
- `read_file` - Read file contents
- `write_file` - Create/modify files
- `delete_path` - Delete files/directories
- `move_path` - Move/rename files
- `copy_path` - Copy files/directories
- `create_directory` - Create directories
- `get_file_info` - Get file metadata
- `get_directory_tree` - Hierarchical tree view

### System Information (12 tools)
- `get_disk_usage` - Disk space information
- `get_all_disks` - All mounted partitions
- `get_memory_info` - RAM and swap usage
- `get_cpu_info` - CPU stats and usage
- `get_system_info` - OS and platform info
- `get_network_info` - Network interfaces
- `get_process_list` - Running processes
- `get_env_variable` - Environment variables
- `get_battery_info` - Battery status
- `get_temperature_sensors` - Thermal monitoring

### Security (10 tools)
- `validate_path` - Path validation
- `check_path_traversal` - Detect traversal attacks
- `get_permissions` - File permissions
- `get_file_hash` - Calculate file hashes
- And more...

## Security Features

- **Safe Zones**: Only operate within configured directories
- **Protected Paths**: System directories are automatically protected
- **Path Validation**: Prevents directory traversal attacks
- **Operation Validation**: Pre-validates destructive operations

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Type checking
mypy agent.py tools/

# Linting
ruff check .
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
