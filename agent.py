"""
LangGraph agent for file system operations.

Orchestrates LLM reasoning with tool execution using local Llama 3.2 via Ollama.
"""

import os

# Disable LangSmith tracing and external connections
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

from typing import Annotated, TypedDict, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from tools import FileSystemTools, SystemTools, SecurityTools


# Define agent state
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class FileSystemAgent:
    """LangGraph-based file system agent with local LLM."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.0,
        safe_zones: list[str] | None = None,
        openai_api_key: str | None = None,
    ):
        """
        Initialize the file system agent.

        Args:
            model_name: Model name. Use 'openai:gpt-4o' for OpenAI, or 'llama3.2' for Ollama
            temperature: LLM temperature (0.0 = deterministic)
            safe_zones: List of allowed directories for operations
            openai_api_key: OpenAI API key (required if using OpenAI models)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.safe_zones = safe_zones or []
        self.openai_api_key = openai_api_key
        
        # System prompt to guide the agent's behavior
        self.system_prompt = """You are a helpful file system assistant.

Guidelines:
- Extract key information from tool outputs and present it clearly
- Be concise - users want answers, not JSON descriptions
- Show all results when listing files
- Only use tool parameters that the user explicitly mentions
- When finding "top N files by size", don't set size filters - let the tool sort by size
- Answer directly: if asked "how many", give the number; if asked "find", list what was found

IMPORTANT - File Type Filtering:
- When user asks for specific file types (e.g., "Python files", ".py files"), you MUST use the file_types parameter
- file_types must be a LIST of extensions: [".py"], [".txt", ".md"], etc.
- Examples:
  * "Python files" → file_types=[".py"]
  * "text files" → file_types=[".txt"]
  * "CSV files" → file_types=[".csv"]
  * "Python and JavaScript files" → file_types=[".py", ".js"]

IMPORTANT - Directory Exclusion:
- When user asks to exclude directories (e.g., "except .venv", "excluding node_modules"), use exclude_dirs parameter
- exclude_dirs must be a LIST of directory names: [".venv"], ["node_modules", "__pycache__"], etc.
- Common exclusions: [".venv", ".git", "node_modules", "__pycache__", ".tox", "dist", "build"]
- Examples:
  * "except .venv folders" → exclude_dirs=[".venv"]
  * "excluding node_modules and cache" → exclude_dirs=["node_modules", "__pycache__"]

Remember: You're helping users manage files efficiently."""

        # Initialize LLM based on model name
        if model_name.startswith("openai:"):
            # OpenAI model
            from langchain_openai import ChatOpenAI
            
            actual_model = model_name.replace("openai:", "")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI models")
            
            self.llm = ChatOpenAI(
                model=actual_model,
                temperature=temperature,
                api_key=openai_api_key,
            )
        else:
            # Ollama model (local)
            self.llm = ChatOllama(
                model=model_name,
            temperature=temperature,
        )

        # Create LangChain tools from our tool classes
        self.tools = self._create_langchain_tools()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _create_langchain_tools(self) -> list:
        """Create LangChain tool wrappers for our tool methods."""
        tools = []

        # File System Tools
        @tool
        def list_directory(
            path: str,
            recursive: bool = False,
            include_hidden: bool = False,
            pattern: str | None = None,
            sort_by: str | None = None,
        ) -> str:
            """List directory contents with optional filtering and sorting.
            
            When listing many files (>50), returns a summary with counts.
            For detailed file lists, use with pattern filter or non-recursive mode.
            
            Args:
                path: Directory path to list
                recursive: If True, list subdirectories recursively
                include_hidden: If True, include hidden files
                pattern: Glob pattern to filter files (e.g., "*.py")
                sort_by: Sort by "name", "size", "modified", "created", or "type" (default: "name")
            
            Returns:
                Summary string with counts, or detailed list for small results
            """
            # Default sort_by to "name" if None
            sort_option = sort_by if sort_by is not None else "name"
            
            result = FileSystemTools.list_directory(
                path=path,
                recursive=recursive,
                include_hidden=include_hidden,
                pattern=pattern,
                sort_by=sort_option,
            )
            
            # Handle errors
            if result.get("status") == "error":
                return f"Error: {result.get('message')}"
            
            file_count = result.get("file_count", 0)
            dir_count = result.get("dir_count", 0)
            total_count = result.get("total_count", 0)
            
            # For large results (>50 items), return summary
            if total_count > 50:
                summary_lines = [
                    f"Directory: {path}",
                    f"Total items: {total_count}",
                    f"Files: {file_count}",
                    f"Directories: {dir_count}",
                ]
                
                if pattern:
                    summary_lines.append(f"Pattern: {pattern}")
                if recursive:
                    summary_lines.append("(Recursive search)")
                
                return "\n".join(summary_lines)
            
            # For small results, return detailed list
            import json
            return json.dumps(result, indent=2)

        @tool
        def search_files(
            path: str,
            name_pattern: Any = None,
            content_pattern: Any = None,
            file_types: Any = None,
            min_size: Any = None,
            max_size: Any = None,
            max_results: Any = None,
            sort_by: Any = None,
            exclude_dirs: Any = None,
        ) -> str:
            """Search for files with multiple criteria.
            
            Args:
                path: Root directory to search
                name_pattern: Glob pattern for filename (e.g., "*.txt")
                content_pattern: Text pattern to search within files
                file_types: List of file extensions (e.g., [".py", ".txt"]). MUST be a list, not a string.
                min_size: Minimum file size in bytes
                max_size: Maximum file size in bytes
                max_results: Maximum number of results to return
                sort_by: Sort order ("size", "name", "modified")
                exclude_dirs: List of directory names to exclude (e.g., [".venv", "node_modules", "__pycache__"])
            """
            # Clean up parameters
            if not isinstance(name_pattern, str) or name_pattern == "": name_pattern = None
            if not isinstance(content_pattern, str) or content_pattern == "": content_pattern = None
            
            # Handle file_types - can be a list or a string representation of a list
            if file_types is not None:
                if isinstance(file_types, str):
                    # Try to parse as JSON if it looks like a list
                    import json
                    try:
                        file_types = json.loads(file_types)
                        if not isinstance(file_types, list) or len(file_types) == 0:
                            file_types = None
                    except (json.JSONDecodeError, ValueError):
                        # If not valid JSON, treat as None
                        file_types = None
                elif not isinstance(file_types, list) or len(file_types) == 0:
                    file_types = None
            
            # Handle exclude_dirs - can be a list or a string representation of a list
            if exclude_dirs is not None:
                if isinstance(exclude_dirs, str):
                    # Try to parse as JSON if it looks like a list
                    import json
                    try:
                        exclude_dirs = json.loads(exclude_dirs)
                        if not isinstance(exclude_dirs, list) or len(exclude_dirs) == 0:
                            exclude_dirs = None
                    except (json.JSONDecodeError, ValueError):
                        # If not valid JSON, treat as None
                        exclude_dirs = None
                elif not isinstance(exclude_dirs, list) or len(exclude_dirs) == 0:
                    exclude_dirs = None
            
            # Convert numbers
            def clean_int(val):
                if isinstance(val, str):
                    try: return int(val)
                    except: return None
                return val if isinstance(val, int) else None

            min_size = clean_int(min_size)
            max_size = clean_int(max_size)
            requested_limit = clean_int(max_results) or 100
            
            # INTELLIGENT PARAMETER HANDLING
            # If user wants to sort by size (top N largest/smallest), we must search MANY files first
            actual_max_results = requested_limit
            is_size_sort = sort_by == "size" or (isinstance(sort_by, str) and "size" in sort_by)
            
            if is_size_sort:
                # Bump search limit significantly to look at ALL files for global sorting
                # 10,000 is too small for a home directory (often >1M files)
                # Using 2,000,000 to be safe while keeping memory usage reasonable (~400MB max)
                actual_max_results = max(2000000, requested_limit)
                # Clear size filters if they seem restrictive (exact match)
                if min_size and max_size and min_size == max_size:
                    min_size = None
                    max_size = None

            result = FileSystemTools.search_files(
                path=path,
                name_pattern=name_pattern,
                content_pattern=content_pattern,
                file_types=file_types,
                min_size=min_size,
                max_size=max_size,
                max_results=actual_max_results,
                exclude_dirs=exclude_dirs,
            )
            
            # Format the output for better LLM understanding
            if result.get("status") == "error":
                return f"Error: {result.get('message')}"
            
            matches = result.get("matches", [])
            match_count = result.get("match_count", 0)
            
            if match_count == 0:
                return f"No files found matching the criteria in {path}"
            
            # SORTING LOGIC
            reverse_sort = True  # Default: Descending (Largest first)
            
            if is_size_sort:
                # Check if user wants smallest first (asc, small, least)
                sort_str = str(sort_by).lower()
                if "asc" in sort_str or "small" in sort_str or "least" in sort_str:
                    reverse_sort = False
            
            matches_sorted = sorted(matches, key=lambda x: x.get("size", 0), reverse=reverse_sort)
            
            # Use requested_limit for display
            display_limit = requested_limit
            
            # Format as numbered list with file sizes
            sort_desc = "smallest" if not reverse_sort else "largest"
            returned_count = result.get("returned_count", len(matches))
            is_truncated = result.get("truncated", False)
            
            # Start with a clear summary that emphasizes the count
            output_lines = [
                f"SEARCH COMPLETE:",
                f"Total files found: {match_count}",
            ]
            
            if is_truncated:
                output_lines.append(f"Showing top {returned_count} {sort_desc} by size (results limited):")
            else:
                output_lines.append(f"Showing all {returned_count} files sorted by size ({sort_desc} first):")
            
            output_lines.append("")
            
            for i, match in enumerate(matches_sorted[:display_limit], 1):
                name = match.get("name", "unknown")
                file_path = match.get("path", "")
                size = match.get("size", 0)
                
                # Format size in human-readable format
                if size >= 1e9:
                    size_str = f"{size / 1e9:.2f} GB"
                elif size >= 1e6:
                    size_str = f"{size / 1e6:.2f} MB"
                elif size >= 1e3:
                    size_str = f"{size / 1e3:.2f} KB"
                else:
                    size_str = f"{size} bytes"
                
                output_lines.append(f"{i}. {name} ({size_str})")
                output_lines.append(f"   Path: {file_path}")
            
            return "\n".join(output_lines)

        @tool
        def search_directories(
            path: str,
            name_pattern: str,
            max_results: int | None = None,
        ) -> str:
            """Search for directories by name pattern across the file system.
            
            Use this tool to find directories/folders by name. Returns a formatted list of absolute paths.
            
            Args:
                path: Root directory to start search (use "/" for entire system)
                name_pattern: Directory name pattern to match (e.g., "Development", "*dev*", "Project*")
                max_results: Maximum number of results (None or large number for unlimited)
            
            Returns:
                Formatted string with list of matching directory paths
            """
            # Use 1000 as default for "unlimited" search
            limit = max_results if max_results is not None else 1000
            result = FileSystemTools.search_directories(
                path=path,
                name_pattern=name_pattern,
                max_results=limit,
            )
            
            # Format the results for better LLM understanding
            if result.get("status") == "error":
                return f"Error: {result.get('message')}"
            
            matches = result.get("matches", [])
            match_count = result.get("match_count", 0)
            
            if match_count == 0:
                return f"No directories found matching pattern '{name_pattern}' in {path}"
            
            # Create a formatted list of paths
            output_lines = [
                f"Found {match_count} director{'y' if match_count == 1 else 'ies'} matching '{name_pattern}':",
                ""
            ]
            
            for i, match in enumerate(matches, 1):
                output_lines.append(f"{i}. {match['path']}")
            
            if result.get("truncated"):
                output_lines.append(f"\n(Results limited to {limit}. There may be more matches.)")
            
            return "\n".join(output_lines)

        @tool
        def read_file(
            path: str,
            encoding: str = "utf-8",
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> dict:
            """Read file contents.
            
            Args:
                path: File path to read
                encoding: Text encoding (default: utf-8)
                start_line: Starting line number (1-indexed)
                end_line: Ending line number (inclusive)
            """
            return FileSystemTools.read_file(
                path=path,
                encoding=encoding,
                start_line=start_line,
                end_line=end_line,
            )

        @tool
        def write_file(
            path: str,
            content: str,
            mode: str = "overwrite",
            create_dirs: bool = True,
        ) -> dict:
            """Write content to a file.
            
            Args:
                path: File path to write
                content: Content to write
                mode: "overwrite", "append", or "create_only"
                create_dirs: If True, create parent directories
            """
            # Validate operation if safe zones are configured
            if self.safe_zones:
                validation = SecurityTools.validate_operation(
                    path=path,
                    operation="write",
                    safe_zones=self.safe_zones,
                )
                if not validation.get("allowed"):
                    return validation

            return FileSystemTools.write_file(
                path=path,
                content=content,
                mode=mode,
                create_dirs=create_dirs,
            )

        @tool
        def delete_path(path: str, recursive: bool = False) -> dict:
            """Delete a file or directory.
            
            Args:
                path: Path to delete
                recursive: If True, delete directories and contents
            """
            # Validate operation if safe zones are configured
            if self.safe_zones:
                validation = SecurityTools.validate_operation(
                    path=path,
                    operation="delete",
                    safe_zones=self.safe_zones,
                    allow_destructive=True,
                )
                if not validation.get("allowed"):
                    return validation

            return FileSystemTools.delete_path(path=path, recursive=recursive)

        @tool
        def move_path(source: str, destination: str, overwrite: bool = False) -> dict:
            """Move or rename a file or directory.
            
            Args:
                source: Source path
                destination: Destination path
                overwrite: If True, overwrite existing destination
            """
            return FileSystemTools.move_path(
                source=source,
                destination=destination,
                overwrite=overwrite,
            )

        @tool
        def copy_path(source: str, destination: str, overwrite: bool = False) -> dict:
            """Copy a file or directory.
            
            Args:
                source: Source path
                destination: Destination path
                overwrite: If True, overwrite existing destination
            """
            return FileSystemTools.copy_path(
                source=source,
                destination=destination,
                overwrite=overwrite,
            )

        @tool
        def create_directory(path: str, parents: bool = True) -> dict:
            """Create a directory.
            
            Args:
                path: Directory path to create
                parents: If True, create parent directories
            """
            return FileSystemTools.create_directory(path=path, parents=parents)

        @tool
        def get_file_info(path: str) -> dict:
            """Get detailed information about a file or directory.
            
            Args:
                path: Path to inspect
            """
            return FileSystemTools.get_file_info(path=path)

        @tool
        def get_directory_tree(
            path: str,
            max_depth: int | None = 3,
            include_hidden: bool = False,
        ) -> dict:
            """Get hierarchical directory tree structure.
            
            Args:
                path: Root directory path
                max_depth: Maximum depth to traverse (use large number like 100 for unlimited)
                include_hidden: If True, include hidden files/directories
            """
            # Handle None max_depth by using a large number
            depth = max_depth if max_depth is not None else 100
            return FileSystemTools.get_directory_tree(
                path=path,
                max_depth=depth,
                include_hidden=include_hidden,
            )

        # System Tools
        @tool
        def get_disk_usage(path: str = "/") -> dict:
            """Get disk usage information for a path.
            
            Args:
                path: Path to check disk usage for
            """
            return SystemTools.get_disk_usage(path=path)

        @tool
        def get_system_info() -> dict:
            """Get comprehensive system information including OS, platform, and uptime."""
            return SystemTools.get_system_info()

        @tool
        def get_memory_info() -> dict:
            """Get system memory (RAM) information including usage and swap."""
            return SystemTools.get_memory_info()

        @tool
        def get_cpu_info(include_per_cpu: bool = False) -> dict:
            """Get CPU information and usage statistics.
            
            Args:
                include_per_cpu: If True, include per-CPU core statistics
            """
            return SystemTools.get_cpu_info(include_per_cpu=include_per_cpu)

        @tool
        def get_process_list(sort_by: str | None = None, limit: int = 10) -> dict:
            """Get list of running processes.
            
            Args:
                sort_by: Sort by "memory", "cpu", "name", or "pid" (default: "memory")
                limit: Maximum number of processes to return
            """
            sort_option = sort_by if sort_by is not None else "memory"
            return SystemTools.get_process_list(sort_by=sort_option, limit=limit)

        # Security Tools
        @tool
        def validate_path(path: str, must_exist: bool = False) -> dict:
            """Validate a path for security and existence.
            
            Args:
                path: Path to validate
                must_exist: If True, path must exist
            """
            return SecurityTools.validate_path(path=path, must_exist=must_exist)

        @tool
        def check_path_traversal(path: str) -> dict:
            """Check if a path contains directory traversal attempts.
            
            Args:
                path: Path to check
            """
            return SecurityTools.check_path_traversal(path=path)

        @tool
        def get_permissions(path: str) -> dict:
            """Get detailed file/directory permissions.
            
            Args:
                path: Path to check permissions for
            """
            return SecurityTools.get_permissions(path=path)

        @tool
        def get_file_hash(path: str, algorithm: str | None = None) -> dict:
            """Calculate cryptographic hash of a file.
            
            Args:
                path: File path to hash
                algorithm: Hash algorithm - "md5", "sha1", "sha256", "sha512" (default: "sha256")
            """
            algo = algorithm if algorithm is not None else "sha256"
            return SecurityTools.get_file_hash(path=path, algorithm=algo)

        # Collect all tools
        tools.extend([
            list_directory,
            search_files,
            search_directories,
            read_file,
            write_file,
            delete_path,
            move_path,
            copy_path,
            create_directory,
            get_file_info,
            get_directory_tree,
            get_disk_usage,
            get_system_info,
            get_memory_info,
            get_cpu_info,
            get_process_list,
            validate_path,
            check_path_traversal,
            get_permissions,
            get_file_hash,
        ])

        return tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM with the current state."""
        from langchain_core.messages import SystemMessage
        import re
        
        messages = state["messages"]
        
        # Prepend system prompt on first user message
        # Check if there's no SystemMessage yet
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        response = self.llm_with_tools.invoke(messages)
        
        # CUSTOM PARSER FOR QWEN / OTHER MODELS
        # If no tool calls were detected but the content looks like a tool call
        if not response.tool_calls and "<function=" in str(response.content):
            content = str(response.content)
            
            # Regex to find <function=name> ... </tool_call> pattern
            # Matches: <function=name> <parameter=key> value ... </tool_call>
            func_match = re.search(r"<function=(\w+)>(.*?)(?:</tool_call>|$)", content, re.DOTALL)
            
            if func_match:
                function_name = func_match.group(1)
                params_str = func_match.group(2)
                
                # Parse parameters: <parameter=key> value
                params = {}
                # Split by <parameter= to get chunks
                # This approach handles multi-line values better
                parts = re.split(r"<parameter=(\w+)>", params_str)
                
                # parts[0] is garbage/whitespace before first param
                # parts[1] is key1, parts[2] is value1, parts[3] is key2, parts[4] is value2...
                if len(parts) > 1:
                    for i in range(1, len(parts), 2):
                        if i+1 < len(parts):
                            key = parts[i]
                            # content is "value </parameter> <next_tag...>"
                            # so we need to remove the closing tag
                            raw_val = parts[i+1]
                            val = raw_val.split("</parameter>")[0].strip()
                            
                            # Try to convert types (int, bool, null)
                            if val.lower() == 'true': val = True
                            elif val.lower() == 'false': val = False
                            elif val.lower() in ('null', 'none'): val = None
                            elif val.isdigit(): val = int(val)
                            
                            params[key] = val
                
                # Construct tool call manually
                import uuid
                tool_call = {
                    "name": function_name,
                    "args": params,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call"
                }
                
                # Patch the response
                response.tool_calls = [tool_call]
        
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"

    def run(self, prompt: str) -> dict:
        """
        Run the agent with a user prompt.

        Args:
            prompt: User's natural language request

        Returns:
            Dictionary with final response and conversation history
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=prompt)]
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Extract final response
        messages = final_state["messages"]
        final_response = messages[-1].content if messages else "No response generated"

        return {
            "response": final_response,
            "messages": messages,
            "total_steps": len(messages),
        }

    def stream(self, prompt: str):
        """
        Stream the agent's execution.

        Args:
            prompt: User's natural language request

        Yields:
            State updates during execution
        """
        initial_state = {
            "messages": [HumanMessage(content=prompt)]
        }

        for state in self.graph.stream(initial_state):
            yield state