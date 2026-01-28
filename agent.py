"""
LangGraph agent for file system operations.

Orchestrates LLM reasoning with tool execution using OpenAI GPT-4o.
"""

import os

# Disable LangSmith tracing and external connections
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

from pathlib import Path

from typing import Annotated, TypedDict, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from tools import FileSystemTools, SystemTools, SecurityTools


# Define agent state
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class FileSystemAgent:
    """LangGraph-based file system agent with OpenAI GPT-4o."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        safe_zones: list[str] | None = None,
        openai_api_key: str | None = None,
    ):
        """
        Initialize the file system agent.

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo')
            temperature: LLM temperature (0.0 = deterministic)
            safe_zones: List of allowed directories for operations
            openai_api_key: OpenAI API key (required)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.safe_zones = safe_zones or []
        self.openai_api_key = openai_api_key
        
        # Load system prompt from file
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        

        # Initialize OpenAI LLM
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
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
            
            # Format and return
            return FileSystemTools.format_directory_listing(result, path, pattern, recursive)

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
                file_types: List of file extensions (e.g., [".py", ".txt"])
                min_size: Minimum file size in bytes
                max_size: Maximum file size in bytes
                max_results: Maximum number of results to return
                sort_by: Sort order ("size", "name", "modified")
                exclude_dirs: List of directory names to exclude (e.g., [".venv", "node_modules"])
            """
            # Clean up empty string parameters
            if not isinstance(name_pattern, str) or name_pattern == "": 
                name_pattern = None
            if not isinstance(content_pattern, str) or content_pattern == "": 
                content_pattern = None
            
            # Validate list parameters
            if file_types is not None and not isinstance(file_types, list):
                file_types = None
            if exclude_dirs is not None and not isinstance(exclude_dirs, list):
                exclude_dirs = None
            
            # Set default max_results
            requested_limit = max_results if isinstance(max_results, int) else 100
            
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
            
            # Format and return
            return FileSystemTools.format_search_results(result, sort_by, requested_limit)


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
            
            # Format and return
            return FileSystemTools.format_directory_search(result)


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
        
        messages = state["messages"]
        
        # Prepend system prompt on first user message
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        response = self.llm_with_tools.invoke(messages)
        
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