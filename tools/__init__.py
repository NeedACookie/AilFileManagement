"""
Tools package for the file system agent.

Exports all available tools from filesystem, system, and security modules.
"""

from tools.filesystem import FileSystemTools
from tools.system import SystemTools
from tools.security import SecurityTools

# Collect all tools in a single list for easy access
all_tools = [
    # File System Tools (12 tools)
    FileSystemTools.list_directory,
    FileSystemTools.search_files,
    FileSystemTools.search_directories,
    FileSystemTools.read_file,
    FileSystemTools.write_file,
    FileSystemTools.delete_path,
    FileSystemTools.move_path,
    FileSystemTools.copy_path,
    FileSystemTools.create_directory,
    FileSystemTools.get_file_info,
    FileSystemTools.get_directory_tree,
    
    # System Information Tools (12 tools)
    SystemTools.get_disk_usage,
    SystemTools.get_all_disks,
    SystemTools.get_memory_info,
    SystemTools.get_cpu_info,
    SystemTools.get_system_info,
    SystemTools.get_network_info,
    SystemTools.get_process_list,
    SystemTools.get_env_variable,
    SystemTools.get_all_env_variables,
    SystemTools.get_battery_info,
    SystemTools.get_temperature_sensors,
    
    # Security Tools (10 tools)
    SecurityTools.validate_path,
    SecurityTools.check_path_traversal,
    SecurityTools.is_protected_path,
    SecurityTools.check_safe_zone,
    SecurityTools.get_permissions,
    SecurityTools.check_access,
    SecurityTools.validate_operation,
    SecurityTools.get_file_hash,
    SecurityTools.compare_files,
]

# Export classes for direct access
__all__ = [
    "FileSystemTools",
    "SystemTools",
    "SecurityTools",
    "all_tools",
]