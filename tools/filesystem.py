"""
File system tools for the agent.

Provides comprehensive file system operations with cross-platform support,
security validation, and flexible parameters for maximum versatility.
"""

import os
import shutil
import fnmatch
import mimetypes
from pathlib import Path
from typing import Any
from datetime import datetime


class FileSystemTools:
    """Collection of file system operation tools."""

    @staticmethod
    def list_directory(
        path: str,
        recursive: bool = False,
        include_hidden: bool = False,
        pattern: str | None = None,
        file_types: list[str] | None = None,
        sort_by: str = "name",
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """
        List directory contents with flexible filtering and sorting.

        Args:
            path: Directory path to list
            recursive: If True, list subdirectories recursively
            include_hidden: If True, include hidden files (starting with .)
            pattern: Glob pattern to filter files (e.g., "*.py", "test_*")
            file_types: List of file extensions to include (e.g., [".py", ".txt"])
            sort_by: Sort criterion - "name", "size", "modified", "created", "type"
            max_depth: Maximum recursion depth (None for unlimited)

        Returns:
            Dictionary with status, items list, and metadata
        """
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}
            if not dir_path.is_dir():
                return {"status": "error", "message": f"Path is not a directory: {path}"}

            items = []

            def _process_entry(entry_path: Path, depth: int = 0):
                """Process a single directory entry."""
                if max_depth is not None and depth > max_depth:
                    return

                try:
                    # Skip hidden files if not included
                    if not include_hidden and entry_path.name.startswith("."):
                        return

                    # Check if it's a file or directory
                    is_dir = entry_path.is_dir()
                    is_file = entry_path.is_file()

                    # Apply pattern filter ONLY to files, not directories
                    # This allows recursion into subdirectories
                    if pattern and is_file and not fnmatch.fnmatch(entry_path.name, pattern):
                        return

                    # Apply file type filter
                    if file_types and is_file:
                        if entry_path.suffix.lower() not in [ft.lower() for ft in file_types]:
                            return

                    stat = entry_path.stat()
                    item = {
                        "name": entry_path.name,
                        "path": str(entry_path),
                        "type": "directory" if is_dir else "file",
                        "size": stat.st_size if is_file else None,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "permissions": oct(stat.st_mode)[-3:],
                        "extension": entry_path.suffix if is_file else None,
                    }
                    
                    # Only add files to the list if pattern is set
                    # Add both files and directories if no pattern
                    if pattern:
                        if is_file:
                            items.append(item)
                    else:
                        items.append(item)

                    # Recurse into subdirectories
                    if recursive and is_dir:
                        for child in entry_path.iterdir():
                            _process_entry(child, depth + 1)

                except (PermissionError, OSError) as e:
                    items.append({
                        "name": entry_path.name,
                        "path": str(entry_path),
                        "type": "error",
                        "error": str(e),
                    })

            # Process all entries in the directory
            for entry in dir_path.iterdir():
                _process_entry(entry)

            # Sort items
            sort_keys = {
                "name": lambda x: x.get("name", "").lower(),
                "size": lambda x: x.get("size") or 0,
                "modified": lambda x: x.get("modified", ""),
                "created": lambda x: x.get("created", ""),
                "type": lambda x: (x.get("type", ""), x.get("name", "").lower()),
            }
            if sort_by in sort_keys:
                items.sort(key=sort_keys[sort_by])

            return {
                "status": "success",
                "path": str(dir_path),
                "items": items,
                "total_count": len(items),
                "file_count": sum(1 for i in items if i.get("type") == "file"),
                "dir_count": sum(1 for i in items if i.get("type") == "directory"),
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to list directory: {str(e)}"}

    @staticmethod
    def search_files(
        path: str,
        name_pattern: str | None = None,
        content_pattern: str | None = None,
        file_types: list[str] | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """
        Search for files with multiple criteria.

        Args:
            path: Root directory to search
            name_pattern: Glob pattern for filename (e.g., "*.py", "*test*")
            content_pattern: Text pattern to search within files
            file_types: List of file extensions (e.g., [".py", ".txt"])
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            modified_after: ISO format datetime string
            modified_before: ISO format datetime string
            case_sensitive: If True, case-sensitive search
            max_results: Maximum number of results to return

        Returns:
            Dictionary with status and matching files
        """
        try:
            search_path = Path(path).resolve()
            if not search_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}

            matches = []
            searched_count = 0

            def _matches_criteria(file_path: Path) -> bool:
                """Check if file matches all search criteria."""
                try:
                    # Name pattern
                    if name_pattern:
                        pattern = name_pattern if case_sensitive else name_pattern.lower()
                        name = file_path.name if case_sensitive else file_path.name.lower()
                        if not fnmatch.fnmatch(name, pattern):
                            return False

                    # File type
                    if file_types and file_path.suffix.lower() not in [ft.lower() for ft in file_types]:
                        return False

                    stat = file_path.stat()

                    # Size filters
                    if min_size is not None and stat.st_size < min_size:
                        return False
                    if max_size is not None and stat.st_size > max_size:
                        return False

                    # Date filters
                    if modified_after:
                        after_dt = datetime.fromisoformat(modified_after)
                        if datetime.fromtimestamp(stat.st_mtime) < after_dt:
                            return False
                    if modified_before:
                        before_dt = datetime.fromisoformat(modified_before)
                        if datetime.fromtimestamp(stat.st_mtime) > before_dt:
                            return False

                    # Content search (for text files)
                    if content_pattern:
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                search_content = content if case_sensitive else content.lower()
                                search_pattern = content_pattern if case_sensitive else content_pattern.lower()
                                if search_pattern not in search_content:
                                    return False
                        except Exception:
                            return False

                    return True

                except (PermissionError, OSError):
                    return False

            # Walk through directory tree
            for root, _, files in os.walk(search_path):
                for filename in files:
                    if len(matches) >= max_results:
                        break

                    file_path = Path(root) / filename
                    searched_count += 1

                    if _matches_criteria(file_path):
                        stat = file_path.stat()
                        matches.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "extension": file_path.suffix,
                        })

                if len(matches) >= max_results:
                    break

            return {
                "status": "success",
                "matches": matches,
                "match_count": len(matches),
                "searched_count": searched_count,
                "truncated": len(matches) >= max_results,
            }

        except Exception as e:
            return {"status": "error", "message": f"Search failed: {str(e)}"}

    @staticmethod
    def search_directories(
        path: str,
        name_pattern: str,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """
        Search for directories by name pattern.

        Args:
            path: Root directory to start search
            name_pattern: Directory name pattern to match (e.g., "Development", "*dev*")
            case_sensitive: If True, case-sensitive search
            max_results: Maximum number of results to return

        Returns:
            Dictionary with status and matching directories
        """
        try:
            search_path = Path(path).resolve()
            if not search_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}

            matches = []
            searched_count = 0

            for root, dirs, _ in os.walk(search_path):
                for dirname in dirs:
                    searched_count += 1

                    # Match pattern using fnmatch for glob support
                    if case_sensitive:
                        pattern_match = fnmatch.fnmatch(dirname, name_pattern)
                    else:
                        pattern_match = fnmatch.fnmatch(dirname.lower(), name_pattern.lower())
                    
                    if pattern_match:
                        dir_path = Path(root) / dirname
                        try:
                            stat = dir_path.stat()
                            matches.append({
                                "name": dirname,
                                "path": str(dir_path.absolute()),
                                "parent": str(Path(root).absolute()),
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            })

                            if len(matches) >= max_results:
                                break
                        except (PermissionError, OSError):
                            continue

                if len(matches) >= max_results:
                    break

            return {
                "status": "success",
                "matches": matches,
                "match_count": len(matches),
                "searched_count": searched_count,
                "truncated": len(matches) >= max_results,
            }

        except Exception as e:
            return {"status": "error", "message": f"Directory search failed: {str(e)}"}


    @staticmethod
    def read_file(
        path: str,
        encoding: str = "utf-8",
        max_size: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Read file contents with flexible options.

        Args:
            path: File path to read
            encoding: Text encoding (default: utf-8)
            max_size: Maximum file size to read in bytes (safety limit)
            start_line: Starting line number (1-indexed, for text files)
            end_line: Ending line number (inclusive, for text files)

        Returns:
            Dictionary with status, content, and metadata
        """
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return {"status": "error", "message": f"File does not exist: {path}"}
            if not file_path.is_file():
                return {"status": "error", "message": f"Path is not a file: {path}"}

            stat = file_path.stat()

            # Check size limit
            if max_size and stat.st_size > max_size:
                return {
                    "status": "error",
                    "message": f"File size ({stat.st_size} bytes) exceeds limit ({max_size} bytes)",
                }

            # Detect if file is binary
            mime_type, _ = mimetypes.guess_type(str(file_path))
            is_text = mime_type is None or mime_type.startswith("text/")

            if is_text:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        if start_line is not None or end_line is not None:
                            lines = f.readlines()
                            start = (start_line - 1) if start_line else 0
                            end = end_line if end_line else len(lines)
                            content = "".join(lines[start:end])
                            line_info = {
                                "start_line": start_line or 1,
                                "end_line": end_line or len(lines),
                                "total_lines": len(lines),
                            }
                        else:
                            content = f.read()
                            line_info = {"total_lines": content.count("\n") + 1}

                    return {
                        "status": "success",
                        "content": content,
                        "encoding": encoding,
                        "size": stat.st_size,
                        "type": "text",
                        "mime_type": mime_type,
                        **line_info,
                    }
                except UnicodeDecodeError:
                    is_text = False

            # Binary file
            if not is_text:
                with open(file_path, "rb") as f:
                    content = f.read()
                return {
                    "status": "success",
                    "content": content.hex(),
                    "size": stat.st_size,
                    "type": "binary",
                    "mime_type": mime_type,
                    "note": "Binary content returned as hex string",
                }

        except Exception as e:
            return {"status": "error", "message": f"Failed to read file: {str(e)}"}

    @staticmethod
    def write_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
        mode: str = "overwrite",
        create_dirs: bool = True,
    ) -> dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: File path to write
            content: Content to write
            encoding: Text encoding (default: utf-8)
            mode: Write mode - "overwrite", "append", "create_only"
            create_dirs: If True, create parent directories if they don't exist

        Returns:
            Dictionary with status and operation details
        """
        try:
            file_path = Path(path).resolve()

            # Check if file exists
            exists = file_path.exists()

            if mode == "create_only" and exists:
                return {"status": "error", "message": f"File already exists: {path}"}

            # Create parent directories
            if create_dirs and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            write_mode = "a" if mode == "append" else "w"
            with open(file_path, write_mode, encoding=encoding) as f:
                f.write(content)

            stat = file_path.stat()
            return {
                "status": "success",
                "path": str(file_path),
                "size": stat.st_size,
                "mode": mode,
                "created": not exists,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to write file: {str(e)}"}

    @staticmethod
    def delete_path(
        path: str,
        recursive: bool = False,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        """
        Delete a file or directory.

        Args:
            path: Path to delete
            recursive: If True, delete directories and their contents
            missing_ok: If True, don't error if path doesn't exist

        Returns:
            Dictionary with status and operation details
        """
        try:
            target_path = Path(path).resolve()

            if not target_path.exists():
                if missing_ok:
                    return {
                        "status": "success",
                        "message": "Path does not exist (missing_ok=True)",
                        "path": str(target_path),
                    }
                return {"status": "error", "message": f"Path does not exist: {path}"}

            is_dir = target_path.is_dir()

            if is_dir and not recursive:
                # Check if directory is empty
                if any(target_path.iterdir()):
                    return {
                        "status": "error",
                        "message": f"Directory is not empty (use recursive=True): {path}",
                    }

            # Perform deletion
            if is_dir:
                shutil.rmtree(target_path)
            else:
                target_path.unlink()

            return {
                "status": "success",
                "path": str(target_path),
                "type": "directory" if is_dir else "file",
                "deleted": True,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to delete: {str(e)}"}

    @staticmethod
    def move_path(
        source: str,
        destination: str,
        overwrite: bool = False,
        create_dirs: bool = True,
    ) -> dict[str, Any]:
        """
        Move or rename a file or directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: If True, overwrite existing destination
            create_dirs: If True, create parent directories for destination

        Returns:
            Dictionary with status and operation details
        """
        try:
            src_path = Path(source).resolve()
            dst_path = Path(destination).resolve()

            if not src_path.exists():
                return {"status": "error", "message": f"Source does not exist: {source}"}

            if dst_path.exists() and not overwrite:
                return {
                    "status": "error",
                    "message": f"Destination already exists (use overwrite=True): {destination}",
                }

            # Create parent directories
            if create_dirs and not dst_path.parent.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform move
            shutil.move(str(src_path), str(dst_path))

            return {
                "status": "success",
                "source": str(src_path),
                "destination": str(dst_path),
                "type": "directory" if dst_path.is_dir() else "file",
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to move: {str(e)}"}

    @staticmethod
    def copy_path(
        source: str,
        destination: str,
        overwrite: bool = False,
        create_dirs: bool = True,
        preserve_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Copy a file or directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: If True, overwrite existing destination
            create_dirs: If True, create parent directories for destination
            preserve_metadata: If True, preserve timestamps and permissions

        Returns:
            Dictionary with status and operation details
        """
        try:
            src_path = Path(source).resolve()
            dst_path = Path(destination).resolve()

            if not src_path.exists():
                return {"status": "error", "message": f"Source does not exist: {source}"}

            if dst_path.exists() and not overwrite:
                return {
                    "status": "error",
                    "message": f"Destination already exists (use overwrite=True): {destination}",
                }

            # Create parent directories
            if create_dirs and not dst_path.parent.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform copy
            if src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path, copy_function=shutil.copy2 if preserve_metadata else shutil.copy)
            else:
                if preserve_metadata:
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)

            stat = dst_path.stat()
            return {
                "status": "success",
                "source": str(src_path),
                "destination": str(dst_path),
                "type": "directory" if dst_path.is_dir() else "file",
                "size": stat.st_size if dst_path.is_file() else None,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to copy: {str(e)}"}

    @staticmethod
    def create_directory(
        path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> dict[str, Any]:
        """
        Create a directory.

        Args:
            path: Directory path to create
            parents: If True, create parent directories as needed
            exist_ok: If True, don't error if directory already exists

        Returns:
            Dictionary with status and operation details
        """
        try:
            dir_path = Path(path).resolve()

            existed = dir_path.exists()

            if existed and not exist_ok:
                return {"status": "error", "message": f"Directory already exists: {path}"}

            dir_path.mkdir(parents=parents, exist_ok=exist_ok)

            return {
                "status": "success",
                "path": str(dir_path),
                "created": not existed,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to create directory: {str(e)}"}

    @staticmethod
    def get_file_info(path: str) -> dict[str, Any]:
        """
        Get detailed information about a file or directory.

        Args:
            path: Path to inspect

        Returns:
            Dictionary with comprehensive file/directory metadata
        """
        try:
            target_path = Path(path).resolve()

            if not target_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}

            stat = target_path.stat()
            is_dir = target_path.is_dir()

            info = {
                "status": "success",
                "name": target_path.name,
                "path": str(target_path),
                "absolute_path": str(target_path.absolute()),
                "type": "directory" if is_dir else "file",
                "size": stat.st_size if not is_dir else None,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "owner_uid": stat.st_uid,
                "group_gid": stat.st_gid,
            }

            if not is_dir:
                info["extension"] = target_path.suffix
                info["stem"] = target_path.stem
                mime_type, encoding = mimetypes.guess_type(str(target_path))
                info["mime_type"] = mime_type
                info["encoding"] = encoding
            else:
                # Count directory contents
                try:
                    contents = list(target_path.iterdir())
                    info["item_count"] = len(contents)
                    info["file_count"] = sum(1 for item in contents if item.is_file())
                    info["dir_count"] = sum(1 for item in contents if item.is_dir())
                except PermissionError:
                    info["item_count"] = None

            return info

        except Exception as e:
            return {"status": "error", "message": f"Failed to get file info: {str(e)}"}

    @staticmethod
    def get_directory_tree(
        path: str,
        max_depth: int = 3,
        include_hidden: bool = False,
        include_files: bool = True,
    ) -> dict[str, Any]:
        """
        Get hierarchical directory tree structure.

        Args:
            path: Root directory path
            max_depth: Maximum depth to traverse
            include_hidden: If True, include hidden files/directories
            include_files: If True, include files (not just directories)

        Returns:
            Dictionary with tree structure
        """
        try:
            root_path = Path(path).resolve()

            if not root_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}
            if not root_path.is_dir():
                return {"status": "error", "message": f"Path is not a directory: {path}"}

            def _build_tree(current_path: Path, depth: int = 0) -> dict[str, Any]:
                """Recursively build tree structure."""
                if depth > max_depth:
                    return None

                try:
                    stat = current_path.stat()
                    node = {
                        "name": current_path.name,
                        "path": str(current_path),
                        "type": "directory" if current_path.is_dir() else "file",
                        "size": stat.st_size if current_path.is_file() else None,
                    }

                    if current_path.is_dir():
                        children = []
                        for child in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                            # Skip hidden files if not included
                            if not include_hidden and child.name.startswith("."):
                                continue
                            # Skip files if not included
                            if not include_files and child.is_file():
                                continue

                            child_node = _build_tree(child, depth + 1)
                            if child_node:
                                children.append(child_node)

                        if children:
                            node["children"] = children

                    return node

                except (PermissionError, OSError):
                    return {
                        "name": current_path.name,
                        "path": str(current_path),
                        "type": "error",
                        "error": "Permission denied",
                    }

            tree = _build_tree(root_path)

            return {
                "status": "success",
                "root": str(root_path),
                "tree": tree,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to build directory tree: {str(e)}"}