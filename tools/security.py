"""
Security tools for the agent.

Provides path validation, permission checking, and security enforcement
to prevent unauthorized or dangerous file system operations.
"""

import os
import stat
import platform
from pathlib import Path
from typing import Any


class SecurityTools:
    """Collection of security validation and permission checking tools."""

    # Protected system directories by platform
    PROTECTED_PATHS = {
        "Darwin": [  # macOS
            "/System",
            "/Library/System",
            "/private/var/db",
            "/private/var/root",
            "/bin",
            "/sbin",
            "/usr/bin",
            "/usr/sbin",
        ],
        "Windows": [
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\ProgramData\\Microsoft",
        ],
        "Linux": [
            "/bin",
            "/sbin",
            "/usr/bin",
            "/usr/sbin",
            "/etc",
            "/boot",
            "/sys",
            "/proc",
            "/root",
        ],
    }

    @staticmethod
    def validate_path(
        path: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allow_symlinks: bool = True,
    ) -> dict[str, Any]:
        """
        Validate a path for basic security and existence checks.

        Args:
            path: Path to validate
            must_exist: If True, path must exist
            must_be_file: If True, path must be a file
            must_be_dir: If True, path must be a directory
            allow_symlinks: If True, allow symbolic links

        Returns:
            Dictionary with validation results
        """
        try:
            target_path = Path(path)
            
            # Resolve to absolute path
            try:
                resolved_path = target_path.resolve()
            except (OSError, RuntimeError) as e:
                return {
                    "status": "error",
                    "valid": False,
                    "message": f"Failed to resolve path: {str(e)}",
                }

            issues = []
            warnings = []

            # Check existence
            exists = resolved_path.exists()
            if must_exist and not exists:
                issues.append("Path does not exist")

            if exists:
                # Check if symlink
                if target_path.is_symlink():
                    if not allow_symlinks:
                        issues.append("Symbolic links are not allowed")
                    else:
                        warnings.append("Path is a symbolic link")

                # Check file/directory type
                if must_be_file and not resolved_path.is_file():
                    issues.append("Path is not a file")
                if must_be_dir and not resolved_path.is_dir():
                    issues.append("Path is not a directory")

            # Check for path traversal attempts
            if ".." in path or path.startswith("~"):
                warnings.append("Path contains relative components")

            result = {
                "status": "success",
                "valid": len(issues) == 0,
                "path": str(target_path),
                "resolved_path": str(resolved_path),
                "exists": exists,
            }

            if exists:
                result["is_file"] = resolved_path.is_file()
                result["is_dir"] = resolved_path.is_dir()
                result["is_symlink"] = target_path.is_symlink()

            if issues:
                result["issues"] = issues
            if warnings:
                result["warnings"] = warnings

            return result

        except Exception as e:
            return {
                "status": "error",
                "valid": False,
                "message": f"Path validation failed: {str(e)}",
            }

    @staticmethod
    def check_path_traversal(path: str) -> dict[str, Any]:
        """
        Check if a path contains directory traversal attempts.

        Args:
            path: Path to check

        Returns:
            Dictionary with traversal detection results
        """
        try:
            suspicious_patterns = [
                "..",
                "/../",
                "\\..\\",
                "%2e%2e",
                "..%2f",
                "..%5c",
            ]

            detected = []
            normalized_path = path.lower()

            for pattern in suspicious_patterns:
                if pattern in normalized_path:
                    detected.append(pattern)

            # Check if resolved path escapes intended directory
            try:
                target_path = Path(path).resolve()
                original_parts = Path(path).parts
                
                # Look for .. in parts
                if ".." in original_parts:
                    detected.append(".. in path components")

            except Exception:
                pass

            is_safe = len(detected) == 0

            return {
                "status": "success",
                "safe": is_safe,
                "path": path,
                "detected_patterns": detected if detected else None,
                "message": "Path is safe" if is_safe else "Potential directory traversal detected",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Traversal check failed: {str(e)}",
            }

    @staticmethod
    def is_protected_path(path: str) -> dict[str, Any]:
        """
        Check if a path is in a protected system directory.

        Args:
            path: Path to check

        Returns:
            Dictionary indicating if path is protected
        """
        try:
            target_path = Path(path).resolve()
            system = platform.system()
            protected_dirs = SecurityTools.PROTECTED_PATHS.get(system, [])

            is_protected = False
            matched_protected = None

            for protected in protected_dirs:
                protected_path = Path(protected).resolve()
                try:
                    # Check if target is under protected directory
                    target_path.relative_to(protected_path)
                    is_protected = True
                    matched_protected = str(protected_path)
                    break
                except ValueError:
                    # Not relative to this protected path
                    continue

            return {
                "status": "success",
                "protected": is_protected,
                "path": str(target_path),
                "matched_protected_path": matched_protected,
                "message": f"Path is in protected directory: {matched_protected}" if is_protected else "Path is not protected",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Protected path check failed: {str(e)}",
            }

    @staticmethod
    def check_safe_zone(
        path: str,
        safe_zones: list[str],
    ) -> dict[str, Any]:
        """
        Check if a path is within allowed safe zones.

        Args:
            path: Path to check
            safe_zones: List of allowed directory paths

        Returns:
            Dictionary indicating if path is in safe zone
        """
        try:
            target_path = Path(path).resolve()

            is_safe = False
            matched_zone = None

            for zone in safe_zones:
                zone_path = Path(zone).resolve()
                try:
                    # Check if target is under safe zone
                    target_path.relative_to(zone_path)
                    is_safe = True
                    matched_zone = str(zone_path)
                    break
                except ValueError:
                    # Not relative to this safe zone
                    continue

            return {
                "status": "success",
                "in_safe_zone": is_safe,
                "path": str(target_path),
                "matched_safe_zone": matched_zone,
                "message": f"Path is in safe zone: {matched_zone}" if is_safe else "Path is outside safe zones",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Safe zone check failed: {str(e)}",
            }

    @staticmethod
    def get_permissions(path: str) -> dict[str, Any]:
        """
        Get detailed file/directory permissions.

        Args:
            path: Path to check permissions for

        Returns:
            Dictionary with permission details
        """
        try:
            target_path = Path(path).resolve()

            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"Path does not exist: {path}",
                }

            stat_info = target_path.stat()
            mode = stat_info.st_mode

            # Parse permissions
            permissions = {
                "owner_read": bool(mode & stat.S_IRUSR),
                "owner_write": bool(mode & stat.S_IWUSR),
                "owner_execute": bool(mode & stat.S_IXUSR),
                "group_read": bool(mode & stat.S_IRGRP),
                "group_write": bool(mode & stat.S_IWGRP),
                "group_execute": bool(mode & stat.S_IXGRP),
                "others_read": bool(mode & stat.S_IROTH),
                "others_write": bool(mode & stat.S_IWOTH),
                "others_execute": bool(mode & stat.S_IXOTH),
            }

            # Octal representation
            octal_permissions = oct(stat.S_IMODE(mode))

            # Human-readable format
            def _format_permissions(m: int) -> str:
                perms = ""
                perms += "r" if m & stat.S_IRUSR else "-"
                perms += "w" if m & stat.S_IWUSR else "-"
                perms += "x" if m & stat.S_IXUSR else "-"
                perms += "r" if m & stat.S_IRGRP else "-"
                perms += "w" if m & stat.S_IWGRP else "-"
                perms += "x" if m & stat.S_IXGRP else "-"
                perms += "r" if m & stat.S_IROTH else "-"
                perms += "w" if m & stat.S_IWOTH else "-"
                perms += "x" if m & stat.S_IXOTH else "-"
                return perms

            result = {
                "status": "success",
                "path": str(target_path),
                "permissions": permissions,
                "octal": octal_permissions,
                "formatted": _format_permissions(mode),
                "owner_uid": stat_info.st_uid,
                "group_gid": stat_info.st_gid,
            }

            # Add owner/group names on Unix systems
            if platform.system() != "Windows":
                try:
                    import pwd
                    import grp
                    result["owner_name"] = pwd.getpwuid(stat_info.st_uid).pw_name
                    result["group_name"] = grp.getgrgid(stat_info.st_gid).gr_name
                except (ImportError, KeyError):
                    pass

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get permissions: {str(e)}",
            }

    @staticmethod
    def check_access(
        path: str,
        read: bool = False,
        write: bool = False,
        execute: bool = False,
    ) -> dict[str, Any]:
        """
        Check if current user has specific access rights to a path.

        Args:
            path: Path to check
            read: Check read permission
            write: Check write permission
            execute: Check execute permission

        Returns:
            Dictionary with access check results
        """
        try:
            target_path = Path(path).resolve()

            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"Path does not exist: {path}",
                }

            access_results = {
                "status": "success",
                "path": str(target_path),
            }

            if read:
                access_results["can_read"] = os.access(target_path, os.R_OK)
            if write:
                access_results["can_write"] = os.access(target_path, os.W_OK)
            if execute:
                access_results["can_execute"] = os.access(target_path, os.X_OK)

            # Overall access
            all_checks = []
            if read:
                all_checks.append(access_results.get("can_read", True))
            if write:
                all_checks.append(access_results.get("can_write", True))
            if execute:
                all_checks.append(access_results.get("can_execute", True))

            access_results["has_access"] = all(all_checks) if all_checks else True

            return access_results

        except Exception as e:
            return {
                "status": "error",
                "message": f"Access check failed: {str(e)}",
            }

    @staticmethod
    def validate_operation(
        path: str,
        operation: str,
        safe_zones: list[str] | None = None,
        allow_destructive: bool = True,
    ) -> dict[str, Any]:
        """
        Validate if an operation is safe to perform on a path.

        Args:
            path: Target path for operation
            operation: Operation type - "read", "write", "delete", "move", "copy"
            safe_zones: List of allowed directories (None = no restriction)
            allow_destructive: If False, block delete/move operations

        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            target_path = Path(path).resolve()

            issues = []
            warnings = []
            allowed = True

            # Check if path is protected
            protected_check = SecurityTools.is_protected_path(str(target_path))
            if protected_check.get("protected"):
                issues.append(f"Path is in protected system directory: {protected_check.get('matched_protected_path')}")
                allowed = False

            # Check safe zones
            if safe_zones:
                zone_check = SecurityTools.check_safe_zone(str(target_path), safe_zones)
                if not zone_check.get("in_safe_zone"):
                    issues.append("Path is outside allowed safe zones")
                    allowed = False

            # Check for traversal attempts
            traversal_check = SecurityTools.check_path_traversal(path)
            if not traversal_check.get("safe"):
                issues.append("Path contains directory traversal patterns")
                allowed = False

            # Check destructive operations
            destructive_ops = ["delete", "move"]
            if operation in destructive_ops and not allow_destructive:
                issues.append(f"Destructive operation '{operation}' is not allowed")
                allowed = False

            # Check permissions for operation
            if target_path.exists():
                if operation == "read":
                    access_check = SecurityTools.check_access(str(target_path), read=True)
                    if not access_check.get("can_read"):
                        issues.append("No read permission")
                        allowed = False
                elif operation in ["write", "delete", "move"]:
                    access_check = SecurityTools.check_access(str(target_path), write=True)
                    if not access_check.get("can_write"):
                        issues.append("No write permission")
                        allowed = False

            # Warnings for destructive operations
            if operation in destructive_ops and allowed:
                warnings.append(f"Operation '{operation}' will modify/remove files - confirmation recommended")

            return {
                "status": "success",
                "allowed": allowed,
                "path": str(target_path),
                "operation": operation,
                "issues": issues if issues else None,
                "warnings": warnings if warnings else None,
                "message": "Operation is allowed" if allowed else "Operation is blocked",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Operation validation failed: {str(e)}",
            }

    @staticmethod
    def get_file_hash(
        path: str,
        algorithm: str = "sha256",
    ) -> dict[str, Any]:
        """
        Calculate cryptographic hash of a file.

        Args:
            path: File path to hash
            algorithm: Hash algorithm - "md5", "sha1", "sha256", "sha512"

        Returns:
            Dictionary with file hash
        """
        try:
            import hashlib

            target_path = Path(path).resolve()

            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"File does not exist: {path}",
                }

            if not target_path.is_file():
                return {
                    "status": "error",
                    "message": f"Path is not a file: {path}",
                }

            # Select hash algorithm
            hash_algorithms = {
                "md5": hashlib.md5,
                "sha1": hashlib.sha1,
                "sha256": hashlib.sha256,
                "sha512": hashlib.sha512,
            }

            if algorithm not in hash_algorithms:
                return {
                    "status": "error",
                    "message": f"Unsupported algorithm: {algorithm}. Use: {', '.join(hash_algorithms.keys())}",
                }

            # Calculate hash
            hasher = hash_algorithms[algorithm]()
            with open(target_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)

            return {
                "status": "success",
                "path": str(target_path),
                "algorithm": algorithm,
                "hash": hasher.hexdigest(),
                "size": target_path.stat().st_size,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to calculate hash: {str(e)}",
            }

    @staticmethod
    def compare_files(
        path1: str,
        path2: str,
        compare_content: bool = True,
    ) -> dict[str, Any]:
        """
        Compare two files for equality.

        Args:
            path1: First file path
            path2: Second file path
            compare_content: If True, compare file contents (slower but accurate)

        Returns:
            Dictionary with comparison results
        """
        try:
            file1 = Path(path1).resolve()
            file2 = Path(path2).resolve()

            if not file1.exists():
                return {"status": "error", "message": f"File does not exist: {path1}"}
            if not file2.exists():
                return {"status": "error", "message": f"File does not exist: {path2}"}

            if not file1.is_file():
                return {"status": "error", "message": f"Path is not a file: {path1}"}
            if not file2.is_file():
                return {"status": "error", "message": f"Path is not a file: {path2}"}

            stat1 = file1.stat()
            stat2 = file2.stat()

            # Quick check: size
            size_match = stat1.st_size == stat2.st_size
            
            result = {
                "status": "success",
                "file1": str(file1),
                "file2": str(file2),
                "size_match": size_match,
                "size1": stat1.st_size,
                "size2": stat2.st_size,
            }

            # Content comparison
            if compare_content and size_match:
                # Use hash comparison for efficiency
                hash1 = SecurityTools.get_file_hash(str(file1), "sha256")
                hash2 = SecurityTools.get_file_hash(str(file2), "sha256")
                
                content_match = hash1.get("hash") == hash2.get("hash")
                result["content_match"] = content_match
                result["identical"] = content_match
            else:
                result["identical"] = False
                result["content_match"] = False if compare_content else None

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"File comparison failed: {str(e)}",
            }