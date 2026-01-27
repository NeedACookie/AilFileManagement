"""
System information tools for the agent.

Provides comprehensive system monitoring and information gathering
with cross-platform support for macOS, Windows, and Linux.
"""

import os
import platform
import psutil
from pathlib import Path
from typing import Any
from datetime import datetime


class SystemTools:
    """Collection of system information and monitoring tools."""

    @staticmethod
    def get_disk_usage(
        path: str = "/",
        human_readable: bool = True,
    ) -> dict[str, Any]:
        """
        Get disk usage information for a specific path.

        Args:
            path: Path to check disk usage for (default: root)
            human_readable: If True, format sizes in human-readable units

        Returns:
            Dictionary with disk usage statistics
        """
        try:
            target_path = Path(path).resolve()
            if not target_path.exists():
                return {"status": "error", "message": f"Path does not exist: {path}"}

            usage = psutil.disk_usage(str(target_path))
            
            # Calculate "other" usage (for APFS containers where multiple volumes share space)
            # Total = Used + Free + Other
            other_used = usage.total - usage.used - usage.free
            if other_used < 0: other_used = 0
            
            def _format_bytes(bytes_value: int) -> str:
                """Format bytes to human-readable format."""
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_value < 1024.0:
                        return f"{bytes_value:.2f} {unit}"
                    bytes_value /= 1024.0
                return f"{bytes_value:.2f} PB"

            result = {
                "status": "success",
                "path": str(target_path),
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "other": other_used,
                "percent_used": usage.percent,
            }

            if human_readable:
                result["total_formatted"] = _format_bytes(usage.total)
                result["used_formatted"] = _format_bytes(usage.used)
                result["free_formatted"] = _format_bytes(usage.free)
                result["other_formatted"] = _format_bytes(other_used)

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get disk usage: {str(e)}"}

    @staticmethod
    def get_all_disks(
        human_readable: bool = True,
        include_virtual: bool = False,
    ) -> dict[str, Any]:
        """
        Get information about all mounted disks/partitions.

        Args:
            human_readable: If True, format sizes in human-readable units
            include_virtual: If True, include virtual/network drives

        Returns:
            Dictionary with information about all disks
        """
        try:
            partitions = psutil.disk_partitions(all=include_virtual)
            disks = []

            def _format_bytes(bytes_value: int) -> str:
                """Format bytes to human-readable format."""
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_value < 1024.0:
                        return f"{bytes_value:.2f} {unit}"
                    bytes_value /= 1024.0
                return f"{bytes_value:.2f} PB"

            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "options": partition.opts,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent_used": usage.percent,
                    }

                    if human_readable:
                        disk_info["total_formatted"] = _format_bytes(usage.total)
                        disk_info["used_formatted"] = _format_bytes(usage.used)
                        disk_info["free_formatted"] = _format_bytes(usage.free)

                    disks.append(disk_info)

                except (PermissionError, OSError):
                    # Skip inaccessible partitions
                    continue

            return {
                "status": "success",
                "disk_count": len(disks),
                "disks": disks,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to get disk information: {str(e)}"}

    @staticmethod
    def get_memory_info(
        human_readable: bool = True,
        include_swap: bool = True,
    ) -> dict[str, Any]:
        """
        Get system memory (RAM) information.

        Args:
            human_readable: If True, format sizes in human-readable units
            include_swap: If True, include swap memory information

        Returns:
            Dictionary with memory statistics
        """
        try:
            memory = psutil.virtual_memory()

            def _format_bytes(bytes_value: int) -> str:
                """Format bytes to human-readable format."""
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_value < 1024.0:
                        return f"{bytes_value:.2f} {unit}"
                    bytes_value /= 1024.0
                return f"{bytes_value:.2f} PB"

            result = {
                "status": "success",
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free,
                "percent_used": memory.percent,
            }

            if human_readable:
                result["total_formatted"] = _format_bytes(memory.total)
                result["available_formatted"] = _format_bytes(memory.available)
                result["used_formatted"] = _format_bytes(memory.used)
                result["free_formatted"] = _format_bytes(memory.free)

            if include_swap:
                swap = psutil.swap_memory()
                result["swap"] = {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent_used": swap.percent,
                }

                if human_readable:
                    result["swap"]["total_formatted"] = _format_bytes(swap.total)
                    result["swap"]["used_formatted"] = _format_bytes(swap.used)
                    result["swap"]["free_formatted"] = _format_bytes(swap.free)

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get memory info: {str(e)}"}

    @staticmethod
    def get_cpu_info(
        include_per_cpu: bool = False,
        interval: float = 1.0,
    ) -> dict[str, Any]:
        """
        Get CPU information and usage statistics.

        Args:
            include_per_cpu: If True, include per-CPU core statistics
            interval: Time interval in seconds to measure CPU usage

        Returns:
            Dictionary with CPU information
        """
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=interval)

            result = {
                "status": "success",
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": cpu_percent,
                "frequency_current": cpu_freq.current if cpu_freq else None,
                "frequency_min": cpu_freq.min if cpu_freq else None,
                "frequency_max": cpu_freq.max if cpu_freq else None,
            }

            if include_per_cpu:
                result["per_cpu_usage"] = psutil.cpu_percent(interval=interval, percpu=True)

            # CPU times
            cpu_times = psutil.cpu_times()
            result["cpu_times"] = {
                "user": cpu_times.user,
                "system": cpu_times.system,
                "idle": cpu_times.idle,
            }

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get CPU info: {str(e)}"}

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            Dictionary with operating system and platform details
        """
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time

            result = {
                "status": "success",
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "boot_time": boot_time.isoformat(),
                "uptime_seconds": int(uptime.total_seconds()),
                "uptime_formatted": str(uptime).split(".")[0],  # Remove microseconds
            }

            # Add OS-specific information
            if platform.system() == "Darwin":  # macOS
                result["os_name"] = "macOS"
                result["os_version"] = platform.mac_ver()[0]
            elif platform.system() == "Windows":
                result["os_name"] = "Windows"
                result["os_version"] = platform.win32_ver()[0]
            elif platform.system() == "Linux":
                result["os_name"] = "Linux"
                try:
                    import distro
                    result["distribution"] = distro.name()
                    result["distribution_version"] = distro.version()
                except ImportError:
                    result["distribution"] = "Unknown"

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get system info: {str(e)}"}

    @staticmethod
    def get_network_info(
        include_stats: bool = True,
    ) -> dict[str, Any]:
        """
        Get network interfaces and statistics.

        Args:
            include_stats: If True, include network I/O statistics

        Returns:
            Dictionary with network information
        """
        try:
            interfaces = psutil.net_if_addrs()
            interface_list = []

            for interface_name, addresses in interfaces.items():
                interface_info = {
                    "name": interface_name,
                    "addresses": [],
                }

                for addr in addresses:
                    addr_info = {
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast,
                    }
                    interface_info["addresses"].append(addr_info)

                interface_list.append(interface_info)

            result = {
                "status": "success",
                "interfaces": interface_list,
            }

            if include_stats:
                net_io = psutil.net_io_counters()
                result["io_stats"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_received": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_received": net_io.packets_recv,
                    "errors_in": net_io.errin,
                    "errors_out": net_io.errout,
                    "drops_in": net_io.dropin,
                    "drops_out": net_io.dropout,
                }

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get network info: {str(e)}"}

    @staticmethod
    def get_process_list(
        sort_by: str = "memory",
        limit: int = 10,
        include_details: bool = False,
    ) -> dict[str, Any]:
        """
        Get list of running processes.

        Args:
            sort_by: Sort criterion - "memory", "cpu", "name", "pid"
            limit: Maximum number of processes to return
            include_details: If True, include detailed process information

        Returns:
            Dictionary with process list
        """
        try:
            processes = []

            for proc in psutil.process_iter(["pid", "name", "username", "memory_percent", "cpu_percent"]):
                try:
                    proc_info = proc.info
                    process_data = {
                        "pid": proc_info["pid"],
                        "name": proc_info["name"],
                        "username": proc_info["username"],
                        "memory_percent": proc_info["memory_percent"],
                        "cpu_percent": proc_info["cpu_percent"],
                    }

                    if include_details:
                        proc_detail = proc.as_dict(attrs=["status", "create_time", "num_threads"])
                        process_data["status"] = proc_detail.get("status")
                        process_data["num_threads"] = proc_detail.get("num_threads")
                        if proc_detail.get("create_time"):
                            process_data["create_time"] = datetime.fromtimestamp(
                                proc_detail["create_time"]
                            ).isoformat()

                    processes.append(process_data)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort processes
            sort_keys = {
                "memory": lambda x: x.get("memory_percent") or 0,
                "cpu": lambda x: x.get("cpu_percent") or 0,
                "name": lambda x: x.get("name", "").lower(),
                "pid": lambda x: x.get("pid") or 0,
            }

            if sort_by in sort_keys:
                processes.sort(key=sort_keys[sort_by], reverse=(sort_by in ["memory", "cpu"]))

            return {
                "status": "success",
                "total_processes": len(processes),
                "processes": processes[:limit],
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to get process list: {str(e)}"}

    @staticmethod
    def get_env_variable(
        name: str,
        default: str | None = None,
    ) -> dict[str, Any]:
        """
        Get environment variable value.

        Args:
            name: Environment variable name
            default: Default value if variable is not set

        Returns:
            Dictionary with variable value
        """
        try:
            value = os.getenv(name, default)

            return {
                "status": "success",
                "name": name,
                "value": value,
                "exists": name in os.environ,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to get environment variable: {str(e)}"}

    @staticmethod
    def get_all_env_variables(
        filter_pattern: str | None = None,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Get all environment variables.

        Args:
            filter_pattern: Optional pattern to filter variable names
            case_sensitive: If True, case-sensitive pattern matching

        Returns:
            Dictionary with all environment variables
        """
        try:
            import fnmatch

            env_vars = {}

            for key, value in os.environ.items():
                if filter_pattern:
                    pattern = filter_pattern if case_sensitive else filter_pattern.lower()
                    name = key if case_sensitive else key.lower()
                    if not fnmatch.fnmatch(name, pattern):
                        continue

                env_vars[key] = value

            return {
                "status": "success",
                "count": len(env_vars),
                "variables": env_vars,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to get environment variables: {str(e)}"}

    @staticmethod
    def get_battery_info() -> dict[str, Any]:
        """
        Get battery information (for laptops/mobile devices).

        Returns:
            Dictionary with battery status and charge level
        """
        try:
            battery = psutil.sensors_battery()

            if battery is None:
                return {
                    "status": "success",
                    "has_battery": False,
                    "message": "No battery detected (desktop system or battery not accessible)",
                }

            result = {
                "status": "success",
                "has_battery": True,
                "percent": battery.percent,
                "power_plugged": battery.power_plugged,
                "time_left_seconds": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
            }

            if battery.secsleft not in [psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN]:
                hours, remainder = divmod(battery.secsleft, 3600)
                minutes = remainder // 60
                result["time_left_formatted"] = f"{int(hours)}h {int(minutes)}m"

            return result

        except Exception as e:
            return {"status": "error", "message": f"Failed to get battery info: {str(e)}"}

    @staticmethod
    def get_temperature_sensors() -> dict[str, Any]:
        """
        Get temperature sensor readings (if available).

        Returns:
            Dictionary with temperature sensor data
        """
        try:
            temps = psutil.sensors_temperatures()

            if not temps:
                return {
                    "status": "success",
                    "has_sensors": False,
                    "message": "No temperature sensors detected or not accessible",
                }

            sensors = {}
            for name, entries in temps.items():
                sensors[name] = []
                for entry in entries:
                    sensor_data = {
                        "label": entry.label or "Unknown",
                        "current": entry.current,
                        "high": entry.high,
                        "critical": entry.critical,
                    }
                    sensors[name].append(sensor_data)

            return {
                "status": "success",
                "has_sensors": True,
                "sensors": sensors,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to get temperature sensors: {str(e)}"}