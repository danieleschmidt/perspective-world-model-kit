"""Simple health check utilities for PWMK system status."""

from typing import Dict, List, Any, Optional
import time
import psutil
import torch
from pathlib import Path

from .logging import get_logger
from .monitoring import get_metrics_collector


class HealthChecker:
    """
    Simple health monitoring for PWMK components.
    
    Provides essential system health checks without complex dependencies.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        start_time = time.time()
        
        health_status = {
            "timestamp": int(time.time()),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check CPU and Memory
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            health_status["checks"]["system"] = {
                "status": "healthy" if cpu_percent < 90 and memory.percent < 90 else "warning",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            }
        except Exception as e:
            health_status["checks"]["system"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "error"
            
        # Check PyTorch/GPU
        try:
            torch_status = {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
            
            if torch.cuda.is_available():
                torch_status["cuda_version"] = torch.version.cuda
                torch_status["gpu_count"] = torch.cuda.device_count()
                torch_status["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
            health_status["checks"]["pytorch"] = {
                "status": "healthy",
                **torch_status
            }
        except Exception as e:
            health_status["checks"]["pytorch"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
            
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('/')
            health_status["checks"]["storage"] = {
                "status": "healthy" if disk_usage.percent < 90 else "warning",
                "disk_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024**3)
            }
        except Exception as e:
            health_status["checks"]["storage"] = {
                "status": "error", 
                "error": str(e)
            }
            
        # Check PWMK modules
        health_status["checks"]["pwmk_modules"] = self._check_pwmk_modules()
        
        check_duration = time.time() - start_time
        health_status["check_duration_seconds"] = check_duration
        
        self.logger.info(f"Health check completed in {check_duration:.2f}s - Status: {health_status['overall_status']}")
        
        return health_status
    
    def _check_pwmk_modules(self) -> Dict[str, Any]:
        """Check availability of key PWMK modules."""
        modules_status = {"status": "healthy"}
        
        critical_modules = [
            "pwmk.core.world_model",
            "pwmk.core.beliefs", 
            "pwmk.envs.simple_grid",
            "pwmk.utils.validation"
        ]
        
        optional_modules = [
            "pwmk.quantum",
            "pwmk.autonomous", 
            "pwmk.breakthrough",
            "pwmk.revolution"
        ]
        
        modules_status["critical"] = {}
        modules_status["optional"] = {}
        
        # Check critical modules
        for module in critical_modules:
            try:
                __import__(module)
                modules_status["critical"][module] = "available"
            except ImportError as e:
                modules_status["critical"][module] = f"missing: {str(e)}"
                modules_status["status"] = "error"
                
        # Check optional modules
        for module in optional_modules:
            try:
                __import__(module)
                modules_status["optional"][module] = "available"
            except ImportError:
                modules_status["optional"][module] = "not installed"
                
        return modules_status
    
    def quick_health_check(self) -> bool:
        """Quick boolean health check for scripts and automation."""
        try:
            status = self.check_system_health()
            return status["overall_status"] in ["healthy", "degraded"]
        except Exception:
            return False
    
    def get_health_summary(self) -> str:
        """Get human-readable health summary."""
        health = self.check_system_health()
        
        summary_lines = [
            f"🏥 PWMK System Health: {health['overall_status'].upper()}",
            f"⏱️  Check completed in {health['check_duration_seconds']:.2f}s",
            ""
        ]
        
        # System resources
        if "system" in health["checks"]:
            sys = health["checks"]["system"]
            if sys["status"] != "error":
                summary_lines.extend([
                    f"💻 System Resources:",
                    f"   CPU: {sys['cpu_percent']:.1f}%",
                    f"   Memory: {sys['memory_percent']:.1f}% ({sys['memory_available_gb']:.1f}GB free)",
                    ""
                ])
        
        # PyTorch status
        if "pytorch" in health["checks"]:
            pt = health["checks"]["pytorch"]
            if pt["status"] != "error":
                gpu_info = f" | GPU: {pt.get('gpu_count', 0)} devices" if pt.get("cuda_available") else " | No GPU"
                summary_lines.extend([
                    f"🔥 PyTorch: {pt['torch_version']}{gpu_info}",
                    ""
                ])
        
        # PWMK modules
        if "pwmk_modules" in health["checks"]:
            modules = health["checks"]["pwmk_modules"]
            critical_ok = sum(1 for status in modules["critical"].values() if status == "available")
            critical_total = len(modules["critical"])
            optional_ok = sum(1 for status in modules["optional"].values() if status == "available") 
            optional_total = len(modules["optional"])
            
            summary_lines.extend([
                f"📦 PWMK Modules:",
                f"   Critical: {critical_ok}/{critical_total} available",
                f"   Optional: {optional_ok}/{optional_total} available",
                ""
            ])
            
        return "\n".join(summary_lines)


def quick_health_check() -> bool:
    """Standalone function for quick health verification."""
    checker = HealthChecker()
    return checker.quick_health_check()


def print_health_summary() -> None:
    """Print health summary to console."""
    checker = HealthChecker()
    print(checker.get_health_summary())


if __name__ == "__main__":
    print_health_summary()