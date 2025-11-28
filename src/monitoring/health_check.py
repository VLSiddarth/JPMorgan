"""
System Health Monitoring
Monitors system components, dependencies, and performance
"""

import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize health monitor
        
        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.component_health: Dict[str, ComponentHealth] = {}
        self.last_check: Optional[datetime] = None
        
    def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Run all health checks
        
        Returns:
            Dictionary of component health statuses
        """
        checks = [
            self.check_system_resources(),
            self.check_disk_space(),
            self.check_memory(),
            self.check_cpu(),
            self.check_database_connection(),
            self.check_redis_connection(),
            self.check_api_endpoints()
        ]
        
        for health in checks:
            self.component_health[health.component_name] = health
        
        self.last_check = datetime.now()
        return self.component_health
    
    def check_system_resources(self) -> ComponentHealth:
        """Check overall system resource health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine health status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage"
            elif cpu_percent > 75 or memory_percent > 75 or disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Resources normal"
            
            return ComponentHealth(
                component_name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                }
            )
        except Exception as e:
            logger.error(f"System resources check failed: {e}")
            return ComponentHealth(
                component_name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="Failed to check resources",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_disk_space(self) -> ComponentHealth:
        """Check disk space availability"""
        try:
            disk = psutil.disk_usage('/')
            gb_free = disk.free / (1024**3)
            percent_used = disk.percent
            
            if percent_used > 95 or gb_free < 1:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: Only {gb_free:.1f}GB free"
            elif percent_used > 85 or gb_free < 5:
                status = HealthStatus.DEGRADED
                message = f"Warning: {gb_free:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"{gb_free:.1f}GB free"
            
            return ComponentHealth(
                component_name="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': gb_free,
                    'percent_used': percent_used
                }
            )
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return ComponentHealth(
                component_name="disk_space",
                status=HealthStatus.UNKNOWN,
                message="Failed to check disk",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_memory(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            mem = psutil.virtual_memory()
            gb_available = mem.available / (1024**3)
            
            if mem.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory: {mem.percent:.1f}% used"
            elif mem.percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High memory: {mem.percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory: {mem.percent:.1f}% used"
            
            return ComponentHealth(
                component_name="memory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'total_gb': mem.total / (1024**3),
                    'available_gb': gb_available,
                    'used_percent': mem.percent
                }
            )
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return ComponentHealth(
                component_name="memory",
                status=HealthStatus.UNKNOWN,
                message="Failed to check memory",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_cpu(self) -> ComponentHealth:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical CPU: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High CPU: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU: {cpu_percent:.1f}%"
            
            return ComponentHealth(
                component_name="cpu",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'per_cpu': psutil.cpu_percent(percpu=True)
                }
            )
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return ComponentHealth(
                component_name="cpu",
                status=HealthStatus.UNKNOWN,
                message="Failed to check CPU",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_database_connection(self, test_query: bool = True) -> ComponentHealth:
        """
        Check database connection
        
        Args:
            test_query: Whether to run a test query
        """
        try:
            # This would use your actual DB connection
            # Placeholder implementation
            from pymongo import MongoClient
            from pymongo.errors import ServerSelectionTimeoutError
            
            client = MongoClient('mongodb://localhost:27017/', 
                               serverSelectionTimeoutMS=5000)
            
            # Test connection
            start = time.time()
            client.admin.command('ping')
            latency_ms = (time.time() - start) * 1000
            
            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
                message = f"High DB latency: {latency_ms:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"DB connected ({latency_ms:.0f}ms)"
            
            return ComponentHealth(
                component_name="database",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'latency_ms': latency_ms,
                    'connection': 'active'
                }
            )
            
        except ServerSelectionTimeoutError:
            return ComponentHealth(
                component_name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database unreachable",
                timestamp=datetime.now(),
                error="Connection timeout"
            )
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return ComponentHealth(
                component_name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database error",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_redis_connection(self) -> ComponentHealth:
        """Check Redis cache connection"""
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, 
                          socket_timeout=5, socket_connect_timeout=5)
            
            start = time.time()
            r.ping()
            latency_ms = (time.time() - start) * 1000
            
            info = r.info()
            memory_used_mb = info['used_memory'] / (1024**2)
            
            if latency_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"High Redis latency: {latency_ms:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis connected ({latency_ms:.0f}ms)"
            
            return ComponentHealth(
                component_name="redis",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'latency_ms': latency_ms,
                    'memory_used_mb': memory_used_mb,
                    'connected_clients': info['connected_clients']
                }
            )
            
        except redis.ConnectionError:
            return ComponentHealth(
                component_name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Redis unreachable",
                timestamp=datetime.now(),
                error="Connection failed"
            )
        except Exception as e:
            logger.error(f"Redis check failed: {e}")
            return ComponentHealth(
                component_name="redis",
                status=HealthStatus.UNKNOWN,
                message="Redis check error",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def check_api_endpoints(self) -> ComponentHealth:
        """Check external API health"""
        endpoints = {
            'fred': 'https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=test',
            'yahoo': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL',
        }
        
        failed = []
        slow = []
        
        for name, url in endpoints.items():
            try:
                start = time.time()
                response = requests.get(url, timeout=10)
                latency_ms = (time.time() - start) * 1000
                
                if response.status_code != 200:
                    failed.append(name)
                elif latency_ms > 5000:
                    slow.append(name)
                    
            except requests.RequestException:
                failed.append(name)
        
        if failed:
            status = HealthStatus.DEGRADED
            message = f"APIs failed: {', '.join(failed)}"
        elif slow:
            status = HealthStatus.DEGRADED
            message = f"Slow APIs: {', '.join(slow)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All APIs responsive"
        
        return ComponentHealth(
            component_name="api_endpoints",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={
                'failed': failed,
                'slow': slow,
                'total_checked': len(endpoints)
            }
        )
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status
        
        Returns:
            Worst status among all components
        """
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [h.status for h in self.component_health.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report
        
        Returns:
            Health report dictionary
        """
        return {
            'overall_status': self.get_overall_status().value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'components': {
                name: health.to_dict() 
                for name, health in self.component_health.items()
            },
            'summary': {
                'healthy': sum(1 for h in self.component_health.values() 
                             if h.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for h in self.component_health.values() 
                              if h.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for h in self.component_health.values() 
                               if h.status == HealthStatus.UNHEALTHY),
                'unknown': sum(1 for h in self.component_health.values() 
                             if h.status == HealthStatus.UNKNOWN)
            }
        }