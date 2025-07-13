"""
Discord Command Mixins

Provides threading and performance utilities for Discord bot commands.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional
from discord.ext import commands

logger = logging.getLogger(__name__)


class ThreadedCommandMixin:
    """
    Mixin that adds thread pool execution to Discord commands.
    
    This allows sync operations to run without blocking the Discord bot's event loop.
    Perfect for database queries and analysis operations.
    
    Usage:
        class MyCommands(ThreadedCommandMixin, commands.Cog):
            async def my_command(self, ctx):
                result = await self.run_sync_in_thread(self._sync_operation, arg1, arg2)
                await ctx.send(result)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=5,
            thread_name_prefix="discord_cmd"
        )
        
        # Track active operations for monitoring
        self._active_operations = 0
        
        logger.info("ThreadedCommandMixin initialized with 5 workers")
    
    async def run_sync_in_thread(
        self, 
        sync_function: Callable, 
        *args, 
        timeout: float = 30.0,
        **kwargs
    ) -> Any:
        """
        Execute a synchronous function in the thread pool.
        
        Args:
            sync_function: The sync function to execute
            *args: Positional arguments for the function
            timeout: Maximum time to wait (default: 30 seconds)
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the sync function
            
        Raises:
            asyncio.TimeoutError: If operation takes longer than timeout
            Exception: Any exception raised by the sync function
        """
        
        self._active_operations += 1
        operation_id = self._active_operations
        
        logger.debug(f"Starting sync operation {operation_id}: {sync_function.__name__}")
        
        try:
            # Submit to thread pool and wait with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    lambda: sync_function(*args, **kwargs)
                ),
                timeout=timeout
            )
            
            logger.debug(f"Completed sync operation {operation_id}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Sync operation {operation_id} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Sync operation {operation_id} failed: {e}")
            raise
        finally:
            self._active_operations -= 1
    
    async def run_multiple_sync_in_thread(
        self,
        operations: Dict[str, tuple],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute multiple sync operations concurrently in thread pool.
        
        Args:
            operations: Dict of {name: (function, args, kwargs)}
            timeout: Maximum time to wait for all operations
            
        Returns:
            Dict of {name: result}
            
        Example:
            results = await self.run_multiple_sync_in_thread({
                'channel_data': (analyzer.analyze_channel, ('general',), {}),
                'user_data': (analyzer.get_top_users, (), {'limit': 10})
            })
        """
        
        logger.debug(f"Starting {len(operations)} concurrent sync operations")
        
        # Create tasks for all operations
        tasks = {}
        for name, (func, args, kwargs) in operations.items():
            task = self.run_sync_in_thread(func, *args, timeout=timeout, **kwargs)
            tasks[name] = task
        
        try:
            # Wait for all to complete
            results = {}
            for name, task in tasks.items():
                results[name] = await task
            
            logger.debug(f"Completed all {len(operations)} sync operations")
            return results
            
        except Exception as e:
            logger.error(f"Multiple sync operations failed: {e}")
            # Cancel remaining tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            raise
    
    def get_thread_pool_status(self) -> Dict[str, Any]:
        """Get current thread pool status for monitoring."""
        return {
            'max_workers': self.thread_pool._max_workers,
            'active_operations': self._active_operations,
            'thread_pool_active': not self.thread_pool._shutdown
        }
    
    async def cleanup_thread_pool(self):
        """Clean up thread pool resources. Call this when shutting down."""
        logger.info("Shutting down thread pool...")
        
        # Wait for active operations to complete (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_operations_complete(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some operations didn't complete during shutdown")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("Thread pool shutdown complete")
    
    async def _wait_for_operations_complete(self):
        """Wait for all active operations to complete."""
        while self._active_operations > 0:
            logger.debug(f"Waiting for {self._active_operations} operations to complete...")
            await asyncio.sleep(0.1)


class PerformanceMonitorMixin:
    """
    Mixin that adds performance monitoring to Discord commands.
    
    Tracks command execution times and provides metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._command_metrics = {}
        logger.info("PerformanceMonitorMixin initialized")
    
    async def track_command_performance(
        self,
        command_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, float]:
        """
        Execute operation and track its performance.
        
        Returns:
            tuple: (result, execution_time_seconds)
        """
        import time
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Store metrics
            if command_name not in self._command_metrics:
                self._command_metrics[command_name] = []
            
            self._command_metrics[command_name].append(execution_time)
            
            # Keep only last 100 executions per command
            if len(self._command_metrics[command_name]) > 100:
                self._command_metrics[command_name] = self._command_metrics[command_name][-100:]
            
            logger.debug(f"Command {command_name} completed in {execution_time:.2f}s")
            return result, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Command {command_name} failed after {execution_time:.2f}s: {e}")
            raise
    
    def get_command_metrics(self, command_name: str = None) -> Dict[str, Any]:
        """Get performance metrics for commands."""
        if command_name:
            times = self._command_metrics.get(command_name, [])
            if not times:
                return {"error": f"No metrics for command {command_name}"}
            
            return {
                "command": command_name,
                "executions": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "recent_time": times[-1] if times else None
            }
        else:
            # Return summary for all commands
            summary = {}
            for cmd, times in self._command_metrics.items():
                if times:
                    summary[cmd] = {
                        "executions": len(times),
                        "avg_time": sum(times) / len(times),
                        "recent_time": times[-1]
                    }
            return summary


class ComprehensiveCommandMixin(ThreadedCommandMixin, PerformanceMonitorMixin):
    """
    Combined mixin providing both threading and performance monitoring.
    
    This is the recommended base for Discord analysis commands.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("ComprehensiveCommandMixin initialized (threading + performance)")
    
    async def execute_tracked_sync_operation(
        self,
        command_name: str,
        sync_function: Callable,
        *args,
        timeout: float = 30.0,
        **kwargs
    ) -> tuple[Any, float]:
        """
        Execute sync operation in thread pool with performance tracking.
        
        This is the main method you'll use in Discord commands.
        
        Returns:
            tuple: (result, execution_time)
        """
        
        async def tracked_operation():
            return await self.run_sync_in_thread(
                sync_function, *args, timeout=timeout, **kwargs
            )
        
        return await self.track_command_performance(
            command_name, tracked_operation
        ) 