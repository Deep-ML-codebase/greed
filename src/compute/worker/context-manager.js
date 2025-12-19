/**
 * Context Manager - Maintains Python execution context and manages memory
 *
 * Solves two critical issues:
 * 1. Context loss - preserves Python state across executions
 * 2. Memory leaks - automatic garbage collection and cleanup
 */

class ContextManager {
  constructor() {
    // Execution statistics
    this.stats = {
      executionCount: 0,
      lastCleanup: Date.now(),
      memoryPressure: 0,
      avgExecutionTime: 0
    };

    // Configuration
    this.config = {
      cleanupInterval: 100, // Cleanup every 100 executions
      memoryThreshold: 0.8, // Trigger cleanup at 80% memory
      maxIdleTime: 5 * 60 * 1000, // 5 minutes
      preserveGlobals: new Set([
        // Core Python modules
        'torch', 'nn', 'optim', 'numpy', 'np', 'sys', 'os', 'math',
        'gc', 'builtins', '__builtins__', '__name__', '__doc__',
        '__package__', '__loader__', '__spec__', '__annotations__',

        // User context (preserve across executions)
        'model', 'optimizer', 'loss_fn', 'dataset', 'dataloader',
        'train', 'test', 'val', 'X', 'y', 'data',

        // Common ML variables
        'losses', 'accuracies', 'epochs', 'history', 'results'
      ])
    };

    // Context preservation
    this.userContext = new Map(); // User-defined variables to preserve
    this.lastAccessTime = Date.now();
  }

  /**
   * Mark a variable as important to preserve
   */
  preserve(name) {
    this.config.preserveGlobals.add(name);
  }

  /**
   * Check if a variable should be preserved
   */
  shouldPreserve(name) {
    // Preserve if explicitly marked
    if (this.config.preserveGlobals.has(name)) {
      return true;
    }

    // Preserve if recently accessed
    const lastAccess = this.userContext.get(name);
    if (lastAccess && (Date.now() - lastAccess) < this.config.maxIdleTime) {
      return true;
    }

    return false;
  }

  /**
   * Track variable access
   */
  accessed(name) {
    this.userContext.set(name, Date.now());
    this.lastAccessTime = Date.now();
  }

  /**
   * Record execution
   */
  recordExecution(duration) {
    this.stats.executionCount++;
    this.stats.avgExecutionTime =
      (this.stats.avgExecutionTime * (this.stats.executionCount - 1) + duration) /
      this.stats.executionCount;
  }

  /**
   * Check if cleanup is needed
   */
  needsCleanup() {
    // Cleanup periodically
    if (this.stats.executionCount % this.config.cleanupInterval === 0) {
      return true;
    }

    // Cleanup on memory pressure
    if (this.stats.memoryPressure > this.config.memoryThreshold) {
      return true;
    }

    // Cleanup after long idle
    const idleTime = Date.now() - this.lastAccessTime;
    if (idleTime > this.config.maxIdleTime) {
      return true;
    }

    return false;
  }

  /**
   * Get cleanup code for Python execution
   */
  getCleanupCode() {
    const preservedVars = Array.from(this.config.preserveGlobals).join('", "');

    return `
import gc
import sys

# Preserved variables that should not be deleted
preserved_globals = {
    "${preservedVars}"
}

# Get current globals
current_globals = list(globals().keys())

# Track what we're deleting for debugging
deleted_count = 0

# Remove only temporary variables
for var_name in current_globals:
    # Keep if preserved
    if var_name in preserved_globals:
        continue

    # Keep if starts with __ (Python internals)
    if var_name.startswith('__') and var_name.endswith('__'):
        continue

    # Keep if in sys.modules (imported modules)
    if var_name in sys.modules:
        continue

    # Delete temporary variables
    if var_name.startswith('_temp_') or var_name.startswith('_out_'):
        try:
            del globals()[var_name]
            deleted_count += 1
        except:
            pass

# Force garbage collection to free memory
gc.collect()

# Return cleanup stats
_cleanup_stats = {
    'deleted': deleted_count,
    'memory_freed': True
}
`;
  }

  /**
   * Get memory optimization code
   */
  getMemoryOptimizationCode() {
    return `
import gc
import sys

# Aggressive memory cleanup for PyTorch tensors
def _cleanup_tensors():
    import torch
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear gradient caches
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.grad is not None:
            if not obj.requires_grad:  # Only clear if not needed
                obj.grad = None

# Run tensor cleanup
_cleanup_tensors()

# Clear Python's internal caches
gc.collect(generation=2)  # Full collection

# Return memory stats
_memory_optimized = True
`;
  }

  /**
   * Update memory pressure estimate
   */
  updateMemoryPressure(pressure) {
    this.stats.memoryPressure = pressure;
  }

  /**
   * Reset context manager
   */
  reset() {
    this.stats = {
      executionCount: 0,
      lastCleanup: Date.now(),
      memoryPressure: 0,
      avgExecutionTime: 0
    };
    this.userContext.clear();
    this.lastAccessTime = Date.now();
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      ...this.stats,
      idleTime: Date.now() - this.lastAccessTime,
      preservedVariables: Array.from(this.config.preserveGlobals)
    };
  }
}

// Export for use in worker
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ContextManager;
}

// Make available in worker context
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.ContextManager = ContextManager;
}
