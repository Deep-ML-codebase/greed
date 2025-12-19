/**
 * Pyodide Web Worker - Runs Python code in a separate thread
 * Prevents UI blocking during long-running computations
 *
 * This worker:
 * 1. Loads and initializes Pyodide
 * 2. Executes Python code asynchronously
 * 3. Communicates results back to main thread
 * 4. Manages package loading and global state
 * 5. Maintains execution context across runs
 * 6. Performs automatic memory cleanup
 */

// Context Manager (inline to avoid import issues)
class ContextManager {
  constructor() {
    this.stats = {
      executionCount: 0,
      lastCleanup: Date.now(),
      memoryPressure: 0,
      avgExecutionTime: 0
    };

    this.config = {
      cleanupInterval: 100,
      memoryThreshold: 0.8,
      maxIdleTime: 5 * 60 * 1000,
      preserveGlobals: new Set([
        'torch', 'nn', 'optim', 'numpy', 'np', 'sys', 'os', 'math',
        'gc', 'builtins', '__builtins__', '__name__', '__doc__',
        '__package__', '__loader__', '__spec__', '__annotations__',
        'model', 'optimizer', 'loss_fn', 'dataset', 'dataloader',
        'train', 'test', 'val', 'X', 'y', 'data',
        'losses', 'accuracies', 'epochs', 'history', 'results'
      ])
    };

    this.userContext = new Map();
    this.lastAccessTime = Date.now();
  }

  preserve(name) {
    this.config.preserveGlobals.add(name);
  }

  shouldPreserve(name) {
    if (this.config.preserveGlobals.has(name)) return true;
    const lastAccess = this.userContext.get(name);
    if (lastAccess && (Date.now() - lastAccess) < this.config.maxIdleTime) return true;
    return false;
  }

  accessed(name) {
    this.userContext.set(name, Date.now());
    this.lastAccessTime = Date.now();
  }

  recordExecution(duration) {
    this.stats.executionCount++;
    this.stats.avgExecutionTime =
      (this.stats.avgExecutionTime * (this.stats.executionCount - 1) + duration) /
      this.stats.executionCount;
  }

  needsCleanup() {
    if (this.stats.executionCount % this.config.cleanupInterval === 0) return true;
    if (this.stats.memoryPressure > this.config.memoryThreshold) return true;
    const idleTime = Date.now() - this.lastAccessTime;
    if (idleTime > this.config.maxIdleTime) return true;
    return false;
  }

  getCleanupCode() {
    const preservedVars = Array.from(this.config.preserveGlobals).join('", "');
    return `
import gc
import sys
preserved_globals = {"${preservedVars}"}
current_globals = list(globals().keys())
deleted_count = 0
for var_name in current_globals:
    if var_name in preserved_globals:
        continue
    if var_name.startswith('__') and var_name.endswith('__'):
        continue
    if var_name in sys.modules:
        continue
    if var_name.startswith('_temp_') or var_name.startswith('_out_'):
        try:
            del globals()[var_name]
            deleted_count += 1
        except:
            pass
gc.collect()
_cleanup_stats = {'deleted': deleted_count, 'memory_freed': True}
`;
  }

  getMemoryOptimizationCode() {
    return `
import gc
import sys
def _cleanup_tensors():
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.grad is not None:
                if not obj.requires_grad:
                    obj.grad = None
    except:
        pass
_cleanup_tensors()
gc.collect(generation=2)
_memory_optimized = True
`;
  }

  getStats() {
    return {
      ...this.stats,
      idleTime: Date.now() - this.lastAccessTime,
      preservedVariables: Array.from(this.config.preserveGlobals)
    };
  }

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
}

// Worker global state
let pyodide = null;
let isInitialized = false;
let installedPackages = new Set();
let initializationPromise = null;
let contextManager = null;

/**
 * Initialize Pyodide in the worker context
 */
async function initializePyodide(config) {
  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {
    try {
      // Import Pyodide from CDN
      importScripts(config.pyodideURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');

      // Send progress update
      postMessage({
        type: 'init:progress',
        stage: 'loading',
        message: 'Loading Pyodide...'
      });

      // Load Pyodide
      pyodide = await loadPyodide({
        indexURL: config.indexURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
      });

      postMessage({
        type: 'init:progress',
        stage: 'loaded',
        message: 'Pyodide loaded successfully'
      });

      // Load preload packages
      if (config.preloadPackages && config.preloadPackages.length > 0) {
        postMessage({
          type: 'init:progress',
          stage: 'packages',
          message: `Loading packages: ${config.preloadPackages.join(', ')}`
        });

        await pyodide.loadPackage(config.preloadPackages);
        config.preloadPackages.forEach(pkg => installedPackages.add(pkg));

        postMessage({
          type: 'init:progress',
          stage: 'packages-loaded',
          message: `Packages loaded: ${config.preloadPackages.join(', ')}`
        });
      }

      // Initialize context manager
      contextManager = new ContextManager();

      isInitialized = true;

      postMessage({
        type: 'init:complete',
        installedPackages: Array.from(installedPackages)
      });

      return true;
    } catch (error) {
      postMessage({
        type: 'init:error',
        error: {
          message: error.message,
          stack: error.stack
        }
      });
      throw error;
    }
  })();

  return initializationPromise;
}

/**
 * Load Python packages
 */
async function loadPackages(packages) {
  if (!isInitialized) {
    throw new Error('Pyodide not initialized');
  }

  const packagesToLoad = packages.filter(pkg => !installedPackages.has(pkg));

  if (packagesToLoad.length === 0) {
    return Array.from(installedPackages);
  }

  try {
    postMessage({
      type: 'packages:loading',
      packages: packagesToLoad
    });

    await pyodide.loadPackage(packagesToLoad);
    packagesToLoad.forEach(pkg => installedPackages.add(pkg));

    postMessage({
      type: 'packages:loaded',
      packages: packagesToLoad,
      allPackages: Array.from(installedPackages)
    });

    return Array.from(installedPackages);
  } catch (error) {
    postMessage({
      type: 'packages:error',
      error: {
        message: error.message,
        packages: packagesToLoad
      }
    });
    throw error;
  }
}

/**
 * Execute Python code with optional output capture
 */
async function executePython(taskId, code, options = {}) {
  if (!isInitialized) {
    throw new Error('Pyodide not initialized');
  }

  const {
    captureOutput = false,
    globals = {},
    validateInput = true
  } = options;

  const startTime = performance.now();

  try {
    // Set globals if provided
    for (const [key, value] of Object.entries(globals)) {
      try {
        pyodide.globals.set(key, value);
        contextManager.accessed(key); // Track access
      } catch (error) {
        postMessage({
          type: 'execution:warning',
          taskId,
          warning: `Failed to set global '${key}': ${error.message}`
        });
      }
    }

    let result;

    if (captureOutput) {
      // Extract streamOutput option (default to true)
      const shouldStream = options.streamOutput !== false;

      // Inject streaming callback into Python globals if streaming enabled
      if (shouldStream && taskId) {
        pyodide.globals.set('__greed_worker_emit_stdout__', (tid, output) => {
          postMessage({
            type: 'execution:stdout',
            taskId: tid,
            output: output,
            timestamp: Date.now()
          });
        });
      }

      // Capture stdout with streaming support
      const outputCode = `
import sys
from io import StringIO

class StreamingBuffer:
    def __init__(self, task_id, emit_callback, should_stream):
        self.buffer = StringIO()
        self.task_id = task_id
        self.emit_callback = emit_callback
        self.should_stream = should_stream

    def write(self, text):
        self.buffer.write(text)
        # Emit immediately for real-time streaming
        if self.should_stream and self.emit_callback and text:
            # Emit immediately - don't wait for time intervals
            # This ensures output appears in real-time, even during sleep() calls
            self.emit_callback(self.task_id, text)

    def flush(self):
        self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()

_temp_output_buffer = StreamingBuffer('${taskId || ''}', ${shouldStream ? '__greed_worker_emit_stdout__' : 'None'}, ${shouldStream ? 'True' : 'False'})
_temp_original_stdout = sys.stdout
sys.stdout = _temp_output_buffer

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
finally:
    sys.stdout.flush()
    sys.stdout = _temp_original_stdout
    _temp_captured_output = _temp_output_buffer.getvalue()
    # Debug: also store in a backup variable
    _temp_backup_output = _temp_captured_output
`;

      await pyodide.runPythonAsync(outputCode);

      let capturedOutput = '';
      try {
        // Try both variables
        capturedOutput = pyodide.globals.get('_temp_captured_output') || '';
        const backupOutput = pyodide.globals.get('_temp_backup_output') || '';
        const bufferValue = pyodide.runPython('_temp_output_buffer.getvalue()');

        // Send to main thread console via postMessage
        postMessage({
          type: 'execution:warning',
          taskId,
          warning: `[Worker] _temp_captured_output: "${capturedOutput}" (len: ${capturedOutput.length}), backup: "${backupOutput}" (len: ${backupOutput.length}), buffer: "${bufferValue}" (len: ${bufferValue.length})`
        });
      } catch (error) {
        capturedOutput = 'Output capture failed';
        postMessage({
          type: 'execution:warning',
          taskId,
          warning: `[Worker] Capture failed: ${error.message}`
        });
      }

      result = { output: capturedOutput };
      postMessage({
        type: 'execution:warning',
        taskId,
        warning: `[Worker] Result object: ${JSON.stringify(result)}`
      });

      // Clean up temporary globals (prefixed with _temp_)
      try {
        pyodide.globals.delete('_temp_captured_output');
        pyodide.globals.delete('_temp_output_buffer');
        pyodide.globals.delete('_temp_original_stdout');
      } catch (error) {
        // Ignore cleanup errors
      }
    } else {
      // Direct execution
      result = await pyodide.runPythonAsync(code);
    }

    // Record execution for stats
    const duration = performance.now() - startTime;
    contextManager.recordExecution(duration);

    // Perform automatic cleanup if needed
    if (contextManager.needsCleanup()) {
      try {
        await pyodide.runPythonAsync(contextManager.getCleanupCode());

        // Serialize stats to ensure it's cloneable
        const stats = contextManager.getStats();
        postMessage({
          type: 'execution:cleanup',
          taskId,
          stats: JSON.parse(JSON.stringify(stats))
        });
      } catch (cleanupError) {
        // Non-fatal - just log
        postMessage({
          type: 'execution:warning',
          taskId,
          warning: `Cleanup warning: ${cleanupError.message}`
        });
      }
    }

    // Memory optimization every 50 executions
    if (contextManager.stats.executionCount % 50 === 0) {
      try {
        await pyodide.runPythonAsync(contextManager.getMemoryOptimizationCode());
      } catch (error) {
        // Non-fatal
      }
    }

    // Convert PyProxy objects to JavaScript before sending
    let serializedResult = result;
    if (result && typeof result === 'object' && result.toJs) {
      // It's a PyProxy - convert to JS
      try {
        serializedResult = result.toJs({ dict_converter: Object.fromEntries });
      } catch (error) {
        // If conversion fails, try toString
        serializedResult = result.toString();
      }
    }

    postMessage({
      type: 'execution:complete',
      taskId,
      result: serializedResult,
      stats: {
        duration,
        executionCount: contextManager.stats.executionCount
      }
    });

    return result;
  } catch (error) {
    postMessage({
      type: 'execution:error',
      taskId,
      error: {
        message: error.message,
        stack: error.stack,
        type: error.constructor.name
      }
    });
    throw error;
  }
}

/**
 * Get a global variable from Python
 */
function getGlobal(name) {
  if (!isInitialized) {
    throw new Error('Pyodide not initialized');
  }

  try {
    return pyodide.globals.get(name);
  } catch (error) {
    return undefined;
  }
}

/**
 * Set a global variable in Python
 */
function setGlobal(name, value) {
  if (!isInitialized) {
    throw new Error('Pyodide not initialized');
  }

  try {
    pyodide.globals.set(name, value);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Delete a global variable from Python
 */
function deleteGlobal(name) {
  if (!isInitialized) {
    throw new Error('Pyodide not initialized');
  }

  try {
    pyodide.globals.delete(name);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Interrupt current execution
 */
function interrupt() {
  if (!isInitialized || !pyodide) {
    return;
  }

  try {
    pyodide.interruptBuffer[0] = 2;
    postMessage({
      type: 'execution:interrupted',
      message: 'Execution interrupted by user'
    });
  } catch (error) {
    postMessage({
      type: 'interrupt:error',
      error: {
        message: error.message
      }
    });
  }
}

/**
 * Reset the Python environment
 */
async function reset() {
  if (!isInitialized) {
    return;
  }

  try {
    // Clear all user-defined globals
    await pyodide.runPythonAsync(`
import sys
# Get all user-defined names
user_names = [name for name in dir() if not name.startswith('_')]
# Delete them
for name in user_names:
    try:
        del globals()[name]
    except:
        pass
`);

    postMessage({
      type: 'reset:complete',
      message: 'Python environment reset'
    });
  } catch (error) {
    postMessage({
      type: 'reset:error',
      error: {
        message: error.message
      }
    });
  }
}

/**
 * Message handler - receives commands from main thread
 */
self.onmessage = async function(event) {
  const { type, id, ...data } = event.data;

  try {
    switch (type) {
      case 'init':
        await initializePyodide(data.config);
        postMessage({ type: 'init:ack', id });
        break;

      case 'loadPackages':
        await loadPackages(data.packages);
        postMessage({ type: 'loadPackages:ack', id, packages: Array.from(installedPackages) });
        break;

      case 'execute':
        await executePython(data.taskId, data.code, data.options);
        postMessage({ type: 'execute:ack', id, taskId: data.taskId });
        break;

      case 'getGlobal':
        const value = getGlobal(data.name);
        postMessage({ type: 'getGlobal:result', id, name: data.name, value });
        break;

      case 'setGlobal':
        const setResult = setGlobal(data.name, data.value);
        postMessage({ type: 'setGlobal:result', id, name: data.name, success: setResult });
        break;

      case 'deleteGlobal':
        const deleteResult = deleteGlobal(data.name);
        postMessage({ type: 'deleteGlobal:result', id, name: data.name, success: deleteResult });
        break;

      case 'interrupt':
        interrupt();
        postMessage({ type: 'interrupt:ack', id });
        break;

      case 'reset':
        await reset();
        postMessage({ type: 'reset:ack', id });
        break;

      case 'ping':
        postMessage({ type: 'pong', id });
        break;

      default:
        postMessage({
          type: 'error',
          id,
          error: { message: `Unknown message type: ${type}` }
        });
    }
  } catch (error) {
    postMessage({
      type: 'error',
      id,
      error: {
        message: error.message,
        stack: error.stack
      }
    });
  }
};

// Signal that worker is ready
try {
  postMessage({ type: 'worker:ready' });
} catch (error) {
  // If we can't even send the ready message, something is very wrong
  console.error('Worker failed to send ready message:', error);
}
