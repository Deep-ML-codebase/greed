/**
 * RuntimeManager - Handles Pyodide initialization and Python package management
 * Supports both main-thread and Web Worker modes
 * Extracted from monolithic Greed class for better separation of concerns
 */
import EventEmitter from './event-emitter.js';
import PyodideWorkerManager from '../compute/worker/pyodide-worker-manager.js';

class RuntimeManager extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      pyodideIndexURL: config.pyodideIndexURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      preloadPackages: config.preloadPackages || ['numpy'],
      availablePackages: config.availablePackages || [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn',
        'plotly', 'seaborn', 'statsmodels', 'sympy', 'networkx'
      ],
      timeout: config.initTimeout || 30000,
      enableWorkers: config.enableWorkers !== false, // Worker mode by default
      ...config
    };

    // Runtime mode (worker or main-thread)
    this.mode = this.config.enableWorkers ? 'worker' : 'main';

    // Main-thread mode properties
    this.pyodide = null;

    // Worker mode manager
    this.workerManager = null;

    // Common properties
    this.isReady = false;
    this.installedPackages = new Set();
    this.initPromise = null;
  }

  /**
   * Initialize Pyodide runtime with error handling and progress tracking
   */
  async initialize() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this._initializeInternal();
    return this.initPromise;
  }

  async _initializeInternal() {
    if (this.mode === 'worker') {
      return this._initializeWorkerMode();
    } else {
      return this._initializeMainThreadMode();
    }
  }

  /**
   * Initialize in Web Worker mode (prevents UI blocking)
   */
  async _initializeWorkerMode() {
    try {
      this.emit('init:start', { stage: 'worker', mode: 'worker' });

      // Create worker manager
      this.workerManager = new PyodideWorkerManager({
        pyodideIndexURL: this.config.pyodideIndexURL,
        preloadPackages: this.config.preloadPackages,
        timeout: this.config.timeout
      });

      // Forward worker events
      this.workerManager.on('init:progress', (data) => this.emit('init:progress', data));
      this.workerManager.on('init:complete', (data) => {
        this.installedPackages = new Set(data.installedPackages);
        this.emit('init:complete', data);
      });
      this.workerManager.on('init:error', (data) => this.emit('init:error', data));
      this.workerManager.on('packages:loading', (data) => this.emit('packages:loading', data));
      this.workerManager.on('packages:loaded', (data) => this.emit('packages:loaded', data));
      this.workerManager.on('execution:complete', (data) => this.emit('execution:complete', data));
      this.workerManager.on('execution:error', (data) => this.emit('execution:error', data));
      this.workerManager.on('execution:stdout', (data) => this.emit('execution:stdout', data));

      // Initialize worker
      await this.workerManager.initialize();

      this.isReady = true;
      return true;
    } catch (error) {
      this.emit('init:error', { error, stage: 'worker-initialization' });
      throw error;
    }
  }

  /**
   * Initialize in main-thread mode (may block UI)
   */
  async _initializeMainThreadMode() {
    try {
      this.emit('init:start', { stage: 'pyodide', mode: 'main' });

      // Validate Pyodide availability
      if (typeof loadPyodide === 'undefined') {
        throw new Error('Pyodide not loaded. Please include pyodide.js in your HTML.');
      }

      // Initialize with timeout and 4GB memory limit
      const pyodidePromise = loadPyodide({
        indexURL: this.config.pyodideIndexURL,
        // Enable 4GB WASM memory for large models
        // Requires Memory64 support (Chrome 119+, Firefox 128+)
        args: ['-Xalloc-env=PYODIDE_WASM_MEMORY=4294967296']
      });

      this.pyodide = await Promise.race([
        pyodidePromise,
        this._createTimeoutPromise(this.config.timeout, 'Pyodide initialization timeout')
      ]);

      this.emit('init:progress', { stage: 'pyodide', status: 'loaded' });

      // Pre-load essential packages
      if (this.config.preloadPackages.length > 0) {
        this.emit('init:progress', { stage: 'packages', packages: this.config.preloadPackages });
        await this._loadPackages(this.config.preloadPackages);
      }

      this.isReady = true;
      this.emit('init:complete', { installedPackages: Array.from(this.installedPackages) });

      return true;
    } catch (error) {
      this.emit('init:error', { error, stage: 'initialization' });
      throw error;
    }
  }

  /**
   * Load Python packages with progress tracking
   */
  async loadPackages(packages) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }

    if (this.mode === 'worker') {
      const result = await this.workerManager.loadPackages(packages);
      result.forEach(pkg => this.installedPackages.add(pkg));
      return result;
    } else {
      return this._loadPackages(packages);
    }
  }

  async _loadPackages(packages) {
    const packagesToLoad = packages.filter(pkg => !this.installedPackages.has(pkg));

    if (packagesToLoad.length === 0) {
      return Array.from(this.installedPackages);
    }

    try {
      this.emit('packages:loading', { packages: packagesToLoad });

      await this.pyodide.loadPackage(packagesToLoad);

      packagesToLoad.forEach(pkg => this.installedPackages.add(pkg));

      this.emit('packages:loaded', {
        loaded: packagesToLoad,
        total: Array.from(this.installedPackages)
      });

      return Array.from(this.installedPackages);
    } catch (error) {
      // Log the error but don't throw - allows execution to continue
      this.emit('packages:error', { error, packages: packagesToLoad });

      // Mark packages as installed anyway to prevent retry loops
      packagesToLoad.forEach(pkg => this.installedPackages.add(pkg));

      return Array.from(this.installedPackages);
    }
  }

  /**
   * Execute Python code with error handling and context isolation
   */
  async runPython(code, options = {}) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }

    // Extract validateInput for security check, but pass full options to execution methods
    const { validateInput = true } = options;

    if (validateInput && this._containsDangerousPatterns(code)) {
      throw new SecurityError('Potentially dangerous code patterns detected');
    }

    // Auto-load packages if code imports them
    const packageChecks = [
      { check: this._needsMatplotlib.bind(this), loader: this._ensureMatplotlibLoaded.bind(this) },
      { check: this._needsPandas.bind(this), loader: this._ensurePandasLoaded.bind(this) },
      { check: this._needsScipy.bind(this), loader: this._ensureScipyLoaded.bind(this) },
      { check: this._needsPlotly.bind(this), loader: this._ensurePlotlyLoaded.bind(this) },
      { check: this._needsSklearn.bind(this), loader: this._ensureSklearnLoaded.bind(this) }
    ];

    // Check and load packages sequentially to avoid conflicts
    for (const { check, loader } of packageChecks) {
      if (check(code)) {
        await loader();
      }
    }

    // Delegate to worker or main thread - pass all options to preserve taskId, streamOutput, etc.
    if (this.mode === 'worker') {
      return this._runPythonWorker(code, options);
    } else {
      return this._runPythonMain(code, options);
    }
  }

  /**
   * Execute Python code in worker mode (non-blocking)
   */
  async _runPythonWorker(code, options) {
    try {
      const result = await this.workerManager.executePython(code, options);
      return result;
    } catch (error) {
      this.emit('execution:error', { error, code: code.substring(0, 100) });
      throw error;
    }
  }

  /**
   * Execute Python code on main thread (may block UI)
   */
  async _runPythonMain(code, options) {
    const { captureOutput, timeout, globals = {}, taskId, streamOutput } = options;

    try {

      // Set globals if provided - with error handling to prevent fatal errors
      for (const [key, value] of Object.entries(globals)) {
        try {
          this.pyodide.globals.set(key, value);
        } catch (globalError) {
          // Continue with other globals instead of failing completely
          this.emit('global:error', { key, error: globalError.message });
        }
      }

      let result;
      if (captureOutput) {
        // Extract taskId and streamOutput from options
        // Only enable streaming if taskId is provided AND streamOutput is not explicitly false
        const shouldStream = (streamOutput !== false) && Boolean(taskId);

        // Inject streaming callback into Python globals if streaming enabled
        if (shouldStream) {
          this.pyodide.globals.set('__greed_emit_stdout__', (tid, output) => {
            this.emit('execution:stdout', {
              type: 'execution:stdout',
              taskId: tid,
              output: output,
              timestamp: Date.now()
            });
          });
        } else {
          // No matplotlib available
        }

        // Enhanced stdout capture with streaming support
        const outputCode = `
import sys
from io import StringIO
import time

class StreamingBuffer:
    def __init__(self, task_id, emit_callback, should_stream):
        self.buffer = StringIO()
        self.task_id = task_id
        self.emit_callback = emit_callback
        self.should_stream = should_stream
        self.pending_output = ""

    def write(self, text):
        self.buffer.write(text)
        # Emit immediately for real-time streaming
        if self.should_stream and self.emit_callback and text:
            # Emit immediately - don't wait for time intervals
            # This ensures output appears in real-time, even during sleep() calls
            self.emit_callback(self.task_id, text)

    def flush(self):
        self.buffer.flush()
        # Emit any remaining output (usually not needed with immediate emission)
        if self.should_stream and self.emit_callback and self.pending_output:
            self.emit_callback(self.task_id, self.pending_output)
            self.pending_output = ""

    def getvalue(self):
        return self.buffer.getvalue()

_output_buffer = StreamingBuffer('${taskId || ''}', ${shouldStream ? '__greed_emit_stdout__' : 'None'}, ${shouldStream ? 'True' : 'False'})
_original_stdout = sys.stdout
sys.stdout = _output_buffer

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
finally:
    sys.stdout.flush()
    sys.stdout = _original_stdout
    _captured_output = _output_buffer.getvalue()
`;

        // Use runPythonAsync without Promise.race to avoid "interrupted by user" errors
        // Promise.race can interrupt Pyodide mid-execution causing fatal errors

        // Set up a non-interrupting timeout warning (only if timeout > 0)
        let timeoutWarning;
        if (timeout && timeout > 0) {
          timeoutWarning = setTimeout(() => {
            this.emit('execution:timeout', { timeout, stage: 'capture_output' });
          }, timeout);
        }

        try {
          await this.pyodide.runPythonAsync(outputCode);
        } finally {
          if (timeoutWarning) {
            clearTimeout(timeoutWarning);
          }
        }

        // Get captured output safely with error handling
        let capturedOutput = '';
        try {
          capturedOutput = this.pyodide.globals.get('_captured_output') || '';
        } catch (getError) {
          this.emit('output:error', { error: getError.message });
          capturedOutput = 'Output capture failed';
        }
        result = { output: capturedOutput };

        // Clean up globals to prevent memory leaks
        try {
          this.pyodide.globals.delete('_captured_output');
          this.pyodide.globals.delete('_output_buffer');
          this.pyodide.globals.delete('_original_stdout');
          if (shouldStream) {
            this.pyodide.globals.delete('__greed_emit_stdout__');
          }
        } catch (cleanupError) {
          // Ignore cleanup errors but emit them
          this.emit('cleanup:warning', { error: cleanupError.message });
        }
      } else {
        // Use consistent async execution without Promise.race to avoid interruption errors
        // Set up a non-interrupting timeout warning (only if timeout > 0)
        let timeoutWarning;
        if (timeout && timeout > 0) {
          timeoutWarning = setTimeout(() => {
            this.emit('execution:timeout', { timeout, stage: 'no_capture' });
          }, timeout);
        }

        try {
          result = await this.pyodide.runPythonAsync(code);
        } finally {
          if (timeoutWarning) {
            clearTimeout(timeoutWarning);
          }
        }
      }

      return result;
    } catch (error) {
      this.emit('execution:error', { error, code: code.substring(0, 100) });
      throw error;
    }
  }

  /**
   * Get Python global variable
   */
  async getGlobal(name) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }

    if (this.mode === 'worker') {
      return await this.workerManager.getGlobal(name);
    }

    try {
      return this.pyodide.globals.get(name);
    } catch (error) {
      this.emit('global:get:error', { name, error: error.message });
      return undefined;
    }
  }

  /**
   * Set Python global variable
   */
  async setGlobal(name, value) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }

    if (this.mode === 'worker') {
      return await this.workerManager.setGlobal(name, value);
    }

    try {
      this.pyodide.globals.set(name, value);
    } catch (error) {
      this.emit('global:set:error', { name, error: error.message });
      throw error; // Re-throw since this is a direct API call
    }
  }

  /**
   * Check if package is installed
   */
  hasPackage(packageName) {
    return this.installedPackages.has(packageName);
  }

  /**
   * Get runtime status
   */
  getStatus() {
    return {
      isReady: this.isReady,
      installedPackages: Array.from(this.installedPackages),
      pyodideVersion: this.pyodide?.version || null,
      config: this.config
    };
  }

  /**
   * Clear Python globals that might cause state persistence
   * Uses direct Pyodide API to avoid recursion through runPython
   */
  async clearExecutionState() {
    if (!this.isReady) {
      return;
    }

    try {
      // Use direct Pyodide API to avoid recursion through runPython
      await this.pyodide.runPythonAsync(`
import gc
import sys
import builtins

# List of globals to preserve (built-ins and essential modules)
preserved_globals = {
    'torch', 'np', 'numpy', 'sys', 'builtins', '__builtins__',
    'gc', '__name__', '__doc__', '__package__', '__loader__',
    '__spec__', '__annotations__', '__cached__', '__file__'
}

# Get current globals
current_globals = list(globals().keys())

# Remove user-defined variables
for var_name in current_globals:
    if (var_name not in preserved_globals and
        not var_name.startswith('_') and
        not callable(globals().get(var_name, None)) or
        var_name.startswith('_greed_')):
        try:
            del globals()[var_name]
        except:
            pass

# Force garbage collection
gc.collect()
`);

    } catch (error) {
      this.emit('state:clear:error', { error: error.message });
    }
  }

  /**
   * Cleanup runtime resources
   */
  async cleanup() {
    try {
      if (this.mode === 'worker' && this.workerManager) {
        // Terminate the worker
        await this.workerManager.terminate();
        this.workerManager = null;
      } else if (this.pyodide) {
        // Try to clean up gracefully first using async version
        try {
          await this.pyodide.runPythonAsync(`
import gc
import sys

# Clear user globals
user_globals = [k for k in list(globals().keys())
               if not k.startswith('__') and k not in sys.modules]
for k in user_globals:
    try:
        del globals()[k]
    except:
        pass

gc.collect()
`);
        } catch (e) {
          // Ignore cleanup errors during shutdown
        }

        this.pyodide.globals.clear();
        this.pyodide = null;
      }

      this.isReady = false;
      this.installedPackages.clear();
      this.initPromise = null;

      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      // Error already emitted for handling by parent components
    }
  }

  // Private helper methods
  _createTimeoutPromise(timeout, message) {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(message)), timeout);
    });
  }

  /**
   * Check if code requires matplotlib
   */
  _needsMatplotlib(code) {
    const matplotlibPatterns = [
      /import\s+matplotlib/,
      /from\s+matplotlib/,
      /import\s+matplotlib\.pyplot/,
      /from\s+matplotlib\.pyplot/,
      /plt\./
    ];
    return matplotlibPatterns.some(pattern => pattern.test(code));
  }

  /**
   * Ensure matplotlib is loaded and available
   */
  async _ensureMatplotlibLoaded() {
    try {
      if (!this.installedPackages.has('matplotlib')) {
        this.emit('package:loading', { package: 'matplotlib' });
        await this.pyodide.loadPackage('matplotlib');
        this.installedPackages.add('matplotlib');
        this.emit('package:loaded', { package: 'matplotlib' });
      }
    } catch (error) {
      this.emit('package:error', { package: 'matplotlib', error });
      // Don't throw - allow execution to continue
    }
  }

  /**
   * Check if code requires pandas
   */
  _needsPandas(code) {
    const pandasPatterns = [
      /import\s+pandas/,
      /from\s+pandas/,
      /import\s+pandas\s+as\s+pd/,
      /pd\./
    ];
    return pandasPatterns.some(pattern => pattern.test(code));
  }

  /**
   * Ensure pandas is loaded and available
   */
  async _ensurePandasLoaded() {
    try {
      if (!this.installedPackages.has('pandas')) {
        this.emit('package:loading', { package: 'pandas' });
        await this.pyodide.loadPackage('pandas');
        this.installedPackages.add('pandas');
        this.emit('package:loaded', { package: 'pandas' });
      }
    } catch (error) {
      this.emit('package:error', { package: 'pandas', error });
      // Don't throw - allow execution to continue
    }
  }

  /**
   * Check if code requires scipy
   */
  _needsScipy(code) {
    const scipyPatterns = [
      /import\s+scipy/,
      /from\s+scipy/,
      /scipy\./
    ];
    return scipyPatterns.some(pattern => pattern.test(code));
  }

  /**
   * Ensure scipy is loaded and available
   */
  async _ensureScipyLoaded() {
    try {
      if (!this.installedPackages.has('scipy')) {
        this.emit('package:loading', { package: 'scipy' });
        await this.pyodide.loadPackage('scipy');
        this.installedPackages.add('scipy');
        this.emit('package:loaded', { package: 'scipy' });
      }
    } catch (error) {
      this.emit('package:error', { package: 'scipy', error });
    }
  }

  /**
   * Check if code requires plotly
   */
  _needsPlotly(code) {
    const plotlyPatterns = [
      /import\s+plotly/,
      /from\s+plotly/,
      /plotly\./,
      /import\s+plotly\.graph_objects/,
      /import\s+plotly\.express/
    ];
    return plotlyPatterns.some(pattern => pattern.test(code));
  }

  /**
   * Ensure plotly is loaded and available
   */
  async _ensurePlotlyLoaded() {
    try {
      if (!this.installedPackages.has('plotly')) {
        this.emit('package:loading', { package: 'plotly' });
        await this.pyodide.loadPackage('plotly');
        this.installedPackages.add('plotly');
        this.emit('package:loaded', { package: 'plotly' });
      }
    } catch (error) {
      this.emit('package:error', { package: 'plotly', error });
    }
  }

  /**
   * Check if code requires scikit-learn
   */
  _needsSklearn(code) {
    const sklearnPatterns = [
      /import\s+sklearn/,
      /from\s+sklearn/,
      /sklearn\./,
      /from\s+sklearn\.\w+/
    ];
    return sklearnPatterns.some(pattern => pattern.test(code));
  }

  /**
   * Ensure scikit-learn is loaded and available
   */
  async _ensureSklearnLoaded() {
    try {
      if (!this.installedPackages.has('scikit-learn')) {
        this.emit('package:loading', { package: 'scikit-learn' });
        await this.pyodide.loadPackage('scikit-learn');
        this.installedPackages.add('scikit-learn');
        this.emit('package:loaded', { package: 'scikit-learn' });
      }
    } catch (error) {
      this.emit('package:error', { package: 'scikit-learn', error });
    }
  }

  _containsDangerousPatterns(code) {
    const dangerousPatterns = [
      /\beval\s*\(/,
      /\bexec\s*\(/,
      /\b__import__\s*\(/,
      /\bsubprocess\./,
      /\bos\.system\s*\(/,
      /\bopen\s*\(/,
      /\bfile\s*\(/
    ];

    return dangerousPatterns.some(pattern => pattern.test(code));
  }
}

// Custom error for security violations
class SecurityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SecurityError';
  }
}

export default RuntimeManager;
export { SecurityError };