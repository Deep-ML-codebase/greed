/**
 * RuntimeManager - Handles Pyodide initialization and Python package management
 * Extracted from monolithic Greed class for better separation of concerns
 */
import EventEmitter from './event-emitter.js';

class RuntimeManager extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      pyodideIndexURL: config.pyodideIndexURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      preloadPackages: config.preloadPackages || ['numpy'],
      timeout: config.initTimeout || 30000,
      ...config
    };

    this.pyodide = null;
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
    try {
      this.emit('init:start', { stage: 'pyodide' });
      
      // Validate Pyodide availability
      if (typeof loadPyodide === 'undefined') {
        throw new Error('Pyodide not loaded. Please include pyodide.js in your HTML.');
      }

      // Initialize with timeout
      const pyodidePromise = loadPyodide({
        indexURL: this.config.pyodideIndexURL
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

    return this._loadPackages(packages);
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
      console.warn(`⚠️ Failed to install ${packagesToLoad.join(', ')}: ${error.message}`);
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

    const {
      captureOutput = false,
      timeout = 1000,
      globals = {},
      validateInput = true
    } = options;

    if (validateInput && this._containsDangerousPatterns(code)) {
      throw new SecurityError('Potentially dangerous code patterns detected');
    }

    try {
      // DISABLED: Auto-package loading due to 'await' syntax errors
      // These cause browser freeze issues during package installation
      // Users can manually import packages if needed

      // Auto-load matplotlib if code imports it
      // if (this._needsMatplotlib(code)) {
      //   await this._ensureMatplotlibLoaded();
      // }

      // Auto-load pandas if code imports it
      // if (this._needsPandas(code)) {
      //   await this._ensurePandasLoaded();
      // }

      // Set globals if provided - with error handling to prevent fatal errors
      for (const [key, value] of Object.entries(globals)) {
        try {
          this.pyodide.globals.set(key, value);
        } catch (globalError) {
          console.warn(`Failed to set global '${key}':`, globalError.message);
          // Continue with other globals instead of failing completely
        }
      }

      let result;
      if (captureOutput) {
        // Capture stdout for print statements - use async version to prevent fatal errors
        const outputCode = `
import sys
from io import StringIO
_output_buffer = StringIO()
_original_stdout = sys.stdout
sys.stdout = _output_buffer

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
finally:
    sys.stdout = _original_stdout
    _captured_output = _output_buffer.getvalue()
`;

        // Use runPythonAsync without Promise.race to avoid "interrupted by user" errors
        // Promise.race can interrupt Pyodide mid-execution causing fatal errors

        // Set up a non-interrupting timeout warning
        const timeoutWarning = setTimeout(() => {
          console.warn(`⚠️ Python execution taking longer than ${timeout}ms, but continuing to avoid Pyodide corruption`);
        }, timeout);

        try {
          await this.pyodide.runPythonAsync(outputCode);
        } finally {
          clearTimeout(timeoutWarning);
        }

        // Get captured output safely with error handling
        let capturedOutput = '';
        try {
          capturedOutput = this.pyodide.globals.get('_captured_output') || '';
        } catch (getError) {
          console.warn('Failed to get captured output:', getError.message);
          capturedOutput = 'Output capture failed';
        }
        result = { output: capturedOutput };

        // Clean up globals to prevent memory leaks
        try {
          this.pyodide.globals.delete('_captured_output');
          this.pyodide.globals.delete('_output_buffer');
          this.pyodide.globals.delete('_original_stdout');
        } catch (cleanupError) {
          // Ignore cleanup errors but log them
          console.warn('Cleanup warning:', cleanupError.message);
        }
      } else {
        // Use consistent async execution without Promise.race to avoid interruption errors
        // Set up a non-interrupting timeout warning
        const timeoutWarning = setTimeout(() => {
          console.warn(`⚠️ Python execution taking longer than ${timeout}ms, but continuing to avoid Pyodide corruption`);
        }, timeout);

        try {
          result = await this.pyodide.runPythonAsync(code);
        } finally {
          clearTimeout(timeoutWarning);
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
  getGlobal(name) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }
    try {
      return this.pyodide.globals.get(name);
    } catch (error) {
      console.warn(`Failed to get global '${name}':`, error.message);
      return undefined;
    }
  }

  /**
   * Set Python global variable
   */
  setGlobal(name, value) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }
    try {
      this.pyodide.globals.set(name, value);
    } catch (error) {
      console.warn(`Failed to set global '${name}':`, error.message);
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
      console.warn('Failed to clear execution state:', error.message);
    }
  }

  /**
   * Cleanup runtime resources
   */
  async cleanup() {
    try {
      if (this.pyodide) {
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
      console.warn(`⚠️ Failed to load matplotlib: ${error.message}`);
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
      console.warn(`⚠️ Failed to load pandas: ${error.message}`);
      // Don't throw - allow execution to continue
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