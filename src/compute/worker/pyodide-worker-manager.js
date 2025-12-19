/**
 * PyodideWorkerManager - Manages Pyodide Web Worker communication
 *
 * Provides a clean API for the main thread to communicate with the Pyodide worker
 * Handles message passing, promise resolution, and error handling
 */
import EventEmitter from '../../core/event-emitter.js';
import logger from '../../utils/logger.js';

class PyodideWorkerManager extends EventEmitter {
  constructor(config = {}) {
    super();

    this.config = {
      workerURL: config.workerURL || null, // Will be set during initialization
      pyodideIndexURL: config.pyodideIndexURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      pyodideURL: config.pyodideURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js',
      preloadPackages: config.preloadPackages || ['numpy'],
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      ...config
    };

    // Worker state
    this.worker = null;
    this.isInitialized = false;
    this.isReady = false;
    this.installedPackages = new Set();

    // Message handling
    this.messageId = 0;
    this.pendingMessages = new Map();
    this.executionTasks = new Map();

    // Performance tracking
    this.stats = {
      messagesProcessed: 0,
      executionsCompleted: 0,
      averageExecutionTime: 0,
      totalExecutionTime: 0
    };
  }

  /**
   * Initialize the worker and Pyodide
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      this.emit('init:start', { stage: 'worker' });

      // Set up ready listener BEFORE creating worker
      const readyPromise = new Promise((resolve, reject) => {
        this._workerReadyResolve = resolve;
        this._workerReadyReject = reject;

        // Set timeout
        this._workerReadyTimeout = setTimeout(() => {
          logger.error('Worker ready timeout - no response after', this.config.timeout, 'ms');
          reject(new Error('Worker ready timeout'));
        }, this.config.timeout);
      });

      // Create the worker (will immediately send 'worker:ready')
      await this._createWorker();

      // Wait for worker ready signal (may have already arrived)
      await readyPromise;

      // Clear timeout
      if (this._workerReadyTimeout) {
        clearTimeout(this._workerReadyTimeout);
      }

      // Initialize Pyodide in the worker
      await this._initializePyodide();

      this.isInitialized = true;
      this.isReady = true;

      this.emit('init:complete', {
        installedPackages: Array.from(this.installedPackages)
      });

      return true;
    } catch (error) {
      this.emit('init:error', { error });
      throw error;
    }
  }

  /**
   * Create the Web Worker
   */
  async _createWorker() {
    return new Promise((resolve, reject) => {
      try {
        // Create worker from the pyodide-worker.js file
        const workerURL = this.config.workerURL || this._getWorkerURL();
        logger.info('Creating worker from URL:', workerURL);

        // Note: Not using { type: 'module' } because webpack bundles as classic script
        this.worker = new Worker(workerURL);

        // Setup message handler
        this.worker.onmessage = this._handleMessage.bind(this);

        // Setup error handler
        this.worker.onerror = (errorEvent) => {
          const error = new Error(
            `Worker error: ${errorEvent.message || 'Unknown error'} ` +
            `at ${errorEvent.filename}:${errorEvent.lineno}:${errorEvent.colno}`
          );
          logger.error('Worker error event:', {
            message: errorEvent.message,
            filename: errorEvent.filename,
            lineno: errorEvent.lineno,
            colno: errorEvent.colno
          });
          this.emit('worker:error', { error });

          // Also reject the ready promise if it's still pending
          if (this._workerReadyReject) {
            this._workerReadyReject(error);
            this._workerReadyReject = null;
            this._workerReadyResolve = null;
          }

          reject(error);
        };

        logger.info('Worker created successfully');
        resolve();
      } catch (error) {
        logger.error('Failed to create worker:', error);
        reject(error);
      }
    });
  }

  /**
   * Get the worker URL (tries to use bundled worker or creates inline)
   */
  _getWorkerURL() {
    // Try different paths to find the worker
    try {
      // First try: Same directory as main bundle (for npm package usage)
      if (typeof document !== 'undefined') {
        // Get the base URL from greed.js script location
        let baseURL;

        // Find the greed.js script tag
        const scripts = document.getElementsByTagName('script');
        let greedScript = null;
        for (let i = scripts.length - 1; i >= 0; i--) {
          if (scripts[i].src && scripts[i].src.includes('greed')) {
            greedScript = scripts[i];
            break;
          }
        }

        if (greedScript && greedScript.src) {
          baseURL = greedScript.src.replace(/[^/]*$/, '');
          logger.info('Worker path from greed.js script tag:', baseURL + 'pyodide-worker.js');
        } else if (document.currentScript && document.currentScript.src) {
          baseURL = document.currentScript.src.replace(/[^/]*$/, '');
          logger.info('Worker path from currentScript:', baseURL + 'pyodide-worker.js');
        } else {
          // Fallback: try common patterns
          baseURL = window.location.origin + '/dist/';
          logger.info('Worker path from fallback:', baseURL + 'pyodide-worker.js');
        }
        const workerPath = baseURL + 'pyodide-worker.js';
        return workerPath;
      }

      // Second try: Use relative URL (works when both files are served from same directory)
      logger.info('Worker path (no document):', 'dist/pyodide-worker.js');
      return 'dist/pyodide-worker.js';
    } catch (error) {
      logger.warn('Could not determine worker path, using fallback:', error);
      // Fallback: try relative path
      return 'dist/pyodide-worker.js';
    }
  }

  /**
   * Create an inline worker as fallback
   */
  _createInlineWorker() {
    // This is a fallback - in production, use the file-based worker
    const blob = new Blob([
      `importScripts('${this.config.pyodideURL}');
       postMessage({ type: 'worker:ready' });`
    ], { type: 'application/javascript' });

    return URL.createObjectURL(blob);
  }

  /**
   * Initialize Pyodide in the worker
   */
  async _initializePyodide() {
    return this._sendMessage('init', {
      config: {
        indexURL: this.config.pyodideIndexURL,
        pyodideURL: this.config.pyodideURL,
        preloadPackages: this.config.preloadPackages
      }
    });
  }

  /**
   * Handle messages from worker
   */
  _handleMessage(event) {
    const message = event.data;
    this.stats.messagesProcessed++;

    // Handle different message types
    switch (message.type) {
      case 'worker:ready':
        // Resolve the ready promise
        logger.info('Worker ready message received');
        if (this._workerReadyResolve) {
          this._workerReadyResolve();
          this._workerReadyResolve = null;
          this._workerReadyReject = null;
        }
        break;

      case 'init:progress':
      case 'init:complete':
      case 'init:error':
      case 'packages:loading':
      case 'packages:loaded':
      case 'packages:error':
      case 'execution:warning':
      case 'execution:interrupted':
      case 'interrupt:error':
      case 'reset:complete':
      case 'reset:error':
        // Forward events to listeners
        this.emit(message.type, message);
        break;

      case 'execution:stdout':
        // Forward streaming output to RuntimeManager
        this.emit('execution:stdout', {
          type: 'execution:stdout',
          taskId: message.taskId,
          output: message.output,
          timestamp: message.timestamp
        });
        break;

      case 'execution:complete':
        this._handleExecutionComplete(message);
        break;

      case 'execution:error':
        this._handleExecutionError(message);
        break;

      case 'execution:cleanup':
        // Forward cleanup events
        this.emit('execution:cleanup', message);
        break;

      case 'init:ack':
      case 'loadPackages:ack':
      case 'execute:ack':
      case 'getGlobal:result':
      case 'setGlobal:result':
      case 'deleteGlobal:result':
      case 'interrupt:ack':
      case 'reset:ack':
      case 'pong':
        // Resolve pending messages
        this._resolvePendingMessage(message.id, message);
        break;

      case 'error':
        this._rejectPendingMessage(message.id, new Error(message.error.message));
        break;

      default:
        logger.warn('Unknown message type from worker:', message.type);
    }
  }

  /**
   * Handle execution completion
   */
  _handleExecutionComplete(message) {
    const task = this.executionTasks.get(message.taskId);
    if (task) {
      const executionTime = performance.now() - task.startTime;

      this.stats.executionsCompleted++;
      this.stats.totalExecutionTime += executionTime;
      this.stats.averageExecutionTime =
        this.stats.totalExecutionTime / this.stats.executionsCompleted;

      task.resolve(message.result);
      this.executionTasks.delete(message.taskId);

      this.emit('execution:complete', {
        taskId: message.taskId,
        executionTime,
        result: message.result
      });
    }
  }

  /**
   * Handle execution error
   */
  _handleExecutionError(message) {
    const task = this.executionTasks.get(message.taskId);
    if (task) {
      const error = new Error(message.error.message);
      error.stack = message.error.stack;
      error.pythonType = message.error.type;

      task.reject(error);
      this.executionTasks.delete(message.taskId);

      this.emit('execution:error', {
        taskId: message.taskId,
        error
      });
    }
  }

  /**
   * Send a message to the worker and wait for response
   */
  async _sendMessage(type, data = {}) {
    return new Promise((resolve, reject) => {
      const id = this.messageId++;
      const timeout = setTimeout(() => {
        this.pendingMessages.delete(id);
        reject(new Error(`Message timeout: ${type}`));
      }, this.config.timeout);

      this.pendingMessages.set(id, { resolve, reject, timeout });

      this.worker.postMessage({
        type,
        id,
        ...data
      });
    });
  }

  /**
   * Resolve a pending message
   */
  _resolvePendingMessage(id, result) {
    const pending = this.pendingMessages.get(id);
    if (pending) {
      clearTimeout(pending.timeout);
      pending.resolve(result);
      this.pendingMessages.delete(id);
    }
  }

  /**
   * Reject a pending message
   */
  _rejectPendingMessage(id, error) {
    const pending = this.pendingMessages.get(id);
    if (pending) {
      clearTimeout(pending.timeout);
      pending.reject(error);
      this.pendingMessages.delete(id);
    }
  }

  /**
   * Load Python packages
   */
  async loadPackages(packages) {
    if (!this.isReady) {
      throw new Error('Worker not initialized');
    }

    const result = await this._sendMessage('loadPackages', { packages });

    if (result.packages) {
      result.packages.forEach(pkg => this.installedPackages.add(pkg));
    }

    return Array.from(this.installedPackages);
  }

  /**
   * Execute Python code
   */
  async executePython(code, options = {}) {
    if (!this.isReady) {
      throw new Error('Worker not initialized');
    }

    return new Promise((resolve, reject) => {
      // Use taskId from options if provided, otherwise generate one
      const taskId = options.taskId || `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const startTime = performance.now();

      this.executionTasks.set(taskId, {
        taskId,
        startTime,
        resolve,
        reject
      });

      // Send execution request to worker
      this.worker.postMessage({
        type: 'execute',
        id: this.messageId++,
        taskId,
        code,
        options
      });

      // Setup timeout
      if (options.timeout) {
        setTimeout(() => {
          if (this.executionTasks.has(taskId)) {
            this.executionTasks.delete(taskId);
            reject(new Error('Execution timeout'));
          }
        }, options.timeout);
      }
    });
  }

  /**
   * Get a Python global variable
   */
  async getGlobal(name) {
    if (!this.isReady) {
      throw new Error('Worker not initialized');
    }

    const result = await this._sendMessage('getGlobal', { name });
    return result.value;
  }

  /**
   * Set a Python global variable
   */
  async setGlobal(name, value) {
    if (!this.isReady) {
      throw new Error('Worker not initialized');
    }

    const result = await this._sendMessage('setGlobal', { name, value });
    return result.success;
  }

  /**
   * Delete a Python global variable
   */
  async deleteGlobal(name) {
    if (!this.isReady) {
      throw new Error('Worker not initialized');
    }

    const result = await this._sendMessage('deleteGlobal', { name });
    return result.success;
  }

  /**
   * Interrupt current execution
   */
  async interrupt() {
    if (!this.isReady) {
      return;
    }

    await this._sendMessage('interrupt');
  }

  /**
   * Reset the Python environment
   */
  async reset() {
    if (!this.isReady) {
      return;
    }

    await this._sendMessage('reset');
  }

  /**
   * Ping the worker to check if it's alive
   */
  async ping() {
    if (!this.isReady) {
      return false;
    }

    try {
      await this._sendMessage('ping');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get worker statistics
   */
  getStats() {
    return { ...this.stats };
  }

  /**
   * Terminate the worker
   */
  async terminate() {
    if (this.worker) {
      // Reject all pending messages
      for (const [id, pending] of this.pendingMessages) {
        clearTimeout(pending.timeout);
        pending.reject(new Error('Worker terminated'));
      }
      this.pendingMessages.clear();

      // Reject all execution tasks
      for (const [taskId, task] of this.executionTasks) {
        task.reject(new Error('Worker terminated'));
      }
      this.executionTasks.clear();

      // Terminate worker
      this.worker.terminate();
      this.worker = null;
      this.isInitialized = false;
      this.isReady = false;

      this.emit('worker:terminated');
    }
  }
}

export default PyodideWorkerManager;
