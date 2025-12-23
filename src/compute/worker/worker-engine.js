/**
 * Worker Engine - Multi-threaded Web Worker execution for CPU-intensive operations
 * Distributes computation across worker threads for improved performance
 */
import EventEmitter from '../../core/event-emitter.js';
import logger from '../../utils/logger.js';

class WorkerEngine extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      maxWorkers: config.maxWorkers || navigator.hardwareConcurrency || 4,
      workerTimeout: config.workerTimeout || 30000,
      enableLoadBalancing: config.enableLoadBalancing !== false,
      queueMaxSize: config.queueMaxSize || 100,
      ...config
    };

    // Worker pool state
    this.isInitialized = false;
    this.workers = [];
    this.availableWorkers = [];
    this.busyWorkers = new Map();
    this.taskQueue = [];
    
    // Performance tracking
    this.stats = {
      workers: 0,
      operations: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0,
      queuedOperations: 0,
      failedOperations: 0,
      workerRestarts: 0
    };
    
    // Task management
    this.nextTaskId = 1;
    this.pendingTasks = new Map();
  }

  /**
   * Initialize worker pool
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      this.emit('init:start', { maxWorkers: this.config.maxWorkers });
      
      // Create worker pool
      await this._createWorkerPool();
      
      // Setup task processing
      this._startTaskProcessor();
      
      this.isInitialized = true;
      this.emit('init:complete', { 
        workers: this.workers.length,
        ready: this.availableWorkers.length
      });
      
      return true;
    } catch (error) {
      this.emit('init:error', { error });
      throw error;
    }
  }

  /**
   * Execute operation on worker thread
   */
  async execute(operation, tensors, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Worker engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const taskId = this.nextTaskId++;
      const startTime = performance.now();
      
      const task = {
        id: taskId,
        operation,
        tensors,
        options,
        startTime,
        resolve,
        reject,
        timeout: setTimeout(() => {
          this._handleTaskTimeout(taskId);
        }, options.timeout || this.config.workerTimeout)
      };
      
      this.pendingTasks.set(taskId, task);
      this._queueTask(task);
      
      this.emit('task:queued', { taskId, operation, queueSize: this.taskQueue.length });
    });
  }

  /**
   * Execute batch of operations with load balancing
   */
  async executeBatch(operations, options = {}) {
    const { 
      parallel = true, 
      maxConcurrency = this.config.maxWorkers,
      loadBalance = this.config.enableLoadBalancing
    } = options;

    if (!parallel) {
      // Sequential execution
      const results = [];
      for (const op of operations) {
        const result = await this.execute(op.operation, op.tensors, op.options);
        results.push(result);
      }
      return results;
    }

    // Parallel execution with load balancing
    if (loadBalance) {
      const distributedOps = this._distributeOperations(operations);
      const promises = distributedOps.map(op => 
        this.execute(op.operation, op.tensors, op.options)
      );
      return Promise.all(promises);
    }

    // Simple parallel execution
    const semaphore = new Semaphore(maxConcurrency);
    const promises = operations.map(async (op) => {
      await semaphore.acquire();
      try {
        return await this.execute(op.operation, op.tensors, op.options);
      } finally {
        semaphore.release();
      }
    });

    return Promise.all(promises);
  }

  /**
   * Get worker engine statistics
   */
  getStats() {
    return {
      ...this.stats,
      isInitialized: this.isInitialized,
      availableWorkers: this.availableWorkers.length,
      busyWorkers: this.busyWorkers.size,
      queuedTasks: this.taskQueue.length,
      pendingTasks: this.pendingTasks.size,
      type: 'worker'
    };
  }

  /**
   * Cleanup worker pool and resources
   */
  async cleanup() {
    try {
      this.emit('cleanup:start');
      
      // Cancel pending tasks
      for (const [_taskId, task] of this.pendingTasks.entries()) {
        clearTimeout(task.timeout);
        task.reject(new Error('Worker engine shutting down'));
      }
      this.pendingTasks.clear();
      this.taskQueue = [];
      
      // Terminate all workers
      for (const worker of this.workers) {
        worker.terminate();
      }
      
      this.workers = [];
      this.availableWorkers = [];
      this.busyWorkers.clear();
      this.isInitialized = false;
      
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  async _createWorkerPool() {
    const workerScript = this._generateWorkerScript();
    const workerBlob = new Blob([workerScript], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(workerBlob);
    
    try {
      for (let i = 0; i < this.config.maxWorkers; i++) {
        const worker = new Worker(workerUrl);
        const workerId = i;
        
        worker.addEventListener('message', (event) => {
          this._handleWorkerMessage(workerId, event);
        });
        
        worker.addEventListener('error', (error) => {
          this._handleWorkerError(workerId, error);
        });
        
        worker.addEventListener('messageerror', (error) => {
          this._handleWorkerError(workerId, error);
        });
        
        this.workers.push(worker);
        this.availableWorkers.push(workerId);
        this.stats.workers++;
        
        this.emit('worker:created', { workerId });
      }
    } finally {
      URL.revokeObjectURL(workerUrl);
    }
  }

  _generateWorkerScript() {
    return `
// Worker script for tensor operations
class WorkerTensorOps {
  constructor() {
    this.operations = {
      add: (a, b) => a.map((val, i) => val + b[i]),
      subtract: (a, b) => a.map((val, i) => val - b[i]),
      multiply: (a, b) => a.map((val, i) => val * b[i]),
      divide: (a, b) => a.map((val, i) => val / (b[i] || 1e-12)),
      
      matmul: (a, b, shapeA, shapeB) => {
        if (shapeA.length !== 2 || shapeB.length !== 2) {
          throw new Error('Matrix multiplication requires 2D arrays');
        }
        
        const [rowsA, colsA] = shapeA;
        const [rowsB, colsB] = shapeB;
        
        if (colsA !== rowsB) {
          throw new Error('Matrix dimensions incompatible for multiplication');
        }
        
        const result = new Array(rowsA * colsB).fill(0);
        
        for (let i = 0; i < rowsA; i++) {
          for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
              sum += a[i * colsA + k] * b[k * colsB + j];
            }
            result[i * colsB + j] = sum;
          }
        }
        
        return result;
      },
      
      transpose: (a, shape) => {
        if (shape.length !== 2) {
          throw new Error('Transpose currently supports only 2D arrays');
        }
        
        const [rows, cols] = shape;
        const result = new Array(rows * cols);
        
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            result[j * rows + i] = a[i * cols + j];
          }
        }
        
        return result;
      },
      
      sum: (a, axis) => {
        if (axis === null || axis === undefined) {
          return a.reduce((sum, val) => sum + val, 0);
        }
        // Simplified sum along axis (would need full tensor implementation)
        return a.reduce((sum, val) => sum + val, 0);
      },
      
      mean: (a, axis) => {
        const sum = this.operations.sum(a, axis);
        return Array.isArray(sum) ? sum.map(val => val / a.length) : sum / a.length;
      },
      
      relu: (a) => a.map(val => Math.max(0, val)),
      
      sigmoid: (a) => a.map(val => 1 / (1 + Math.exp(-Math.max(-250, Math.min(250, val))))),
      
      tanh: (a) => a.map(val => Math.tanh(val)),
      
      softmax: (a) => {
        const maxVal = Math.max(...a);
        const exp = a.map(val => Math.exp(val - maxVal));
        const sum = exp.reduce((s, val) => s + val, 0);
        return exp.map(val => val / sum);
      },
      
      exp: (a) => a.map(val => Math.exp(Math.max(-250, Math.min(250, val)))),
      
      log: (a) => a.map(val => Math.log(Math.max(1e-12, val))),
      
      sqrt: (a) => a.map(val => Math.sqrt(Math.max(0, val))),
      
      abs: (a) => a.map(val => Math.abs(val)),
      
      power: (a, b) => {
        if (Array.isArray(b)) {
          return a.map((val, i) => Math.pow(val, b[i]));
        } else {
          return a.map(val => Math.pow(val, b));
        }
      }
    };
  }
  
  execute(operation, tensors, options = {}) {
    try {
      const op = this.operations[operation];
      if (!op) {
        throw new Error(\`Unsupported operation: \${operation}\`);
      }
      
      // Convert tensors to arrays if needed
      const tensorArrays = tensors.map(tensor => {
        if (tensor.data && tensor.shape) {
          return { data: Array.from(tensor.data), shape: tensor.shape };
        } else if (Array.isArray(tensor)) {
          return { data: tensor, shape: [tensor.length] };
        } else {
          return { data: Array.from(tensor), shape: [tensor.length] };
        }
      });
      
      // Execute operation
      let result;
      switch (operation) {
        case 'matmul':
          result = op(tensorArrays[0].data, tensorArrays[1].data, 
                      tensorArrays[0].shape, tensorArrays[1].shape);
          break;
        case 'transpose':
          result = op(tensorArrays[0].data, tensorArrays[0].shape);
          break;
        case 'power':
          if (tensorArrays.length > 1) {
            result = op(tensorArrays[0].data, tensorArrays[1].data);
          } else {
            result = op(tensorArrays[0].data, options.exponent || 2);
          }
          break;
        default:
          if (tensorArrays.length === 1) {
            result = op(tensorArrays[0].data, options.axis);
          } else {
            result = op(tensorArrays[0].data, tensorArrays[1].data);
          }
      }
      
      return result;
    } catch (error) {
      throw new Error(\`Worker operation failed: \${error.message}\`);
    }
  }
}

const tensorOps = new WorkerTensorOps();

self.addEventListener('message', function(event) {
  const { taskId, operation, tensors, options } = event.data;
  
  try {
    const startTime = performance.now();
    const result = tensorOps.execute(operation, tensors, options);
    const executionTime = performance.now() - startTime;
    
    self.postMessage({
      taskId,
      success: true,
      result,
      executionTime
    });
  } catch (error) {
    self.postMessage({
      taskId,
      success: false,
      error: error.message
    });
  }
});
`;
  }

  _queueTask(task) {
    if (this.taskQueue.length >= this.config.queueMaxSize) {
      task.reject(new Error('Worker queue full'));
      this.stats.failedOperations++;
      return;
    }
    
    this.taskQueue.push(task);
    this.stats.queuedOperations++;
    this._processTaskQueue();
  }

  _startTaskProcessor() {
    // Process tasks periodically
    setInterval(() => {
      this._processTaskQueue();
    }, 10); // Check every 10ms
  }

  _processTaskQueue() {
    while (this.taskQueue.length > 0 && this.availableWorkers.length > 0) {
      const task = this.taskQueue.shift();
      const workerId = this.availableWorkers.shift();
      
      this.busyWorkers.set(workerId, task);
      
      // Send task to worker
      this.workers[workerId].postMessage({
        taskId: task.id,
        operation: task.operation,
        tensors: task.tensors,
        options: task.options
      });
      
      this.emit('task:started', { 
        taskId: task.id, 
        workerId, 
        operation: task.operation 
      });
    }
  }

  _handleWorkerMessage(workerId, event) {
    const { taskId, success, result, error, executionTime } = event.data;
    
    const task = this.pendingTasks.get(taskId);
    if (!task) {
      logger.warn('Received message for unknown task:', {
        taskId,
        workerId,
        availableTasks: Array.from(this.pendingTasks.keys())
      });
      return;
    }
    
    // Clear task timeout
    clearTimeout(task.timeout);
    this.pendingTasks.delete(taskId);
    
    // Free worker
    this.busyWorkers.delete(workerId);
    this.availableWorkers.push(workerId);
    
    // Update statistics
    const totalTime = performance.now() - task.startTime;
    this.stats.operations++;
    this.stats.totalExecutionTime += totalTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.operations;
    
    if (success) {
      task.resolve(result);
      this.emit('task:completed', { 
        taskId, 
        workerId, 
        executionTime: totalTime,
        workerTime: executionTime
      });
    } else {
      this.stats.failedOperations++;
      task.reject(new Error(error));
      this.emit('task:failed', { 
        taskId, 
        workerId, 
        error,
        executionTime: totalTime
      });
    }
  }

  _handleWorkerError(workerId, error) {
    this.emit('worker:error', { workerId, error });
    
    // Find and reject tasks for this worker
    const task = this.busyWorkers.get(workerId);
    if (task) {
      clearTimeout(task.timeout);
      this.pendingTasks.delete(task.id);
      this.busyWorkers.delete(workerId);
      task.reject(new Error(`Worker error: ${error.message || error}`));
      this.stats.failedOperations++;
    }
    
    // Restart worker
    this._restartWorker(workerId);
  }

  _handleTaskTimeout(taskId) {
    const task = this.pendingTasks.get(taskId);
    if (!task) return;
    
    this.pendingTasks.delete(taskId);
    
    // Find worker handling this task
    for (const [workerId, workerTask] of this.busyWorkers.entries()) {
      if (workerTask.id === taskId) {
        this.busyWorkers.delete(workerId);
        this._restartWorker(workerId); // Restart worker due to timeout
        break;
      }
    }
    
    this.stats.failedOperations++;
    task.reject(new Error(`Task timeout after ${this.config.workerTimeout}ms`));
    
    this.emit('task:timeout', { taskId, timeout: this.config.workerTimeout });
  }

  async _restartWorker(workerId) {
    try {
      // Terminate old worker
      this.workers[workerId]?.terminate();
      
      // Create new worker
      const workerScript = this._generateWorkerScript();
      const workerBlob = new Blob([workerScript], { type: 'application/javascript' });
      const workerUrl = URL.createObjectURL(workerBlob);
      
      const newWorker = new Worker(workerUrl);
      URL.revokeObjectURL(workerUrl);
      
      newWorker.addEventListener('message', (event) => {
        this._handleWorkerMessage(workerId, event);
      });
      
      newWorker.addEventListener('error', (error) => {
        this._handleWorkerError(workerId, error);
      });
      
      this.workers[workerId] = newWorker;
      this.availableWorkers.push(workerId);
      this.stats.workerRestarts++;
      
      this.emit('worker:restarted', { workerId });
    } catch (error) {
      this.emit('worker:restart-failed', { workerId, error });
    }
  }

  _distributeOperations(operations) {
    // Simple round-robin distribution
    // Real implementation could consider operation complexity, worker load, etc.
    return operations.map((op, index) => ({
      ...op,
      preferredWorker: index % this.config.maxWorkers
    }));
  }
}

// Simple semaphore for concurrency control
class Semaphore {
  constructor(max) {
    this.max = max;
    this.current = 0;
    this.queue = [];
  }

  async acquire() {
    return new Promise((resolve) => {
      if (this.current < this.max) {
        this.current++;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  release() {
    this.current--;
    if (this.queue.length > 0) {
      const resolve = this.queue.shift();
      this.current++;
      resolve();
    }
  }
}

export default WorkerEngine;