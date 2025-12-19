/**
 * Greed v2.0 - Modular PyTorch-in-Browser Runtime
 * Lightweight orchestrator class that coordinates all modular components
 * Replaces the monolithic Greed.js with clean separation of concerns
 */
import EventEmitter from './event-emitter.js';
import RuntimeManager from './runtime-manager.js';
import ComputeStrategy from '../compute/compute-strategy.js';
import MemoryManager from '../utils/memory-manager.js';
import SecurityValidator from '../utils/security-validator.js';
import { WebGPUTensor } from '../compute/webgpu/webgpu-tensor.js';
import { createTensorBridge } from '../compute/webgpu/tensor-bridge.js';
import { createPyTorchPolyfill, validatePolyfill } from '../polyfills/pytorch-runtime.js';
import webgpuBridge from '../polyfills/webgpu-bridge.js';
import logger from '../utils/logger.js';

class Greed extends EventEmitter {
  constructor(options = {}) {
    super();
    
    // Validate and merge configuration
    this.config = this._validateConfig({
      // Core settings
      enableWebGPU: true,
      enableWorkers: true,
      maxWorkers: navigator.hardwareConcurrency || 4,
      
      // Security settings
      strictSecurity: true,
      allowEval: false,
      allowFileSystem: false,
      allowNetwork: false,
      
      // Performance settings
      maxMemoryMB: 1024,
      gcThreshold: 0.8,
      enableProfiling: true,
      
      // Runtime settings
      pyodideIndexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      preloadPackages: ['numpy'],
      initTimeout: 0, // No timeout for long-running processes
      
      ...options
    });

    // Component initialization state
    this.isInitialized = false;
    this.initializationPromise = null;
    this.componentsReady = {
      runtime: false,
      compute: false,
      memory: false,
      security: false
    };

    // Core components
    this.runtime = null;
    this.compute = null;
    this.memory = null;
    this.security = null;

    // API state tracking
    this.torchAPI = null;
    this.numpy = null;
    
    // Performance monitoring
    this.stats = {
      initTime: 0,
      operations: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0,
      memoryUsage: 0,
      startTime: performance.now()
    };

    // Error handling
    this.errorCount = 0;
    this.lastError = null;

    // Bind methods to preserve context when passed as callbacks
    this.run = this.run.bind(this);

    // Setup component event forwarding
    this._setupGlobalErrorHandling();
  }

  /**
   * Initialize all components and establish PyTorch API
   */
  async initialize() {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._performInitialization();
    return this.initializationPromise;
  }

  async _performInitialization() {
    if (this.isInitialized) {
      return true;
    }

    const startTime = performance.now();
    this.emit('init:start', { config: this.config });

    try {
      // Phase 1: Initialize core components
      await this._initializeComponents();

      // Phase 2: Setup PyTorch API integration
      await this._setupPyTorchAPI();

      // Phase 3: Validate system readiness
      await this._validateSystemReadiness();

      // Mark as initialized
      this.isInitialized = true;
      this.stats.initTime = performance.now() - startTime;

      this.emit('init:complete', {
        initTime: this.stats.initTime,
        components: Object.keys(this.componentsReady).filter(k => this.componentsReady[k]),
        memoryUsage: this.memory.getStats().memoryUsageMB
      });

      return true;
    } catch (error) {
      this.emit('init:error', { error, phase: 'initialization' });
      this.lastError = error;
      this.errorCount++;
      throw error;
    }
  }

  /**
   * Execute PyTorch operation with automatic optimization
   */
  async run(code, options = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const operationId = this._generateOperationId();
    const startTime = performance.now();

    this.emit('operation:start', { operationId, codeLength: code.length });

    try {
      // Security validation
      const securityResult = this.security.validatePythonCode(code, {
        allowWarnings: options.allowWarnings || false,
        bypassValidation: options.bypassSecurity || false
      });

      if (!securityResult.allowed) {
        throw new Error(`Security validation failed: ${securityResult.riskLevel} risk detected`);
      }

      // Execute in runtime with error handling
      const result = await this.runtime.runPython(code, {
        captureOutput: options.captureOutput !== false,
        timeout: options.timeout !== undefined ? options.timeout : 0, // 0 = no timeout for long-running processes
        globals: options.globals || {},
        validateInput: false, // Already validated above
        taskId: options.taskId, // Pass taskId for streaming correlation
        streamOutput: options.streamOutput !== false, // Pass streamOutput flag
        captureStdout: options.captureStdout !== false,
        captureStderr: options.captureStderr !== false
      });

      // Update statistics
      const executionTime = performance.now() - startTime;
      this._updateOperationStats(executionTime);

      this.emit('operation:complete', {
        operationId,
        executionTime,
        securityWarnings: securityResult.warnings?.length || 0,
        memoryUsage: this.memory.getStats().memoryUsageMB
      });

      return result;
    } catch (error) {
      const executionTime = performance.now() - startTime;
      this.emit('operation:error', { operationId, error, executionTime });
      this.lastError = error;
      this.errorCount++;

      // Clean up after error to prevent state corruption
      await this._cleanupExecutionState();

      throw error;
    }
  }

  /**
   * Alias for run() - maintains backward compatibility
   * Execute Python code with the same options as run()
   */
  async runPython(code, options = {}) {
    return this.run(code, options);
  }

  /**
   * Clear Python execution state (user variables)
   * Use this to reset the Python environment between sessions
   * Note: This preserves torch, numpy, and other library imports
   */
  async clearState() {
    if (!this.isInitialized) {
      throw new Error('Greed not initialized. Call initialize() first.');
    }
    await this._cleanupExecutionState();
  }

  /**
   * Execute tensor operation with compute strategy optimization
   */
  async tensor(operation, tensors, options = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      // Validate tensor inputs
      this.security.validateTensorData(tensors, options);

      // Validate operation
      const validatedOp = this.security.validateOperation(operation, options.params, options);

      // Execute with optimal compute strategy
      const result = await this.compute.execute(
        validatedOp.operation,
        tensors,
        validatedOp.options
      );

      return result;
    } catch (error) {
      this.emit('tensor:error', { operation, error });
      this.lastError = error;
      this.errorCount++;
      throw error;
    }
  }

  /**
   * Load additional Python packages
   */
  async loadPackages(packages) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const packageArray = Array.isArray(packages) ? packages : [packages];
    
    // Validate packages against security allowlist
    const allowedPackages = this.config.allowedPackages || this.security.config.allowedPackages;
    const blockedPackages = packageArray.filter(pkg => !allowedPackages.has(pkg));
    
    if (blockedPackages.length > 0) {
      throw new Error(`Packages not in allowlist: ${blockedPackages.join(', ')}`);
    }

    return this.runtime.loadPackages(packageArray);
  }

  /**
   * Get comprehensive system statistics
   */
  getStats() {
    const baseStats = {
      ...this.stats,
      isInitialized: this.isInitialized,
      errorCount: this.errorCount,
      uptime: performance.now() - this.stats.startTime,
      components: { ...this.componentsReady }
    };

    if (!this.isInitialized) {
      return baseStats;
    }

    return {
      ...baseStats,
      runtime: this.runtime.getStatus(),
      compute: this.compute.getStats(),
      memory: this.memory.getStats(),
      security: this.security.getStats()
    };
  }

  /**
   * Update configuration at runtime
   */
  updateConfig(newConfig) {
    const oldConfig = { ...this.config };
    this.config = { ...this.config, ...newConfig };

    // Update component configurations
    if (this.security && newConfig.security) {
      this.security.updateConfig(newConfig.security);
    }

    this.emit('config:updated', { oldConfig, newConfig: this.config });
  }

  /**
   * Force garbage collection across all components
   */
  async forceGC(options = {}) {
    if (!this.isInitialized) {
      return { cleaned: 0 };
    }

    this.emit('gc:start');

    try {
      const results = await Promise.allSettled([
        this.memory.forceGC(options),
        (this.compute && this.compute.availableStrategies && this.compute.availableStrategies.has('webgpu')) ?
          this.compute.webgpu.bufferManager?.gc(options) :
          Promise.resolve({ destroyed: 0 })
      ]);

      const totalCleaned = results.reduce((sum, result) => {
        if (result.status === 'fulfilled') {
          return sum + (result.value.cleaned || result.value.destroyed || 0);
        }
        return sum;
      }, 0);

      this.emit('gc:complete', { totalCleaned });
      return { cleaned: totalCleaned };
    } catch (error) {
      this.emit('gc:error', { error });
      throw error;
    }
  }

  /**
   * Clean up execution state to prevent freezing between cell runs
   */
  async _cleanupExecutionState() {
    try {
      // Clean up Python global state
      if (this.runtime && this.runtime.isReady) {
        await this.runtime.clearExecutionState();

        // Additional cleanup for specific libraries - use direct Pyodide API
        await this.runtime.pyodide.runPythonAsync(`
# Clean up matplotlib state if present
try:
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.clf()
    plt.cla()
except:
    pass

# Clean up pandas state if present
try:
    import pandas as pd
    # Clear any cached DataFrames
    pass
except:
    pass
`);
      }

      // Clean up WebGPU buffers
      if (this.compute && this.compute.availableStrategies && this.compute.availableStrategies.has('webgpu')) {
        await this.compute.webgpu.bufferManager.forceGC({ emergency: false });
      }

      // Clear memory manager
      if (this.memory) {
        await this.memory.forceGC({ targetReduction: 0.3 });
      }

    } catch (error) {
      // Don't throw - just log cleanup errors
      console.warn('State cleanup warning:', error.message);
    }
  }

  /**
   * Reset instance to allow fresh initialization
   */
  async reset() {
    try {
      await this.destroy();

      // Clear all references
      this.runtime = null;
      this.compute = null;
      this.memory = null;
      this.security = null;
      this.torchAPI = null;
      this.numpy = null;
      this.tensorBridge = null;

      // Reset stats
      this.stats = {
        initTime: 0,
        operations: 0,
        totalExecutionTime: 0,
        averageExecutionTime: 0,
        memoryUsage: 0,
        startTime: performance.now()
      };

      this.errorCount = 0;
      this.lastError = null;

      this.emit('reset:complete');
      return true;
    } catch (error) {
      this.emit('reset:error', { error });
      return false;
    }
  }

  /**
   * Graceful shutdown and resource cleanup
   */
  async destroy() {
    if (!this.isInitialized) {
      return;
    }

    this.emit('destroy:start');

    try {
      // First clean up execution state
      await this._cleanupExecutionState();

      // Cleanup components in reverse order of initialization
      const cleanupPromises = [];

      if (this.compute) {
        cleanupPromises.push(this.compute.cleanup());
      }

      if (this.memory) {
        cleanupPromises.push(this.memory.cleanup());
      }

      if (this.runtime) {
        cleanupPromises.push(this.runtime.cleanup());
      }

      await Promise.all(cleanupPromises);

      // Reset state
      this.isInitialized = false;
      this.initializationPromise = null;
      this.componentsReady = {
        runtime: false,
        compute: false,
        memory: false,
        security: false
      };

      this.emit('destroy:complete');
    } catch (error) {
      this.emit('destroy:error', { error });
      throw error;
    }
  }

  // Private methods
  async _initializeComponents() {
    this.emit('components:init:start');

    // Initialize Security Validator (no dependencies)
    this.security = new SecurityValidator({
      strictMode: this.config.strictSecurity,
      allowEval: this.config.allowEval,
      allowFileSystem: this.config.allowFileSystem,
      allowNetwork: this.config.allowNetwork,
      maxTensorSize: this.config.maxTensorSize,
      maxMemoryMB: this.config.maxMemoryMB
    });
    this.componentsReady.security = true;
    this._forwardEvents(this.security, 'security');

    // Initialize Memory Manager (no dependencies)
    this.memory = new MemoryManager({
      maxMemoryMB: this.config.maxMemoryMB,
      gcThreshold: this.config.gcThreshold,
      enableAutoGC: true
    });
    this.componentsReady.memory = true;
    this._forwardEvents(this.memory, 'memory');

    // Initialize Runtime Manager (depends on security)
    this.runtime = new RuntimeManager({
      pyodideIndexURL: this.config.pyodideIndexURL,
      preloadPackages: this.config.preloadPackages,
      timeout: this.config.initTimeout,
      enableWorkers: this.config.enableWorkers  // Pass worker config
    });
    
    await this.runtime.initialize();
    this.componentsReady.runtime = true;
    this._forwardEvents(this.runtime, 'runtime');

    // Forward execution:stdout events from runtime (for streaming output)
    this.runtime.on('execution:stdout', (data) => this.emit('execution:stdout', data));

    // Initialize Compute Strategy (depends on memory, runtime)
    this.compute = new ComputeStrategy({
      enableWebGPU: this.config.enableWebGPU,
      enableWorkers: this.config.enableWorkers,
      maxWorkers: this.config.maxWorkers,
      webgpu: {
        maxBufferSize: this.config.maxMemoryMB * 1024 * 1024 * 0.8, // 80% of max memory
        enableProfiling: this.config.enableProfiling
      }
    });

    try {
      const strategies = await this.compute.initialize();
      this.componentsReady.compute = true;
      this._forwardEvents(this.compute, 'compute');

      // Log available strategies for debugging
      logger.debug('Compute strategies initialized:', {
        availableStrategies: Array.from(strategies || []),
        hasWebGPU: strategies?.has('webgpu') ?? false
      });
    } catch (error) {
      logger.error('Compute strategy initialization failed:', error);
      // Still mark as ready but with limited functionality
      this.componentsReady.compute = true;
      this._forwardEvents(this.compute, 'compute');
    }

    this.emit('components:init:complete', {
      components: Object.keys(this.componentsReady).filter(k => this.componentsReady[k])
    });
  }

  async _setupPyTorchAPI() {
    this.emit('pytorch:setup:start');

    try {
      // Set up WebGPU tensor integration
      if (this.compute && this.compute.availableStrategies && this.compute.availableStrategies.has('webgpu')) {
        WebGPUTensor.setComputeEngine(this.compute.webgpu);
        this.tensorBridge = createTensorBridge(this.compute.webgpu);

        // Initialize WebGPU bridge for Python operations
        webgpuBridge.setComputeEngine(this.compute.webgpu);
        logger.info('[Greed] WebGPU bridge initialized for Python↔GPU operations');
      }

      // Ensure numpy is loaded before installing PyTorch polyfill
      logger.debug('Ensuring numpy is loaded before PyTorch polyfill installation');
      try {
        await this.runtime.loadPackages(['numpy']);
        logger.debug('Numpy loading completed successfully');
      } catch (error) {
        logger.error('Failed to load numpy:', error);
        throw error;
      }

      // Check if PyTorch polyfill is already installed to prevent re-installation conflicts
      logger.debug('Checking if PyTorch polyfill is already installed');
      const checkResult = await this.runtime.runPython(`
try:
    import torch
    print("PyTorch polyfill already installed")
    _polyfill_installed = True
except ImportError:
    print("PyTorch polyfill not found, installing...")
    _polyfill_installed = False
`, { captureOutput: true });

      const isAlreadyInstalled = checkResult.output && checkResult.output.includes("already installed");

      if (!isAlreadyInstalled) {
        // Install PyTorch polyfill in Python runtime using extracted module
        const polyfillCode = createPyTorchPolyfill();
        validatePolyfill(polyfillCode);

        logger.debug('Installing PyTorch polyfill from extracted module');
        // Increase timeout for large polyfill installation (can take 60-120 seconds)
        await this.runtime.runPython(polyfillCode, {
          captureOutput: false,
          validateInput: false,
          timeout: 180000 // 3 minutes for polyfill installation
        });
      } else {
        logger.debug('PyTorch polyfill already installed, skipping re-installation');
      }
      
      /* Original inline version: await this.runtime.runPython(`
# WebGPU-enabled PyTorch polyfill setup
import numpy as np
import sys

class WebGPUDevice:
    def __init__(self, device_type):
        self.type = device_type
        
    def __str__(self):
        return self.type
        
    def __repr__(self):
        return f"device(type='{self.type}')"

class WebGPUTensor:
    def __init__(self, data, device='cpu', dtype='float32', requires_grad=False):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        
        self.device = WebGPUDevice(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = None
        self.grad_fn = None  # Gradient function for autograd
        
    def numpy(self):
        return self.data
        
    def tolist(self):
        return self.data.tolist()
        
    def view(self, *shape):
        """Reshape tensor maintaining data"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        # Handle -1 for automatic size calculation
        if -1 in shape:
            total_size = self.data.size
            known_size = 1
            unknown_idx = -1
            for i, s in enumerate(shape):
                if s == -1:
                    unknown_idx = i
                else:
                    known_size *= s
            if unknown_idx != -1:
                shape = list(shape)
                shape[unknown_idx] = total_size // known_size
                shape = tuple(shape)
        
        reshaped_data = self.data.reshape(shape)
        return WebGPUTensor(reshaped_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def reshape(self, *shape):
        """Reshape tensor (alias for view)"""
        return self.view(*shape)
    
    def transpose(self, dim0, dim1):
        """Transpose two dimensions"""
        transposed_data = np.swapaxes(self.data, dim0, dim1)
        return WebGPUTensor(transposed_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def permute(self, *dims):
        """Permute the dimensions"""
        permuted_data = np.transpose(self.data, dims)
        return WebGPUTensor(permuted_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def unsqueeze(self, dim):
        """Add a dimension of size 1"""
        new_shape = list(self.data.shape)
        if dim < 0:
            dim = len(new_shape) + dim + 1
        new_shape.insert(dim, 1)
        reshaped_data = self.data.reshape(new_shape)
        return WebGPUTensor(reshaped_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        if dim is None:
            squeezed_data = np.squeeze(self.data)
        else:
            squeezed_data = np.squeeze(self.data, axis=dim)
        return WebGPUTensor(squeezed_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions"""
        if end_dim == -1:
            end_dim = self.ndim - 1
        
        shape = list(self.shape)
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]
        
        new_shape = shape[:start_dim] + [flattened_size] + shape[end_dim + 1:]
        flattened_data = self.data.reshape(new_shape)
        return WebGPUTensor(flattened_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def expand(self, *sizes):
        """Expand tensor to new size"""
        expanded_data = np.broadcast_to(self.data, sizes)
        return WebGPUTensor(expanded_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def repeat(self, *sizes):
        """Repeat tensor along dimensions"""
        repeated_data = np.tile(self.data, sizes)
        return WebGPUTensor(repeated_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def sum(self, dim=None, keepdim=False):
        """Sum tensor elements"""
        if dim is None:
            result_data = np.sum(self.data)
        else:
            result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        
        # Store computation graph for autograd
        if self.requires_grad:
            result._backward_fn = lambda grad: self._sum_backward(grad, dim, keepdim)
            result._inputs = [self]
        
        return result
    
    def mean(self, dim=None, keepdim=False):
        """Mean of tensor elements"""
        if dim is None:
            result_data = np.mean(self.data)
        else:
            result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def std(self, dim=None, keepdim=False):
        """Standard deviation of tensor elements"""
        if dim is None:
            result_data = np.std(self.data)
        else:
            result_data = np.std(self.data, axis=dim, keepdims=keepdim)
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def var(self, dim=None, keepdim=False):
        """Variance of tensor elements"""
        if dim is None:
            result_data = np.var(self.data)
        else:
            result_data = np.var(self.data, axis=dim, keepdims=keepdim)
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def max(self, dim=None, keepdim=False):
        """Maximum values"""
        if dim is None:
            result_data = np.max(self.data)
            return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            values = np.max(self.data, axis=dim, keepdims=keepdim)
            indices = np.argmax(self.data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return (WebGPUTensor(values, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad),
                    WebGPUTensor(indices, device=self.device, dtype='int64'))
    
    def min(self, dim=None, keepdim=False):
        """Minimum values"""
        if dim is None:
            result_data = np.min(self.data)
            return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            values = np.min(self.data, axis=dim, keepdims=keepdim)
            indices = np.argmin(self.data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return (WebGPUTensor(values, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad),
                    WebGPUTensor(indices, device=self.device, dtype='int64'))
    
    def backward(self, gradient=None):
        """Compute gradients via backpropagation"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
        
        if hasattr(self, '_backward_fn'):
            grad = self._backward_fn(gradient)
            for inp in self._inputs:
                if inp.grad is None:
                    inp.grad = WebGPUTensor(grad, device=inp.device, dtype=inp.dtype)
                else:
                    inp.grad.data += grad
    
    def _sum_backward(self, grad, dim, keepdim):
        """Backward pass for sum operation"""
        if dim is None:
            return np.full_like(self.data, grad)
        else:
            if not keepdim:
                grad = np.expand_dims(grad, axis=dim)
            return np.broadcast_to(grad, self.data.shape)
    
    def to(self, device):
        """Move tensor to device"""
        new_device = WebGPUDevice(device) if isinstance(device, str) else device
        return WebGPUTensor(self.data.copy(), device=new_device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def cpu(self):
        """Move tensor to CPU"""
        return self.to('cpu')
    
    def cuda(self):
        """Move tensor to CUDA (simulated via WebGPU)"""
        return self.to('cuda')
        
    def __repr__(self):
        return f"tensor({self.data}, device='{self.device}', dtype='{self.dtype}')"

# Neural network functional operations
class TorchNNFunctional:
    @staticmethod
    def relu(input_tensor):
        """ReLU activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor.data, 0)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.maximum(input_tensor, 0)
    
    @staticmethod
    def sigmoid(input_tensor):
        """Sigmoid activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-input_tensor.data))
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return 1 / (1 + np.exp(-input_tensor))
    
    @staticmethod
    def tanh(input_tensor):
        """Tanh activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.tanh(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.tanh(input_tensor)
    
    @staticmethod
    def softmax(input_tensor, dim=-1):
        """Softmax activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            # Numerical stability: subtract max
            exp_data = np.exp(input_tensor.data - np.max(input_tensor.data, axis=dim, keepdims=True))
            sum_exp = np.sum(exp_data, axis=dim, keepdims=True)
            result_data = exp_data / sum_exp
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            exp_data = np.exp(input_tensor - np.max(input_tensor, axis=dim, keepdims=True))
            return exp_data / np.sum(exp_data, axis=dim, keepdims=True)
    
    @staticmethod
    def cross_entropy(input_tensor, target, reduction='mean'):
        """Cross entropy loss function"""
        if isinstance(input_tensor, WebGPUTensor):
            logits = input_tensor.data
            targets = target.data if isinstance(target, WebGPUTensor) else target
            
            # Apply softmax and compute cross entropy
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # One-hot encoding for targets if needed
            if targets.ndim == 1:
                num_classes = logits.shape[1]
                one_hot_targets = np.eye(num_classes)[targets]
            else:
                one_hot_targets = targets
            
            # Compute cross entropy
            loss = -np.sum(one_hot_targets * np.log(softmax_probs + 1e-8), axis=1)
            
            if reduction == 'mean':
                loss = np.mean(loss)
            elif reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            raise NotImplementedError("cross_entropy requires WebGPUTensor input")
    
    @staticmethod
    def leaky_relu(input_tensor, negative_slope=0.01):
        """Leaky ReLU activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.where(input_tensor.data > 0, input_tensor.data, negative_slope * input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.where(input_tensor > 0, input_tensor, negative_slope * input_tensor)
    
    @staticmethod
    def log_softmax(input_tensor, dim=-1):
        """Log-softmax activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            # Numerical stability
            max_vals = np.max(input_tensor.data, axis=dim, keepdims=True)
            shifted = input_tensor.data - max_vals
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
            result_data = shifted - log_sum_exp
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            max_vals = np.max(input_tensor, axis=dim, keepdims=True)
            shifted = input_tensor - max_vals
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
            return shifted - log_sum_exp
    
    @staticmethod
    def gelu(input_tensor):
        """GELU activation function"""
        if isinstance(input_tensor, WebGPUTensor):
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            x = input_tensor.data
            result_data = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return 0.5 * input_tensor * (1 + np.tanh(np.sqrt(2 / np.pi) * (input_tensor + 0.044715 * input_tensor**3)))
    
    @staticmethod
    def dropout(input_tensor, p=0.5, training=True):
        """Dropout function"""
        if not training or p == 0:
            return input_tensor
        
        if isinstance(input_tensor, WebGPUTensor):
            if p == 1:
                result_data = np.zeros_like(input_tensor.data)
            else:
                mask = np.random.random(input_tensor.data.shape) > p
                result_data = input_tensor.data * mask / (1 - p)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            if p == 1:
                return np.zeros_like(input_tensor)
            else:
                mask = np.random.random(input_tensor.shape) > p
                return input_tensor * mask / (1 - p)
    
    @staticmethod
    def mse_loss(input_tensor, target, reduction='mean'):
        """Mean Squared Error loss function"""
        if isinstance(input_tensor, WebGPUTensor):
            pred_data = input_tensor.data
            target_data = target.data if isinstance(target, WebGPUTensor) else target
            
            loss = (pred_data - target_data) ** 2
            
            if reduction == 'mean':
                loss = np.mean(loss)
            elif reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            loss = (input_tensor - target) ** 2
            if reduction == 'mean':
                return np.mean(loss)
            elif reduction == 'sum':
                return np.sum(loss)
            return loss
    
    @staticmethod
    def binary_cross_entropy(input_tensor, target, reduction='mean'):
        """Binary Cross Entropy loss function"""
        if isinstance(input_tensor, WebGPUTensor):
            pred_data = input_tensor.data
            target_data = target.data if isinstance(target, WebGPUTensor) else target
            
            # Clamp predictions to avoid log(0)
            pred_data = np.clip(pred_data, 1e-8, 1 - 1e-8)
            loss = -(target_data * np.log(pred_data) + (1 - target_data) * np.log(1 - pred_data))
            
            if reduction == 'mean':
                loss = np.mean(loss)
            elif reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            pred_data = np.clip(input_tensor, 1e-8, 1 - 1e-8)
            loss = -(target * np.log(pred_data) + (1 - target) * np.log(1 - pred_data))
            if reduction == 'mean':
                return np.mean(loss)
            elif reduction == 'sum':
                return np.sum(loss)
            return loss
    
    @staticmethod
    def conv2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """2D convolution operation"""
        if isinstance(input_tensor, WebGPUTensor):
            # Simple convolution implementation for demonstration
            # In production, this would use WebGPU compute shaders
            input_data = input_tensor.data
            weight_data = weight.data if isinstance(weight, WebGPUTensor) else weight
            
            # Basic convolution using correlate (simplified)
            from scipy import ndimage
            output_data = ndimage.convolve(input_data, weight_data, mode='constant')
            
            if bias is not None:
                bias_data = bias.data if isinstance(bias, WebGPUTensor) else bias
                output_data += bias_data
            
            return WebGPUTensor(output_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            # Fallback numpy implementation
            from scipy import ndimage
            output = ndimage.convolve(input_tensor, weight, mode='constant')
            if bias is not None:
                output += bias
            return output
    
    @staticmethod
    def max_pool2d(input_tensor, kernel_size, stride=None, padding=0):
        """2D max pooling operation"""
        if isinstance(input_tensor, WebGPUTensor):
            # Simple max pooling implementation
            from skimage.measure import block_reduce
            if stride is None:
                stride = kernel_size
            
            input_data = input_tensor.data
            pool_func = np.max
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            output_data = block_reduce(input_data, kernel_size, pool_func)
            return WebGPUTensor(output_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            from skimage.measure import block_reduce
            if stride is None:
                stride = kernel_size
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            return block_reduce(input_tensor, kernel_size, np.max)
    
    @staticmethod
    def avg_pool2d(input_tensor, kernel_size, stride=None, padding=0):
        """2D average pooling operation"""
        if isinstance(input_tensor, WebGPUTensor):
            # Simple average pooling implementation
            from skimage.measure import block_reduce
            if stride is None:
                stride = kernel_size
            
            input_data = input_tensor.data
            pool_func = np.mean
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            output_data = block_reduce(input_data, kernel_size, pool_func)
            return WebGPUTensor(output_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            from skimage.measure import block_reduce
            if stride is None:
                stride = kernel_size
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            return block_reduce(input_tensor, kernel_size, np.mean)
    
    @staticmethod
    def linear(input_tensor, weight, bias=None):
        """Linear transformation"""
        if isinstance(input_tensor, WebGPUTensor):
            input_data = input_tensor.data
            weight_data = weight.data if isinstance(weight, WebGPUTensor) else weight
            
            output_data = np.dot(input_data, weight_data.T)
            
            if bias is not None:
                bias_data = bias.data if isinstance(bias, WebGPUTensor) else bias
                output_data += bias_data
            
            return WebGPUTensor(output_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            output = np.dot(input_tensor, weight.T)
            if bias is not None:
                output += bias
            return output
    
    @staticmethod
    def batch_norm(input_tensor, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5):
        """Batch normalization operation"""
        if isinstance(input_tensor, WebGPUTensor):
            input_data = input_tensor.data
            
            if training:
                # Compute batch statistics
                batch_mean = np.mean(input_data, axis=0)
                batch_var = np.var(input_data, axis=0)
                
                # Update running statistics
                running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean
                running_var.data = (1 - momentum) * running_var.data + momentum * batch_var
                
                # Normalize using batch statistics
                normalized = (input_data - batch_mean) / np.sqrt(batch_var + eps)
            else:
                # Use running statistics for inference
                mean_data = running_mean.data if isinstance(running_mean, WebGPUTensor) else running_mean
                var_data = running_var.data if isinstance(running_var, WebGPUTensor) else running_var
                normalized = (input_data - mean_data) / np.sqrt(var_data + eps)
            
            # Apply scale and shift
            if weight is not None:
                weight_data = weight.data if isinstance(weight, WebGPUTensor) else weight
                normalized = normalized * weight_data
            
            if bias is not None:
                bias_data = bias.data if isinstance(bias, WebGPUTensor) else bias
                normalized = normalized + bias_data
            
            return WebGPUTensor(normalized, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            # Fallback numpy implementation
            if training:
                batch_mean = np.mean(input_tensor, axis=0)
                batch_var = np.var(input_tensor, axis=0)
                running_mean = (1 - momentum) * running_mean + momentum * batch_mean
                running_var = (1 - momentum) * running_var + momentum * batch_var
                normalized = (input_tensor - batch_mean) / np.sqrt(batch_var + eps)
            else:
                normalized = (input_tensor - running_mean) / np.sqrt(running_var + eps)
            
            if weight is not None:
                normalized = normalized * weight
            if bias is not None:
                normalized = normalized + bias
            return normalized

# CUDA support simulation
class TorchCuda:
    @staticmethod
    def is_available():
        return True  # Simulate CUDA availability via WebGPU
    
    @staticmethod
    def device_count():
        return 1  # Simulate one WebGPU device
    
    @staticmethod
    def current_device():
        return 0  # Simulate current device index

# Optimizer classes
class TorchOptim:
    class SGD:
        def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0):
            self.param_groups = [{'params': list(params), 'lr': lr, 'momentum': momentum, 'dampening': dampening, 'weight_decay': weight_decay}]
            self.state = {}
            
        def zero_grad(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
        
        def step(self):
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                dampening = group['dampening']
                weight_decay = group['weight_decay']
                
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        grad = param.grad.data
                        
                        if weight_decay != 0:
                            grad = grad + weight_decay * param.data
                        
                        if momentum != 0:
                            param_state = self.state.get(id(param), {})
                            if 'momentum_buffer' not in param_state:
                                param_state['momentum_buffer'] = np.zeros_like(grad)
                            buf = param_state['momentum_buffer']
                            buf = momentum * buf + (1 - dampening) * grad
                            param_state['momentum_buffer'] = buf
                            grad = buf
                            self.state[id(param)] = param_state
                        
                        param.data = param.data - lr * grad
    
    class Adam:
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
            self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}]
            self.state = {}
            
        def zero_grad(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
        
        def step(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        grad = param.grad.data
                        
                        param_state = self.state.get(id(param), {})
                        
                        # Initialize state
                        if len(param_state) == 0:
                            param_state['step'] = 0
                            param_state['exp_avg'] = np.zeros_like(grad)
                            param_state['exp_avg_sq'] = np.zeros_like(grad)
                        
                        exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                        beta1, beta2 = group['betas']
                        
                        param_state['step'] += 1
                        
                        if group['weight_decay'] != 0:
                            grad = grad + group['weight_decay'] * param.data
                        
                        # Exponential moving average of gradient values
                        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                        # Exponential moving average of squared gradient values
                        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
                        
                        param_state['exp_avg'] = exp_avg
                        param_state['exp_avg_sq'] = exp_avg_sq
                        
                        bias_correction1 = 1 - beta1 ** param_state['step']
                        bias_correction2 = 1 - beta2 ** param_state['step']
                        
                        step_size = group['lr'] / bias_correction1
                        bias_correction2_sqrt = np.sqrt(bias_correction2)
                        
                        param.data = param.data - step_size * exp_avg / (np.sqrt(exp_avg_sq) / bias_correction2_sqrt + group['eps'])
                        
                        self.state[id(param)] = param_state
    
    class AdamW:
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}]
            self.state = {}
            
        def zero_grad(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
        
        def step(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        grad = param.grad.data
                        
                        param_state = self.state.get(id(param), {})
                        
                        # Initialize state
                        if len(param_state) == 0:
                            param_state['step'] = 0
                            param_state['exp_avg'] = np.zeros_like(grad)
                            param_state['exp_avg_sq'] = np.zeros_like(grad)
                        
                        exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                        beta1, beta2 = group['betas']
                        
                        param_state['step'] += 1
                        
                        # Exponential moving average of gradient values
                        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                        # Exponential moving average of squared gradient values
                        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
                        
                        param_state['exp_avg'] = exp_avg
                        param_state['exp_avg_sq'] = exp_avg_sq
                        
                        bias_correction1 = 1 - beta1 ** param_state['step']
                        bias_correction2 = 1 - beta2 ** param_state['step']
                        
                        step_size = group['lr'] / bias_correction1
                        bias_correction2_sqrt = np.sqrt(bias_correction2)
                        
                        # AdamW weight decay (decoupled from gradient)
                        param.data = param.data * (1 - group['lr'] * group['weight_decay'])
                        param.data = param.data - step_size * exp_avg / (np.sqrt(exp_avg_sq) / bias_correction2_sqrt + group['eps'])
                        
                        self.state[id(param)] = param_state
    
    class RMSprop:
        def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
            self.param_groups = [{'params': list(params), 'lr': lr, 'alpha': alpha, 'eps': eps, 'weight_decay': weight_decay, 'momentum': momentum}]
            self.state = {}
            
        def zero_grad(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
        
        def step(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        grad = param.grad.data
                        
                        param_state = self.state.get(id(param), {})
                        
                        # Initialize state
                        if len(param_state) == 0:
                            param_state['square_avg'] = np.zeros_like(grad)
                            if group['momentum'] > 0:
                                param_state['momentum_buffer'] = np.zeros_like(grad)
                        
                        square_avg = param_state['square_avg']
                        alpha = group['alpha']
                        
                        if group['weight_decay'] != 0:
                            grad = grad + group['weight_decay'] * param.data
                        
                        # Exponential moving average of squared gradients
                        square_avg = alpha * square_avg + (1 - alpha) * grad * grad
                        param_state['square_avg'] = square_avg
                        
                        avg = np.sqrt(square_avg) + group['eps']
                        
                        if group['momentum'] > 0:
                            buf = param_state['momentum_buffer']
                            buf = group['momentum'] * buf + grad / avg
                            param_state['momentum_buffer'] = buf
                            param.data = param.data - group['lr'] * buf
                        else:
                            param.data = param.data - group['lr'] * grad / avg
                        
                        self.state[id(param)] = param_state
    
    class Adagrad:
        def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10):
            self.param_groups = [{'params': list(params), 'lr': lr, 'lr_decay': lr_decay, 'weight_decay': weight_decay, 'eps': eps}]
            self.state = {}
            
        def zero_grad(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
        
        def step(self):
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'grad') and param.grad is not None:
                        grad = param.grad.data
                        
                        param_state = self.state.get(id(param), {})
                        
                        # Initialize state
                        if len(param_state) == 0:
                            param_state['step'] = 0
                            param_state['sum'] = np.zeros_like(grad)
                        
                        param_state['step'] += 1
                        
                        if group['weight_decay'] != 0:
                            grad = grad + group['weight_decay'] * param.data
                        
                        # Accumulate squared gradients
                        param_state['sum'] += grad * grad
                        
                        # Compute learning rate with decay
                        clr = group['lr'] / (1 + (param_state['step'] - 1) * group['lr_decay'])
                        
                        # Update parameters
                        param.data = param.data - clr * grad / (np.sqrt(param_state['sum']) + group['eps'])
                        
                        self.state[id(param)] = param_state

# Neural network modules
class TorchNNModule:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        
    def parameters(self):
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError

class TorchNNLinear(TorchNNModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.weight = WebGPUTensor(weight_data, requires_grad=True)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = WebGPUTensor(bias_data, requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result = WebGPUTensor(np.dot(x.data, self.weight.data.T), device=x.device, dtype=x.dtype)
            if self.bias is not None:
                result.data = result.data + self.bias.data
            return result
        else:
            raise TypeError("Input must be WebGPUTensor")

class TorchNNReLU(TorchNNModule):
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result_data = np.maximum(x.data, 0)
            return WebGPUTensor(result_data, device=x.device, dtype=x.dtype)
        else:
            return np.maximum(x, 0)

class TorchNNMSELoss(TorchNNModule):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        if isinstance(input_tensor, WebGPUTensor) and isinstance(target, WebGPUTensor):
            diff = input_tensor.data - target.data
            loss = diff ** 2
            
            if self.reduction == 'mean':
                loss = np.mean(loss)
            elif self.reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            raise TypeError("Both inputs must be WebGPUTensor")

class TorchNNSequential(TorchNNModule):
    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)
        for i, layer in enumerate(self.layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Placeholder classes for advanced modules
class TorchNNConv2d(TorchNNModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Simplified implementation - just return a placeholder
    
    def forward(self, x):
        # Placeholder implementation
        return x

class TorchNNMaxPool2d(TorchNNModule):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        return x

class TorchNNBatchNorm2d(TorchNNModule):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
    
    def forward(self, x):
        return x

class TorchNNDropout(TorchNNModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return x

class TorchNNSigmoid(TorchNNModule):
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-x.data))
            return WebGPUTensor(result_data, device=x.device, dtype=x.dtype)
        else:
            return 1 / (1 + np.exp(-x))

class TorchNNTanh(TorchNNModule):
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result_data = np.tanh(x.data)
            return WebGPUTensor(result_data, device=x.device, dtype=x.dtype)
        else:
            return np.tanh(x)

class TorchNNSoftmax(TorchNNModule):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            exp_data = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
            sum_exp = np.sum(exp_data, axis=self.dim, keepdims=True)
            result_data = exp_data / sum_exp
            return WebGPUTensor(result_data, device=x.device, dtype=x.dtype)
        else:
            exp_data = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
            return exp_data / np.sum(exp_data, axis=self.dim, keepdims=True)

class TorchNNLeakyReLU(TorchNNModule):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result_data = np.where(x.data > 0, x.data, self.negative_slope * x.data)
            return WebGPUTensor(result_data, device=x.device, dtype=x.dtype)
        else:
            return np.where(x > 0, x, self.negative_slope * x)

class TorchNNCrossEntropyLoss(TorchNNModule):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        if isinstance(input_tensor, WebGPUTensor):
            logits = input_tensor.data
            targets = target.data if isinstance(target, WebGPUTensor) else target
            
            # Apply softmax and compute cross entropy
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # One-hot encoding for targets if needed
            if targets.ndim == 1:
                num_classes = logits.shape[1]
                one_hot_targets = np.eye(num_classes)[targets]
            else:
                one_hot_targets = targets
            
            # Compute cross entropy
            loss = -np.sum(one_hot_targets * np.log(softmax_probs + 1e-8), axis=1)
            
            if self.reduction == 'mean':
                loss = np.mean(loss)
            elif self.reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            raise TypeError("Input must be WebGPUTensor")

class TorchNNBCELoss(TorchNNModule):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        if isinstance(input_tensor, WebGPUTensor):
            pred_data = np.clip(input_tensor.data, 1e-8, 1 - 1e-8)
            target_data = target.data if isinstance(target, WebGPUTensor) else target
            
            loss = -(target_data * np.log(pred_data) + (1 - target_data) * np.log(1 - pred_data))
            
            if self.reduction == 'mean':
                loss = np.mean(loss)
            elif self.reduction == 'sum':
                loss = np.sum(loss)
            
            return WebGPUTensor(loss, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            raise TypeError("Input must be WebGPUTensor")

# Neural network module
class TorchNN:
    def __init__(self):
        self.functional = TorchNNFunctional()
        # Basic layers
        self.Linear = TorchNNLinear
        self.Conv2d = TorchNNConv2d
        self.Conv1d = TorchNNConv2d  # Placeholder
        self.MaxPool2d = TorchNNMaxPool2d
        self.AvgPool2d = TorchNNMaxPool2d  # Placeholder
        self.BatchNorm2d = TorchNNBatchNorm2d
        self.BatchNorm1d = TorchNNBatchNorm2d  # Placeholder
        self.Dropout = TorchNNDropout
        # Activation functions
        self.ReLU = TorchNNReLU
        self.LeakyReLU = TorchNNLeakyReLU
        self.Sigmoid = TorchNNSigmoid
        self.Tanh = TorchNNTanh
        self.Softmax = TorchNNSoftmax
        # Loss functions
        self.MSELoss = TorchNNMSELoss
        self.CrossEntropyLoss = TorchNNCrossEntropyLoss
        self.BCELoss = TorchNNBCELoss
        # Container modules
        self.Sequential = TorchNNSequential
        # Advanced modules (placeholders)
        self.LSTM = TorchNNModule  # Placeholder
        self.GRU = TorchNNModule   # Placeholder
        self.Embedding = TorchNNModule  # Placeholder

# Create torch module with essential functions
class TorchModule:
    def __init__(self):
        self.tensor = self._tensor
        self.zeros = self._zeros
        self.ones = self._ones
        self.randn = self._randn
        self.randint = self._randint
        self.arange = self._arange
        self.linspace = self._linspace
        self.eye = self._eye
        self.empty = self._empty
        self.full = self._full
        self.add = self._add
        self.sub = self._sub
        self.mul = self._mul
        self.div = self._div
        self.matmul = self._matmul
        self.mm = self._mm
        self.bmm = self._bmm
        self.dot = self._dot
        self.cross = self._cross
        self.pow = self._pow
        self.sqrt = self._sqrt
        self.exp = self._exp
        self.log = self._log
        self.abs = self._abs
        self.sin = self._sin
        self.cos = self._cos
        self.tan = self._tan
        self.floor = self._floor
        self.ceil = self._ceil
        self.round = self._round
        self.sigmoid = self._sigmoid
        self.tanh = self._tanh_fn
        self.cat = self._cat
        self.stack = self._stack
        self.split = self._split
        self.chunk = self._chunk
        self.argmax = self._argmax
        self.argmin = self._argmin
        self.topk = self._topk
        self.sort = self._sort
        self.argsort = self._argsort
        self.median = self._median
        self.from_numpy = self._from_numpy
        self.device = self._device
        self.nn = TorchNN()
        self.optim = TorchOptim()
        self.cuda = TorchCuda()
        self.no_grad = self._no_grad
        self.enable_grad = self._enable_grad
        
    def _tensor(self, data, **kwargs):
        return WebGPUTensor(data, **kwargs)
    
    def _zeros(self, *shape, **kwargs):
        data = np.zeros(shape)
        return WebGPUTensor(data, **kwargs)
    
    def _ones(self, *shape, **kwargs):
        data = np.ones(shape)
        return WebGPUTensor(data, **kwargs)
    
    def _randn(self, *shape, **kwargs):
        data = np.random.randn(*shape)
        return WebGPUTensor(data, **kwargs)
    
    def _randint(self, low, high, size, **kwargs):
        data = np.random.randint(low, high, size)
        return WebGPUTensor(data, **kwargs)
        
    def _arange(self, start, end=None, step=1, **kwargs):
        if end is None:
            end = start
            start = 0
        data = np.arange(start, end, step)
        return WebGPUTensor(data, **kwargs)
    
    def _linspace(self, start, end, steps, **kwargs):
        data = np.linspace(start, end, steps)
        return WebGPUTensor(data, **kwargs)
    
    def _eye(self, n, m=None, **kwargs):
        if m is None:
            m = n
        data = np.eye(n, m)
        return WebGPUTensor(data, **kwargs)
    
    def _empty(self, *shape, **kwargs):
        data = np.empty(shape)
        return WebGPUTensor(data, **kwargs)
    
    def _full(self, size, fill_value, **kwargs):
        data = np.full(size, fill_value)
        return WebGPUTensor(data, **kwargs)
    
    def _add(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(a.data + b.data, device=a.device)
        elif isinstance(a, WebGPUTensor):
            return WebGPUTensor(a.data + b, device=a.device)
        else:
            return WebGPUTensor(a + b.data, device=b.device)
    
    def _sub(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(a.data - b.data, device=a.device)
        elif isinstance(a, WebGPUTensor):
            return WebGPUTensor(a.data - b, device=a.device)
        else:
            return WebGPUTensor(a - b.data, device=b.device)
    
    def _mul(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(a.data * b.data, device=a.device)
        elif isinstance(a, WebGPUTensor):
            return WebGPUTensor(a.data * b, device=a.device)
        else:
            return WebGPUTensor(a * b.data, device=b.device)
    
    def _div(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(a.data / b.data, device=a.device)
        elif isinstance(a, WebGPUTensor):
            return WebGPUTensor(a.data / b, device=a.device)
        else:
            return WebGPUTensor(a / b.data, device=b.device)
    
    def _matmul(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(np.dot(a.data, b.data), device=a.device)
        return WebGPUTensor(np.dot(a, b))
    
    def _mm(self, a, b):
        """Matrix multiplication (alias for matmul)"""
        return self._matmul(a, b)
    
    def _bmm(self, a, b):
        """Batch matrix multiplication"""
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            result_data = np.matmul(a.data, b.data)
            return WebGPUTensor(result_data, device=a.device)
        return WebGPUTensor(np.matmul(a, b))
    
    def _dot(self, a, b):
        """Dot product"""
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            result_data = np.dot(a.data, b.data)
            return WebGPUTensor(result_data, device=a.device)
        return WebGPUTensor(np.dot(a, b))
    
    def _cross(self, a, b, dim=-1):
        """Cross product"""
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            result_data = np.cross(a.data, b.data, axis=dim)
            return WebGPUTensor(result_data, device=a.device)
        return WebGPUTensor(np.cross(a, b, axis=dim))
    
    def _pow(self, input_tensor, exponent):
        """Element-wise power operation"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.power(input_tensor.data, exponent)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.power(input_tensor, exponent)
    
    def _sqrt(self, input_tensor):
        """Element-wise square root"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sqrt(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.sqrt(input_tensor)
    
    def _exp(self, input_tensor):
        """Element-wise exponential"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.exp(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.exp(input_tensor)
    
    def _log(self, input_tensor):
        """Element-wise natural logarithm"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.log(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.log(input_tensor)
    
    def _abs(self, input_tensor):
        """Element-wise absolute value"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.abs(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.abs(input_tensor)
    
    def _sin(self, input_tensor):
        """Element-wise sine"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sin(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.sin(input_tensor)
    
    def _cos(self, input_tensor):
        """Element-wise cosine"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.cos(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.cos(input_tensor)
    
    def _tan(self, input_tensor):
        """Element-wise tangent"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.tan(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.tan(input_tensor)
    
    def _floor(self, input_tensor):
        """Element-wise floor"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.floor(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.floor(input_tensor)
    
    def _ceil(self, input_tensor):
        """Element-wise ceiling"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.ceil(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.ceil(input_tensor)
    
    def _round(self, input_tensor):
        """Element-wise round"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.round(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.round(input_tensor)
    
    def _sigmoid(self, input_tensor):
        """Element-wise sigmoid"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-input_tensor.data))
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return 1 / (1 + np.exp(-input_tensor))
    
    def _tanh_fn(self, input_tensor):
        """Element-wise tanh"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.tanh(input_tensor.data)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.tanh(input_tensor)
    
    def _cat(self, tensors, dim=0):
        """Concatenate tensors along a dimension"""
        if all(isinstance(t, WebGPUTensor) for t in tensors):
            data_list = [t.data for t in tensors]
            result_data = np.concatenate(data_list, axis=dim)
            return WebGPUTensor(result_data, device=tensors[0].device, dtype=tensors[0].dtype)
        else:
            raise TypeError("All tensors must be WebGPUTensor instances")
    
    def _stack(self, tensors, dim=0):
        """Stack tensors along a new dimension"""
        if all(isinstance(t, WebGPUTensor) for t in tensors):
            data_list = [t.data for t in tensors]
            result_data = np.stack(data_list, axis=dim)
            return WebGPUTensor(result_data, device=tensors[0].device, dtype=tensors[0].dtype)
        else:
            raise TypeError("All tensors must be WebGPUTensor instances")
    
    def _split(self, tensor, split_size_or_sections, dim=0):
        """Split tensor into chunks"""
        if isinstance(tensor, WebGPUTensor):
            if isinstance(split_size_or_sections, int):
                split_data = np.array_split(tensor.data, tensor.shape[dim] // split_size_or_sections, axis=dim)
            else:
                split_data = np.split(tensor.data, split_size_or_sections, axis=dim)
            return [WebGPUTensor(chunk, device=tensor.device, dtype=tensor.dtype) for chunk in split_data]
        else:
            raise TypeError("Input must be WebGPUTensor")
    
    def _chunk(self, tensor, chunks, dim=0):
        """Split tensor into specific number of chunks"""
        if isinstance(tensor, WebGPUTensor):
            split_data = np.array_split(tensor.data, chunks, axis=dim)
            return [WebGPUTensor(chunk, device=tensor.device, dtype=tensor.dtype) for chunk in split_data]
        else:
            raise TypeError("Input must be WebGPUTensor")
    
    def _argmax(self, input_tensor, dim=None, keepdim=False):
        """Return indices of maximum values"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.argmax(input_tensor.data)
            else:
                result_data = np.argmax(input_tensor.data, axis=dim)
                if keepdim:
                    result_data = np.expand_dims(result_data, axis=dim)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype='int64')
        else:
            return np.argmax(input_tensor, axis=dim)
    
    def _argmin(self, input_tensor, dim=None, keepdim=False):
        """Return indices of minimum values"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.argmin(input_tensor.data)
            else:
                result_data = np.argmin(input_tensor.data, axis=dim)
                if keepdim:
                    result_data = np.expand_dims(result_data, axis=dim)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype='int64')
        else:
            return np.argmin(input_tensor, axis=dim)
    
    def _topk(self, input_tensor, k, dim=-1, largest=True, sorted=True):
        """Return k largest/smallest elements and their indices"""
        if isinstance(input_tensor, WebGPUTensor):
            data = input_tensor.data
            if not largest:
                # For smallest, negate the data
                data = -data
            
            # Get top k indices
            if dim == -1 or dim == data.ndim - 1:
                indices = np.argpartition(data, -k, axis=dim)[..., -k:]
                if sorted:
                    # Sort the top k
                    sort_indices = np.argsort(np.take_along_axis(data, indices, axis=dim), axis=dim)
                    indices = np.take_along_axis(indices, sort_indices[..., ::-1], axis=dim)
            else:
                indices = np.argpartition(data, -k, axis=dim)
                indices = np.take(indices, range(-k, 0), axis=dim)
            
            values = np.take_along_axis(input_tensor.data, indices, axis=dim)
            
            if not largest:
                values = -values
            
            return (WebGPUTensor(values, device=input_tensor.device, dtype=input_tensor.dtype),
                    WebGPUTensor(indices, device=input_tensor.device, dtype='int64'))
        else:
            raise TypeError("topk requires WebGPUTensor input")
    
    def _sort(self, input_tensor, dim=-1, descending=False):
        """Sort tensor elements"""
        if isinstance(input_tensor, WebGPUTensor):
            sorted_data = np.sort(input_tensor.data, axis=dim)
            indices = np.argsort(input_tensor.data, axis=dim)
            if descending:
                sorted_data = np.flip(sorted_data, axis=dim)
                indices = np.flip(indices, axis=dim)
            return (WebGPUTensor(sorted_data, device=input_tensor.device, dtype=input_tensor.dtype),
                    WebGPUTensor(indices, device=input_tensor.device, dtype='int64'))
        else:
            raise TypeError("sort requires WebGPUTensor input")
    
    def _argsort(self, input_tensor, dim=-1, descending=False):
        """Return indices that sort the tensor"""
        if isinstance(input_tensor, WebGPUTensor):
            indices = np.argsort(input_tensor.data, axis=dim)
            if descending:
                indices = np.flip(indices, axis=dim)
            return WebGPUTensor(indices, device=input_tensor.device, dtype='int64')
        else:
            return np.argsort(input_tensor, axis=dim)
    
    def _median(self, input_tensor, dim=None, keepdim=False):
        """Return median values"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.median(input_tensor.data)
            else:
                result_data = np.median(input_tensor.data, axis=dim, keepdims=keepdim)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.median(input_tensor, axis=dim, keepdims=keepdim)
    
    def _from_numpy(self, ndarray):
        """Create tensor from numpy array"""
        return WebGPUTensor(ndarray)
    
    def _no_grad(self):
        """Context manager that disables gradient computation"""
        class NoGradContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return NoGradContext()
    
    def _enable_grad(self):
        """Context manager that enables gradient computation"""
        class EnableGradContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return EnableGradContext()
    
    def _device(self, device_string):
        """Create a device object"""
        return WebGPUDevice(device_string)

# Install in global namespace
torch = TorchModule()
sys.modules['torch'] = torch

# Create torch.nn.functional module
functional_module = TorchNNFunctional()
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = functional_module

# Create torch.optim module
sys.modules['torch.optim'] = torch.optim

# Create torch.cuda module  
sys.modules['torch.cuda'] = torch.cuda
      `, { captureOutput: false }); */

      // Get references to Python objects
      this.torchAPI = this.runtime.getGlobal('torch');
      this.numpy = this.runtime.getGlobal('np');

      // Inject WebGPU bridge functions into Python if WebGPU is available
      if (this.compute && this.compute.availableStrategies && this.compute.availableStrategies.has('webgpu')) {
        logger.info('[Greed] Injecting WebGPU bridge functions into Python');

        // Set Python globals to call JavaScript WebGPU bridge
        this.runtime.setGlobal('__webgpu_allocate__', (data, shape, dtype) => {
          return webgpuBridge.allocate(data, shape, dtype);
        });

        this.runtime.setGlobal('__webgpu_read__', async (bufferId) => {
          return await webgpuBridge.read(bufferId);
        });

        this.runtime.setGlobal('__webgpu_matmul__', async (aId, bId, aShape, bShape) => {
          return await webgpuBridge.matmul(aId, bId, aShape, bShape);
        });

        this.runtime.setGlobal('__webgpu_add__', async (aId, bId, shape) => {
          return await webgpuBridge.add(aId, bId, shape);
        });

        this.runtime.setGlobal('__webgpu_multiply__', async (aId, bId, shape) => {
          return await webgpuBridge.multiply(aId, bId, shape);
        });

        this.runtime.setGlobal('__webgpu_relu__', async (aId, shape) => {
          return await webgpuBridge.relu(aId, shape);
        });

        this.runtime.setGlobal('__webgpu_transpose__', async (aId, shape) => {
          return await webgpuBridge.transpose(aId, shape);
        });

        this.runtime.setGlobal('__webgpu_free__', (bufferId) => {
          return webgpuBridge.free(bufferId);
        });

        logger.info('[Greed] WebGPU bridge functions injected successfully');
      }

      this.emit('pytorch:setup:complete');
    } catch (error) {
      this.emit('pytorch:setup:error', { error });
      throw new Error(`PyTorch API setup failed: ${error.message}`);
    }
  }

  async _validateSystemReadiness() {
    this.emit('validation:start');

    const checks = [
      { name: 'runtime', check: () => this.runtime.isReady },
      { name: 'compute', check: () => this.compute && this.compute.isInitialized },
      { name: 'memory', check: () => this.memory.getStats().memoryUsage >= 0 },
      { name: 'security', check: () => this.security.getStats().totalValidations >= 0 },
      { name: 'pytorch', check: () => this.torchAPI !== null }
    ];

    const results = [];
    for (const { name, check } of checks) {
      try {
        const passed = check();
        results.push({ name, passed, error: null });
        if (!passed) {
          throw new Error(`${name} component failed readiness check`);
        }
      } catch (error) {
        results.push({ name, passed: false, error: error.message });
        throw new Error(`System validation failed for ${name}: ${error.message}`);
      }
    }

    this.emit('validation:complete', { results });
  }

  _validateConfig(config) {
    const required = ['pyodideIndexURL', 'maxMemoryMB'];
    for (const key of required) {
      if (config[key] === undefined) {
        throw new Error(`Required configuration missing: ${key}`);
      }
    }

    if (config.maxMemoryMB <= 0 || config.maxMemoryMB > 4096) {
      throw new Error(`maxMemoryMB must be between 1 and 4096, got ${config.maxMemoryMB}`);
    }

    return config;
  }

  _forwardEvents(component, prefix) {
    const eventsToForward = [
      'error', 'warning', 'init:complete', 'init:error',
      'cleanup:complete', 'cleanup:error'
    ];

    for (const event of eventsToForward) {
      component.on(event, (data) => {
        this.emit(`${prefix}:${event}`, data);
      });
    }
  }

  _setupGlobalErrorHandling() {
    this.on('error', (error) => {
      // System error handling - error details available via getStats()
      this.lastError = error;
      this.errorCount++;
    });

    // Forward component errors to main error handler
    const errorEvents = [
      'security:error', 'memory:error', 'runtime:error', 'compute:error'
    ];

    for (const event of errorEvents) {
      this.on(event, (data) => {
        this.emit('error', new Error(`${event}: ${data.error?.message || 'Unknown error'}`));
      });
    }
  }

  _updateOperationStats(executionTime) {
    this.stats.operations++;
    this.stats.totalExecutionTime += executionTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.operations;
    
    if (this.memory) {
      this.stats.memoryUsage = this.memory.getStats().memoryUsageMB;
    }
  }

  _generateOperationId() {
    return `greed_op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Export as default and named export for compatibility
export default Greed;
export { Greed };