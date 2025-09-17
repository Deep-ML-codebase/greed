/**
 * PipelineCache - WebGPU compute pipeline management and caching
 * Optimizes shader compilation and pipeline creation with intelligent caching
 */
import EventEmitter from '../../core/event-emitter.js';
import { WebGPUShaders } from './webgpu-shaders.js';

class PipelineCache extends EventEmitter {
  constructor(device, options = {}) {
    super();
    this.device = device;
    this.config = {
      maxCacheSize: options.maxCacheSize || 100,
      enableWarmup: options.enableWarmup !== false,
      enableMetrics: options.enableMetrics !== false,
      shaderOptimization: options.shaderOptimization || 'balanced', // 'speed', 'size', 'balanced'
      ...options
    };

    // Pipeline and shader caches
    this.pipelines = new Map(); // operationKey -> ComputePipeline
    this.shaderModules = new Map(); // shaderHash -> ShaderModule
    this.bindGroupLayouts = new Map(); // layoutKey -> BindGroupLayout
    
    // LRU tracking for cache eviction
    this.accessOrder = new Map(); // key -> timestamp
    
    // Compilation queue for async pipeline creation
    this.compilationQueue = new Map(); // key -> Promise
    
    // Statistics
    this.stats = {
      hits: 0,
      misses: 0,
      compilations: 0,
      evictions: 0,
      averageCompileTime: 0,
      totalCompileTime: 0
    };

    // Shader templates for common operations
    this.shaderTemplates = WebGPUShaders.getShaderTemplates();
  }

  /**
   * Get or create compute pipeline for operation
   */
  async get(operation, options = {}) {

    const key = this._generateKey(operation, options);

    // Check if already compiled or in progress
    if (this.pipelines.has(key)) {
      this._updateAccess(key);
      this.stats.hits++;
      this.emit('cache:hit', { operation, key });
      return this.pipelines.get(key);
    }

    // Check if compilation is in progress
    if (this.compilationQueue.has(key)) {
      this.emit('cache:wait', { operation, key });
      return await this.compilationQueue.get(key);
    }
    
    // Start compilation
    this.stats.misses++;
    this.emit('cache:miss', { operation, key });

    const compilationPromise = this._compilePipeline(operation, options, key);
    this.compilationQueue.set(key, compilationPromise);

    try {
      const pipeline = await compilationPromise;
      this.compilationQueue.delete(key);
      return pipeline;
    } catch (error) {
      this.compilationQueue.delete(key);
      throw error;
    }
  }

  /**
   * Precompile common operations for faster first use
   */
  async warmup(operations = null) {
    if (!this.config.enableWarmup) {
      return;
    }

    const commonOps = operations || [
      'add', 'multiply', 'matmul', 'relu', 'sigmoid', 
      'softmax', 'conv2d', 'maxpool', 'transpose'
    ];

    this.emit('warmup:start', { operations: commonOps });
    const startTime = performance.now();
    
    const warmupPromises = commonOps.map(async (op) => {
      try {
        await this.get(op, { warmup: true });
        this.emit('warmup:operation', { operation: op });
      } catch (error) {
        this.emit('warmup:error', { operation: op, error });
      }
    });
    
    await Promise.allSettled(warmupPromises);
    
    const duration = performance.now() - startTime;
    this.emit('warmup:complete', { operations: commonOps, duration });
  }

  /**
   * Get bind group layout for operation
   */
  getBindGroupLayout(operation, options = {}) {
    const key = this._generateLayoutKey(operation, options);
    
    if (this.bindGroupLayouts.has(key)) {
      return this.bindGroupLayouts.get(key);
    }
    
    const layout = this._createBindGroupLayout(operation, options);
    this.bindGroupLayouts.set(key, layout);
    
    return layout;
  }

  /**
   * Get optimal workgroup size for operation
   */
  getOptimalWorkgroupSize(operation, tensorShape, deviceLimits) {
    return WebGPUShaders.getOptimalWorkgroupSize(operation, tensorShape, deviceLimits);
  }

  /**
   * Generate parameters for operation
   */
  generateOperationParams(operation, tensors, options = {}) {
    return WebGPUShaders.generateParams(operation, tensors, options);
  }

  /**
   * Create shader module with caching
   */
  async createShaderModule(shaderSource, options = {}) {
    const hash = this._hashString(shaderSource);
    
    if (this.shaderModules.has(hash)) {
      return this.shaderModules.get(hash);
    }
    
    try {
      const shaderModule = this.device.createShaderModule({
        code: shaderSource,
        ...options
      });
      
      this.shaderModules.set(hash, shaderModule);
      this.emit('shader:compiled', { hash, size: shaderSource.length });
      
      return shaderModule;
    } catch (error) {
      this.emit('shader:error', { hash, error, source: shaderSource.substring(0, 100) });
      throw error;
    }
  }

  /**
   * Generate optimized shader for operation
   */
  generateShader(operation, options = {}) {
    const template = this.shaderTemplates.get(operation);
    if (!template) {
      throw new Error(`No shader template found for operation: ${operation}`);
    }

    const shaderOptions = {
      workgroupSize: options.workgroupSize || [8, 8, 1],
      dataType: options.dataType || 'f32',
      optimization: this.config.shaderOptimization,
      ...options
    };

    return template(shaderOptions);
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0 ? 
      this.stats.hits / (this.stats.hits + this.stats.misses) : 0;
    
    return {
      ...this.stats,
      hitRate,
      cacheSize: this.pipelines.size,
      shaderCacheSize: this.shaderModules.size,
      layoutCacheSize: this.bindGroupLayouts.size,
      averageCompileTimeMs: Math.round(this.stats.averageCompileTime * 100) / 100
    };
  }

  /**
   * Clear cache and reset statistics
   */
  clear() {
    this.pipelines.clear();
    this.shaderModules.clear();
    this.bindGroupLayouts.clear();
    this.accessOrder.clear();
    this.compilationQueue.clear();
    
    // Reset stats but keep historical data
    this.stats.hits = 0;
    this.stats.misses = 0;
    
    this.emit('cache:cleared');
  }

  /**
   * Cleanup resources
   */
  cleanup() {
    this.clear();
    this.shaderTemplates.clear();
    this.emit('cleanup:complete');
  }

  // Private methods
  async _compilePipeline(operation, options, key) {
    const startTime = performance.now();

    try {

      // Generate shader source
      const shaderSource = this.generateShader(operation, options);

      // Create shader module
      const shaderModule = await this.createShaderModule(shaderSource);

      // Create bind group layout
      const bindGroupLayout = this.getBindGroupLayout(operation, options);

      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      });

      // Create compute pipeline
      const pipeline = await this.device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
      
      // Store in cache
      this.pipelines.set(key, pipeline);
      this._updateAccess(key);
      this._enforceMaxCacheSize();
      
      // Update statistics
      const compileTime = performance.now() - startTime;
      this.stats.compilations++;
      this.stats.totalCompileTime += compileTime;
      this.stats.averageCompileTime = this.stats.totalCompileTime / this.stats.compilations;
      
      this.emit('pipeline:compiled', { 
        operation, 
        key, 
        compileTime,
        cacheSize: this.pipelines.size
      });
      
      return pipeline;
    } catch (error) {
      this.emit('pipeline:error', { operation, key, error });
      throw error;
    }
  }

  _generateKey(operation, options) {
    const keyParts = [
      operation,
      options.workgroupSize?.join(',') || '8,8,1',
      options.dataType || 'f32',
      options.inputCount || 2,
      options.outputCount || 1,
      JSON.stringify(options.constants || {})
    ];
    
    return keyParts.join('|');
  }

  _generateLayoutKey(operation, options) {
    return `${operation}|${options.inputCount || 2}|${options.outputCount || 1}`;
  }

  _createBindGroupLayout(operation, options) {
    const layout = WebGPUShaders.getBufferLayout(operation, options.inputCount, options.outputCount);
    const entries = [];
    
    // Input buffers (read-only)
    for (let i = 0; i < layout.inputs; i++) {
      entries.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      });
    }
    
    // Output buffers (write-only)
    for (let i = 0; i < layout.outputs; i++) {
      entries.push({
        binding: layout.inputs + i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      });
    }
    
    // Uniform buffer for parameters
    if (layout.uniforms > 0) {
      entries.push({
        binding: layout.inputs + layout.outputs,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      });
    }
    
    return this.device.createBindGroupLayout({ entries });
  }

  _updateAccess(key) {
    this.accessOrder.set(key, performance.now());
  }

  _enforceMaxCacheSize() {
    if (this.pipelines.size <= this.config.maxCacheSize) {
      return;
    }
    
    // Find least recently used item
    let lruKey = null;
    let lruTime = Infinity;
    
    for (const [key, time] of this.accessOrder.entries()) {
      if (time < lruTime) {
        lruTime = time;
        lruKey = key;
      }
    }
    
    if (lruKey) {
      this.pipelines.delete(lruKey);
      this.accessOrder.delete(lruKey);
      this.stats.evictions++;
      this.emit('cache:eviction', { key: lruKey, cacheSize: this.pipelines.size });
    }
  }

  _hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

}

export default PipelineCache;