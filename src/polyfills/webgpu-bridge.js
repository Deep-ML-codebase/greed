/**
 * WebGPU Bridge - Connects Python PyTorch polyfill to JavaScript WebGPU tensors
 *
 * This module provides Python-callable functions that execute tensor operations
 * on WebGPU, eliminating NumPy bottlenecks.
 */

import { WebGPUTensor } from '../compute/webgpu/webgpu-tensor.js';
import logger from '../utils/logger.js';

class WebGPUBridge {
  constructor() {
    this.computeEngine = null;
    this.bufferCache = new Map(); // Cache GPU buffers by ID
    this.nextBufferId = 1;
    this.stats = {
      operations: 0,
      gpuTime: 0,
      cpuToGpuTransfers: 0,
      gpuToCpuTransfers: 0
    };
  }

  /**
   * Set the compute engine reference
   */
  setComputeEngine(engine) {
    this.computeEngine = engine;
    logger.info('[WebGPU Bridge] Compute engine initialized');
  }

  /**
   * Allocate GPU buffer from Python data
   * @param {Array|Float32Array} data - Python array data
   * @param {Array} shape - Tensor shape
   * @param {string} dtype - Data type
   * @returns {number} Buffer ID for Python to reference
   */
  allocate(data, shape, dtype = 'float32') {
    console.log('[WebGPU Bridge] allocate() called with:', {
      dataType: typeof data,
      dataIsArray: Array.isArray(data),
      shape,
      dtype,
      computeEngine: !!this.computeEngine
    });

    if (!this.computeEngine) {
      const error = '[WebGPU Bridge] Compute engine not initialized';
      console.error(error);
      throw new Error(error);
    }

    const bufferId = this.nextBufferId++;

    // Convert Python data to Float32Array if needed
    let float32Data;
    if (data instanceof Float32Array) {
      float32Data = data;
    } else if (Array.isArray(data)) {
      float32Data = new Float32Array(data);
    } else {
      // Handle nested arrays (multidimensional)
      float32Data = new Float32Array(this._flattenArray(data));
    }

    console.log('[WebGPU Bridge] Creating tensor with:', { bufferId, shape, size: float32Data.length });

    // Create WebGPU tensor
    const tensor = new WebGPUTensor(float32Data, {
      shape: shape,
      dtype: dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    this.bufferCache.set(bufferId, tensor);
    this.stats.cpuToGpuTransfers++;

    logger.debug(`[WebGPU Bridge] Allocated buffer ${bufferId}, shape: [${shape}], size: ${float32Data.length}`);
    console.log('[WebGPU Bridge] Successfully allocated buffer ID:', bufferId);

    return bufferId;
  }

  /**
   * Read GPU buffer back to CPU for Python
   * @param {number} bufferId - Buffer ID to read
   * @returns {Float32Array} Data on CPU
   */
  async read(bufferId) {
    const tensor = this.bufferCache.get(bufferId);
    if (!tensor) {
      throw new Error(`[WebGPU Bridge] Buffer ${bufferId} not found`);
    }

    this.stats.gpuToCpuTransfers++;

    // WebGPUTensor stores data in .data property
    return tensor.data;
  }

  /**
   * Matrix multiplication: C = A @ B
   * @param {number} aId - Buffer ID for matrix A
   * @param {number} bId - Buffer ID for matrix B
   * @param {Array} aShape - Shape of A [M, K]
   * @param {Array} bShape - Shape of B [K, N]
   * @returns {number} Buffer ID for result C [M, N]
   */
  async matmul(aId, bId, aShape, bShape) {
    if (!this.computeEngine) {
      throw new Error('[WebGPU Bridge] Compute engine not initialized');
    }

    const tensorA = this.bufferCache.get(aId);
    const tensorB = this.bufferCache.get(bId);

    if (!tensorA || !tensorB) {
      throw new Error(`[WebGPU Bridge] Invalid buffer IDs: ${aId}, ${bId}`);
    }

    const startTime = performance.now();

    // Execute matmul on GPU
    const result = await this.computeEngine.execute('matmul', [tensorA.data, tensorB.data], {
      shape: aShape,
      otherShape: bShape,
      dtype: tensorA.dtype
    });

    const resultShape = [aShape[0], bShape[1]];

    // Create result tensor
    const resultTensor = new WebGPUTensor(result, {
      shape: resultShape,
      dtype: tensorA.dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    const resultId = this.nextBufferId++;
    this.bufferCache.set(resultId, resultTensor);

    this.stats.operations++;
    this.stats.gpuTime += performance.now() - startTime;

    logger.debug(`[WebGPU Bridge] matmul: [${aShape}] @ [${bShape}] = [${resultShape}], time: ${(performance.now() - startTime).toFixed(2)}ms`);

    return resultId;
  }

  /**
   * Element-wise addition: C = A + B
   */
  async add(aId, bId, shape) {
    const tensorA = this.bufferCache.get(aId);
    const tensorB = this.bufferCache.get(bId);

    if (!tensorA || !tensorB) {
      throw new Error(`[WebGPU Bridge] Invalid buffer IDs: ${aId}, ${bId}`);
    }

    const result = await this.computeEngine.execute('add', [tensorA.data, tensorB.data], {
      shape: shape,
      dtype: tensorA.dtype
    });

    const resultTensor = new WebGPUTensor(result, {
      shape: shape,
      dtype: tensorA.dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    const resultId = this.nextBufferId++;
    this.bufferCache.set(resultId, resultTensor);
    this.stats.operations++;

    return resultId;
  }

  /**
   * Element-wise multiplication: C = A * B
   */
  async multiply(aId, bId, shape) {
    const tensorA = this.bufferCache.get(aId);
    const tensorB = this.bufferCache.get(bId);

    if (!tensorA || !tensorB) {
      throw new Error(`[WebGPU Bridge] Invalid buffer IDs: ${aId}, ${bId}`);
    }

    const result = await this.computeEngine.execute('multiply', [tensorA.data, tensorB.data], {
      shape: shape,
      dtype: tensorA.dtype
    });

    const resultTensor = new WebGPUTensor(result, {
      shape: shape,
      dtype: tensorA.dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    const resultId = this.nextBufferId++;
    this.bufferCache.set(resultId, resultTensor);
    this.stats.operations++;

    return resultId;
  }

  /**
   * ReLU activation: C = max(0, A)
   */
  async relu(aId, shape) {
    const tensorA = this.bufferCache.get(aId);

    if (!tensorA) {
      throw new Error(`[WebGPU Bridge] Invalid buffer ID: ${aId}`);
    }

    const result = await this.computeEngine.execute('relu', [tensorA.data], {
      shape: shape,
      dtype: tensorA.dtype
    });

    const resultTensor = new WebGPUTensor(result, {
      shape: shape,
      dtype: tensorA.dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    const resultId = this.nextBufferId++;
    this.bufferCache.set(resultId, resultTensor);
    this.stats.operations++;

    return resultId;
  }

  /**
   * Transpose: C = A.T
   */
  async transpose(aId, shape) {
    const tensorA = this.bufferCache.get(aId);

    if (!tensorA) {
      throw new Error(`[WebGPU Bridge] Invalid buffer ID: ${aId}`);
    }

    const result = await this.computeEngine.execute('transpose', [tensorA.data], {
      shape: shape,
      dtype: tensorA.dtype
    });

    const resultShape = [shape[1], shape[0]];
    const resultTensor = new WebGPUTensor(result, {
      shape: resultShape,
      dtype: tensorA.dtype,
      device: 'webgpu',
      computeEngine: this.computeEngine
    });

    const resultId = this.nextBufferId++;
    this.bufferCache.set(resultId, resultTensor);
    this.stats.operations++;

    return resultId;
  }

  /**
   * Free GPU buffer
   */
  free(bufferId) {
    if (this.bufferCache.has(bufferId)) {
      this.bufferCache.delete(bufferId);
      logger.debug(`[WebGPU Bridge] Freed buffer ${bufferId}`);
      return true;
    }
    return false;
  }

  /**
   * Get performance statistics
   */
  getStats() {
    return {
      ...this.stats,
      cachedBuffers: this.bufferCache.size,
      avgGpuTime: this.stats.operations > 0 ? this.stats.gpuTime / this.stats.operations : 0
    };
  }

  /**
   * Clear all cached buffers
   */
  clearCache() {
    this.bufferCache.clear();
    logger.info('[WebGPU Bridge] Cache cleared');
  }

  /**
   * Helper: Flatten nested arrays
   */
  _flattenArray(arr) {
    const result = [];
    const flatten = (a) => {
      for (const item of a) {
        if (Array.isArray(item)) {
          flatten(item);
        } else {
          result.push(item);
        }
      }
    };
    flatten(arr);
    return result;
  }
}

// Singleton instance
export const webgpuBridge = new WebGPUBridge();
export default webgpuBridge;
