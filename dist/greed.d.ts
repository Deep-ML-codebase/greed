/**
 * Greed.js - PyTorch in Browser with WebGPU Acceleration
 * TypeScript definitions for v3.x
 */

// ============================================================================
// Core Types
// ============================================================================

export interface GreedConfig {
  // Core settings
  enableWebGPU?: boolean;
  enableWorkers?: boolean;
  maxWorkers?: number;

  // Security settings
  strictSecurity?: boolean;
  allowEval?: boolean;
  allowFileSystem?: boolean;
  allowNetwork?: boolean;

  // Performance settings
  maxMemoryMB?: number;
  gcThreshold?: number;
  enableProfiling?: boolean;

  // Runtime settings
  pyodideIndexURL?: string;
  preloadPackages?: string[];
  initTimeout?: number;

  // Extension mode (for browser extensions)
  extensionMode?: boolean;
  customPyodideLoader?: () => Promise<any>;
  workerScriptPath?: string;
}

export interface GreedStats {
  initTime: number;
  operations: number;
  totalExecutionTime: number;
  averageExecutionTime: number;
  memoryUsage: number;
  startTime: number;
}

export interface RunOptions {
  globals?: Record<string, any>;
  timeout?: number;
  returnResult?: boolean;
  captureOutput?: boolean;
}

export interface TensorShape extends Array<number> {}

export type DType = 'float32' | 'float64' | 'int32' | 'int16' | 'int8' | 'uint8' | 'bool';

export type Device = 'cpu' | 'cuda' | 'webgpu';

// ============================================================================
// Main Greed Class
// ============================================================================

export default class Greed {
  constructor(options?: GreedConfig);

  // Initialization
  initialize(): Promise<void>;
  isInitialized: boolean;

  // Python execution
  run(code: string, options?: RunOptions): Promise<any>;
  runPython(code: string, options?: RunOptions): Promise<any>;

  // State management
  clearState(): Promise<void>;
  reset(): Promise<void>;

  // Package management
  loadPackages(packages: string[]): Promise<void>;

  // Statistics
  getStats(): GreedStats;

  // Tensor operations
  tensor(operation: string, tensors: any[], options?: any): Promise<any>;

  // Cleanup
  destroy(): Promise<void>;

  // Event emitter methods
  on(event: string, listener: (...args: any[]) => void): this;
  once(event: string, listener: (...args: any[]) => void): this;
  off(event: string, listener: (...args: any[]) => void): this;
  emit(event: string, ...args: any[]): boolean;

  // Extension support
  static createForExtension(config: GreedConfig): Greed;
}

// ============================================================================
// WebGPU Tensor API
// ============================================================================

export class WebGPUTensor {
  constructor(data: ArrayLike<number> | ArrayBuffer, shape?: TensorShape, dtype?: DType, device?: Device);

  // Properties
  data: Float32Array | Int32Array | Uint8Array;
  shape: TensorShape;
  dtype: DType;
  device: Device;
  length: number;

  // Array-like methods
  get(index: number): number;
  set(index: number, value: number): void;
  slice(start?: number, end?: number): WebGPUTensor;
  toArray(): number[];
  toString(): string;

  // Tensor operations
  add(other: WebGPUTensor | number): WebGPUTensor;
  sub(other: WebGPUTensor | number): WebGPUTensor;
  mul(other: WebGPUTensor | number): WebGPUTensor;
  div(other: WebGPUTensor | number): WebGPUTensor;
  matmul(other: WebGPUTensor): WebGPUTensor;

  // Activation functions
  relu(): WebGPUTensor;
  sigmoid(): WebGPUTensor;
  tanh(): WebGPUTensor;
  softmax(dim?: number): WebGPUTensor;

  // Reductions
  sum(dim?: number): WebGPUTensor | number;
  mean(dim?: number): WebGPUTensor | number;
  max(dim?: number): WebGPUTensor | number;
  min(dim?: number): WebGPUTensor | number;

  // Shape operations
  reshape(shape: TensorShape): WebGPUTensor;
  transpose(dim0?: number, dim1?: number): WebGPUTensor;
  squeeze(dim?: number): WebGPUTensor;
  unsqueeze(dim: number): WebGPUTensor;

  // Device management
  to(device: Device): WebGPUTensor;
  cpu(): WebGPUTensor;
  cuda(): WebGPUTensor;
  webgpu(): WebGPUTensor;

  // Gradient support
  backward(gradient?: WebGPUTensor): void;
  requires_grad: boolean;
  grad: WebGPUTensor | null;
}

// ============================================================================
// PyTorch-like API (available via greed.run())
// ============================================================================

export interface TorchModule {
  // Tensor creation
  tensor(data: ArrayLike<number>, options?: { dtype?: DType; device?: Device; requires_grad?: boolean }): any;
  zeros(shape: TensorShape, options?: { dtype?: DType; device?: Device }): any;
  ones(shape: TensorShape, options?: { dtype?: DType; device?: Device }): any;
  randn(shape: TensorShape, options?: { dtype?: DType; device?: Device }): any;
  rand(shape: TensorShape, options?: { dtype?: DType; device?: Device }): any;

  // Operations
  add(a: any, b: any): any;
  sub(a: any, b: any): any;
  mul(a: any, b: any): any;
  div(a: any, b: any): any;
  matmul(a: any, b: any): any;

  // Neural network module
  nn: {
    Module: any;
    Linear: any;
    Conv2d: any;
    ReLU: any;
    Sigmoid: any;
    Tanh: any;
    Softmax: any;
    Dropout: any;
    BatchNorm1d: any;
    BatchNorm2d: any;
    Sequential: any;
  };

  // Optimizers
  optim: {
    SGD: any;
    Adam: any;
    AdamW: any;
  };

  // Functional API
  functional: any;
}

// ============================================================================
// Security Validator
// ============================================================================

export class SecurityValidator {
  constructor(config?: any);

  validatePythonCode(code: string): { allowed: boolean; riskLevel: string; warnings: string[] };
  validateTensorData(tensors: any, options?: any): void;
}

// ============================================================================
// Memory Manager
// ============================================================================

export class MemoryManager {
  constructor(config?: any);

  register(resource: any, cleanup: () => void, options?: any): string;
  unregister(id: string): void;
  forceGC(): Promise<void>;
  getStats(): { totalMemory: number; usedMemory: number; memoryUtilization: number };
}

// ============================================================================
// Event Emitter
// ============================================================================

export class EventEmitter {
  on(event: string, listener: (...args: any[]) => void): this;
  once(event: string, listener: (...args: any[]) => void): this;
  off(event: string, listener: (...args: any[]) => void): this;
  emit(event: string, ...args: any[]): boolean;
  removeAllListeners(event?: string): this;
  listenerCount(event: string): number;
}

// ============================================================================
// Utility Types
// ============================================================================

export interface PyodideProxy {
  toJs(): any;
  destroy(): void;
}

export interface ComputeResult {
  success: boolean;
  data?: any;
  error?: string;
  executionTime?: number;
}

// ============================================================================
// Events
// ============================================================================

export type GreedEvent =
  | 'init:start'
  | 'init:complete'
  | 'init:error'
  | 'operation:start'
  | 'operation:complete'
  | 'operation:error'
  | 'memory:warning'
  | 'memory:critical'
  | 'gc:start'
  | 'gc:complete'
  | 'components:init:start'
  | 'components:init:complete'
  | 'pytorch:setup:start'
  | 'pytorch:setup:complete'
  | 'validation:start'
  | 'validation:complete';

// ============================================================================
// Module Exports
// ============================================================================

export { Greed };
export { WebGPUTensor };
export { SecurityValidator };
export { MemoryManager };
export { EventEmitter };
