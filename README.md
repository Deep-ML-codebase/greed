![logo](greed.png)

# Greed.js v3.1

[![npm version](https://badge.fury.io/js/greed.js.svg)](https://badge.fury.io/js/greed.js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/adityakhalkar/greed/workflows/CI/badge.svg)](https://github.com/adityakhalkar/greed/actions)

A high-performance JavaScript library enabling seamless execution of Python code in web browsers with WebGPU-accelerated PyTorch support using compute shaders and intelligent worker-based parallel execution.

## What's New in v3.1

### GPU Acceleration Performance
- **Optimized WebGPU matmul shader**: Tiled matrix multiplication with shared memory
- **Cached transpose optimization**: Eliminates redundant GPU uploads in neural network layers
- **Intelligent device management**: Automatic GPU buffer allocation and lifecycle management
- **Performance gains**: Up to 8.8x speedup on browser-appropriate workloads

### Modular Architecture Rewrite
- **Component-based design**: Separated concerns across RuntimeManager, ComputeStrategy, MemoryManager, and SecurityValidator
- **EventEmitter system**: Clean inter-component communication
- **Better maintainability**: Each module has a single responsibility
- **Improved testability**: Components can be tested in isolation

### Notebook-Style State Persistence
- **Variables persist between cells**: Define `a = 1` in one cell, use it in the next
- **Session-based execution**: Python globals maintained across multiple `run()` calls
- **Explicit cleanup API**: New `clearState()` method for manual state reset
- **Smart memory management**: Cleanup on errors, preserve on success

### Enhanced Security & Stability
- **Comprehensive input validation**: Advanced threat detection system
- **Graceful error recovery**: Automatic state cleanup after errors
- **Production-ready**: Extensive testing and validation

### Better Developer Experience
- **Dual API**: Use `run()` or `runPython()` - both work identically
- **Comprehensive events**: Monitor initialization, operations, errors, and cleanup
- **Detailed statistics**: Memory usage, operation count, performance metrics
- **Better error messages**: Clear, actionable error information

## Installation

```bash
npm install greed.js
```

```bash
yarn add greed.js
```

```html
<!-- CDN -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<script src="https://unpkg.com/greed.js@3.1.0/dist/greed.min.js"></script>
```

## Features

- **Modular Architecture**: Clean separation of concerns with EventEmitter-based communication
- **Notebook-Style Execution**: Variables persist between cells like Jupyter notebooks
- **PyTorch in Browser**: Full PyTorch polyfill with neural networks, tensors, and deep learning operations
- **WebGPU Compute Shaders**: True GPU acceleration with 50+ optimized WGSL compute shaders for tensor operations
- **Intelligent Fallback**: WebGPU → CPU → Worker execution strategy with automatic optimization
- **Complete Neural Networks**: Support for `torch.nn.Module`, layers, loss functions, and training
- **Python in Browser**: Execute Python code directly using Pyodide/WebAssembly
- **Enhanced Security**: Advanced input validation and threat detection system
- **Smart Compute Strategy**: Intelligent fallback between WebGPU → CPU → Worker execution
- **Memory Management**: Automatic resource cleanup and memory pressure monitoring
- **Dynamic Package Installation**: Automatically install Python packages on-demand
- **Simple API**: Easy-to-use interface with comprehensive PyTorch compatibility
- **Production Ready**: Comprehensive testing, security validation, and performance optimization

## Quick Start

### Basic Usage

```html
<!DOCTYPE html>
<html>
<head>
    <title>Greed.js v3.1 Demo</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
    <script type="module">
        import Greed from 'https://unpkg.com/greed.js@3.1.0/dist/greed.js';

        async function main() {
            // Initialize Greed.js
            const greed = new Greed({ enableWebGPU: true });
            await greed.initialize();

            // Cell 1: Define variables
            await greed.run(`
                import torch
                a = 5
                b = 10
                print(f"Defined: a={a}, b={b}")
            `);

            // Cell 2: Use variables from previous cell (like Jupyter!)
            await greed.run(`
                c = a + b
                print(f"Result: {a} + {b} = {c}")
            `);

            // WebGPU-accelerated tensor operations
            const result = await greed.run(`
                # Tensors automatically use WebGPU when available
                x = torch.randn(1000, 1000, device='webgpu')
                y = torch.randn(1000, 1000, device='webgpu')

                # GPU-accelerated matrix multiplication
                result = torch.matmul(x, y)
                print(f"Matrix result shape: {result.shape}")
                print(f"Device: {result.device}")

                result.mean().item()
            `);

            console.log('Result:', result);

            // Clear state when needed
            await greed.clearState();
        }

        main();
    </script>
</head>
<body>
    <h1>Greed.js v3.1 - Notebook-Style Execution</h1>
</body>
</html>
```

## Architecture v3.1

Greed.js v3.1 features a modular architecture designed for performance, maintainability, and extensibility:

```
┌─────────────────────────────────────────────────────────┐
│                      Greed Core                         │
│  (Orchestration, Events, Public API)                    │
└────────┬────────────────────────────────────┬───────────┘
         │                                    │
    ┌────▼──────────┐                   ┌────▼──────────┐
    │ RuntimeManager│                   │ComputeStrategy│
    │  - Pyodide    │                   │  - WebGPU     │
    │  - Packages   │                   │  - CPU        │
    │  - Execution  │                   │  - Workers    │
    └────┬──────────┘                   └────┬──────────┘
         │                                    │
    ┌────▼──────────┐                   ┌────▼──────────┐
    │MemoryManager  │                   │SecurityValida-│
    │  - GC         │                   │tor            │
    │  - Monitoring │                   │  - Validation │
    │  - Cleanup    │                   │  - Threat Det.│
    └───────────────┘                   └───────────────┘
```

### Core Components

- **`Greed`**: Main orchestrator with EventEmitter-based communication
- **`RuntimeManager`**: Pyodide initialization, package management, Python execution
- **`ComputeStrategy`**: WebGPU/CPU/Worker compute orchestration with intelligent fallback
- **`WebGPUComputeEngine`**: Hardware-accelerated tensor operations using WebGPU compute shaders
- **`WebGPUTensor`**: PyTorch-compatible tensor implementation with GPU acceleration
- **`TensorBridge`**: Seamless interoperability between JavaScript and Python tensors
- **`MemoryManager`**: Advanced resource cleanup with automatic garbage collection
- **`SecurityValidator`**: Comprehensive input validation and threat detection
- **`EventEmitter`**: Base class providing event-driven inter-component communication

## Notebook-Style Execution

v3.1 introduces true notebook-style execution where Python variables persist between cells:

```javascript
const greed = new Greed();
await greed.initialize();

// Cell 1: Define data
await greed.run(`
    import torch
    import torch.nn as nn

    # Define model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleNet()
    print("Model created")
`);

// Cell 2: Use model from previous cell
await greed.run(`
    # Model is still available!
    x = torch.randn(32, 10)
    output = model(x)
    print(f"Output shape: {output.shape}")
`);

// Cell 3: Continue training
await greed.run(`
    optimizer = torch.optim.Adam(model.parameters())
    loss = output.mean()
    loss.backward()
    optimizer.step()
    print("Training step complete")
`);

// Clear state when starting new session
await greed.clearState();
```

## API Reference

### Constructor

```javascript
const greed = new Greed({
    // Core settings
    enableWebGPU: true,           // Enable WebGPU acceleration
    enableWorkers: true,           // Enable Web Workers
    maxWorkers: 4,                 // Number of worker threads

    // Security settings
    strictSecurity: true,          // Strict security validation
    allowEval: false,              // Block eval() in Python
    allowFileSystem: false,        // Block file system access
    allowNetwork: false,           // Block network access

    // Performance settings
    maxMemoryMB: 1024,            // Max memory allocation
    gcThreshold: 0.8,             // GC trigger threshold
    enableProfiling: true,        // Performance profiling

    // Runtime settings
    pyodideIndexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
    preloadPackages: ['numpy'],   // Packages to preload
    initTimeout: 30000            // Initialization timeout
});
```

### Main Methods

#### `await greed.initialize()`
Initialize all components and establish PyTorch API.

```javascript
await greed.initialize();
```

#### `await greed.run(code, options)` or `await greed.runPython(code, options)`
Execute Python code with notebook-style state persistence.

```javascript
const result = await greed.run(`
    import torch
    x = torch.tensor([1, 2, 3])
    x.sum().item()
`, {
    captureOutput: true,     // Capture print() output
    timeout: 5000,           // Execution timeout
    globals: {},             // Additional globals
    allowWarnings: false,    // Allow security warnings
    bypassSecurity: false    // Bypass security validation
});

console.log(result.output); // Printed output
```

#### `await greed.clearState()`
Clear Python execution state (user variables). Preserves torch, numpy, and library imports.

```javascript
await greed.clearState();
```

#### `await greed.loadPackages(packages)`
Load additional Python packages.

```javascript
await greed.loadPackages(['pandas', 'matplotlib']);
```

#### `greed.getStats()`
Get comprehensive system statistics.

```javascript
const stats = greed.getStats();
console.log('Memory usage:', stats.memory.memoryUsageMB);
console.log('Operations:', stats.operations);
console.log('Runtime status:', stats.runtime);
```

#### `await greed.destroy()`
Graceful shutdown and resource cleanup.

```javascript
await greed.destroy();
```

### Event System

```javascript
greed.on('init:complete', (data) => {
    console.log('Initialization complete:', data.initTime, 'ms');
});

greed.on('operation:start', (data) => {
    console.log('Executing code:', data.codeLength, 'bytes');
});

greed.on('operation:complete', (data) => {
    console.log('Execution time:', data.executionTime, 'ms');
});

greed.on('operation:error', (data) => {
    console.error('Execution error:', data.error);
});

greed.on('memory:warning', (data) => {
    console.warn('Memory pressure:', data.memoryUsageMB, 'MB');
});
```

## PyTorch Support

### Tensor Operations
```python
import torch

# Tensor creation
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
y = torch.randn(2, 2)

# GPU acceleration
x_gpu = x.cuda()  # Move to WebGPU
result = torch.mm(x_gpu, y.cuda())  # Matrix multiplication on GPU

# All standard operations supported
z = x + y * 2.0 - torch.ones_like(x)
```

### Neural Networks
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create and use model
model = SimpleNet()
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)

# Training with loss functions
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (32,))
loss = criterion(output, target)
```

### GPU Acceleration Features
- **Element-wise operations**: `+`, `-`, `*`, `/` with smart GPU thresholds
- **Matrix operations**: `torch.mm()`, `torch.matmul()`, `@` operator
- **Reduction operations**: `torch.sum()`, `torch.mean()`, `torch.max()`
- **Neural network layers**: `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`
- **Automatic fallback**: Seamless CPU fallback for small tensors or when WebGPU unavailable

## Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| **Pyodide/WebAssembly** | 57+ | 52+ | 11+ | 16+ |
| **WebGPU Acceleration** | 113+ | Experimental | Experimental | 113+ |
| **Web Workers** | Yes | Yes | Yes | Yes |
| **Notebook State Persistence** | Yes | Yes | Yes | Yes |

## Framework Integration

### React

```jsx
import { useState, useEffect } from 'react';
import Greed from 'greed.js';

function PyTorchNotebook() {
  const [greed, setGreed] = useState(null);
  const [output, setOutput] = useState('');

  useEffect(() => {
    const init = async () => {
      const instance = new Greed({ enableWebGPU: true });
      await instance.initialize();
      setGreed(instance);
    };
    init();
    return () => greed?.destroy();
  }, []);

  const runCell = async (code) => {
    if (!greed) return;
    const result = await greed.run(code);
    setOutput(result.output);
  };

  return (
    <div>
      <button onClick={() => runCell('a = 5; print(a)')}>
        Cell 1: Define a
      </button>
      <button onClick={() => runCell('print(a * 2)')}>
        Cell 2: Use a
      </button>
      <button onClick={() => greed.clearState()}>
        Clear State
      </button>
      <pre>{output}</pre>
    </div>
  );
}
```

### Next.js

```jsx
import dynamic from 'next/dynamic';

// Disable SSR for Greed.js
const PyTorchRunner = dynamic(() => import('../components/PyTorchRunner'), {
  ssr: false,
  loading: () => <p>Loading PyTorch...</p>
});

export default function HomePage() {
  return <PyTorchRunner />;
}
```

## Development

```bash
# Clone repository
git clone https://github.com/adityakhalkar/greed.git
cd greed

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run test suite
npm test
```

## Project Structure

```
greed/
├── src/
│   ├── core/
│   │   ├── greed-v2.js          # Main orchestrator
│   │   ├── runtime-manager.js    # Pyodide runtime
│   │   └── event-emitter.js      # Event system
│   ├── compute/
│   │   ├── compute-strategy.js   # Compute orchestration
│   │   └── webgpu/               # WebGPU implementation
│   ├── utils/
│   │   ├── memory-manager.js     # Memory management
│   │   └── security-validator.js # Security validation
│   └── polyfills/
│       └── pytorch-runtime.js    # PyTorch polyfill
├── dist/                         # Built files
├── tests/                        # Test suite
└── examples/                     # Usage examples
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. **Bug Reports**: Use GitHub Issues with detailed reproduction steps
2. **Feature Requests**: Propose new PyTorch operations or WebGPU optimizations
3. **Pull Requests**: Include tests and ensure all examples still work

## License

This software is dual-licensed under AGPL v3.0 and commercial licenses.

### Open Source License (AGPL v3.0)
- Free for open source projects and personal use
- Requires your application to be open-sourced under AGPL v3.0
- Suitable for academic research and community contributions
- Must make complete source code available to users

### Commercial License
- Permits use in proprietary commercial applications
- Allows keeping application source code confidential
- No AGPL obligations for end users
- Includes technical support and maintenance services

For commercial licensing inquiries, contact khalkaraditya8@gmail.com

Complete licensing terms are available in the [LICENSE](LICENSE) file.

## Acknowledgments

- **[Pyodide](https://pyodide.org/)**: Python-to-WebAssembly runtime
- **[WebGPU](https://gpuweb.github.io/gpuweb/)**: GPU acceleration standard
- **[PyTorch](https://pytorch.org/)**: Deep learning framework inspiration
- **Python Community**: For the incredible ecosystem

---

**Greed.js v3.1** - Bringing the power of PyTorch, GPU acceleration, and notebook-style execution to every web browser.
