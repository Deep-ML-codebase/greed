# Greed.js v3.0 - Getting Started

WebGPU-accelerated PyTorch runtime for browsers with notebook-style execution.

## 🚀 Quick Install

### CDN (Recommended)
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
    <script type="module">
        import Greed from 'https://unpkg.com/greed.js@3.0.0/dist/greed.js';

        async function main() {
            const greed = new Greed({ enableWebGPU: true });
            await greed.initialize();

            // Cell 1: Define variables (like Jupyter!)
            await greed.run(`
                import torch
                a = torch.tensor([1, 2, 3, 4, 5])
                print(f"Created tensor: {a}")
            `);

            // Cell 2: Variables persist between cells
            await greed.run(`
                b = a * 2 + 1
                print(f"Result: {b}")
            `);
        }
        main();
    </script>
</head>
<body>
    <h1>Greed.js v3.0 - Notebook-Style Execution</h1>
</body>
</html>
```

### NPM
```bash
npm install greed.js
```

```javascript
import Greed from 'greed.js';

const greed = new Greed({ enableWebGPU: true });
await greed.initialize();

// Variables persist between calls (notebook-style)
await greed.run(`
    import torch
    model = torch.nn.Linear(10, 5)
    print("Model created")
`);

await greed.run(`
    # model is still available!
    x = torch.randn(32, 10)
    output = model(x)
    print(f"Output shape: {output.shape}")
`);

// Clear state when needed
await greed.clearState();
```

## 🎉 What's New in v3.0

### 📔 Notebook-Style Execution
Variables now persist between `run()` calls, just like Jupyter notebooks:

```javascript
const greed = new Greed();
await greed.initialize();

// Cell 1
await greed.run('a = 5');

// Cell 2 - 'a' is still available!
await greed.run('print(a * 2)');  // Outputs: 10
```

### 🔧 New APIs
- **`runPython()`**: Alias for `run()` - use whichever you prefer
- **`clearState()`**: Manually clear Python variables when needed
- **Better error recovery**: Automatic cleanup on errors, preserve state on success

### 🏗️ Modular Architecture
- Separated RuntimeManager, ComputeStrategy, MemoryManager, SecurityValidator
- EventEmitter-based communication
- Better performance and maintainability

## 🎯 Basic Configuration

```javascript
const greed = new Greed({
    // WebGPU settings
    enableWebGPU: true,
    webgpu: {
        webgpuMinElements: 1,  // Use GPU for all operations
        enableProfiling: true
    },

    // Security settings
    security: {
        strictMode: false  // Allow broader PyTorch operations
    },

    // Performance settings
    maxMemoryMB: 1024,
    gcThreshold: 0.8
});
```

## 📔 Notebook-Style Examples

### Example 1: Multi-Cell Training

```javascript
const greed = new Greed();
await greed.initialize();

// Cell 1: Setup data
await greed.run(`
    import torch
    import torch.nn as nn

    # Generate synthetic data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    print(f"Data created: X={X.shape}, y={y.shape}")
`);

// Cell 2: Define model (uses data from Cell 1)
await greed.run(`
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    print("Model defined")
`);

// Cell 3: Train (uses model and data from previous cells)
await greed.run(`
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
`);

// Clear state when done
await greed.clearState();
```

### Example 2: Interactive Exploration

```javascript
const greed = new Greed();
await greed.initialize();

// Define some tensors
await greed.run(`
    import torch
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    print(f"x = {x}")
`);

// Try different operations interactively
await greed.run(`print(f"Mean: {x.mean()}")`);
await greed.run(`print(f"Sum: {x.sum()}")`);
await greed.run(`print(f"Max: {x.max()}")`);

// Modify and continue
await greed.run(`
    y = x * 2
    print(f"y = {y}")
`);
```

## 🔥 WebGPU Setup

### Chrome/Edge/Brave
1. Go to: `chrome://flags/#enable-unsafe-webgpu` (or `brave://flags`)
2. Set: **Enabled**
3. Restart browser

### Check WebGPU Status

```javascript
const greed = new Greed({ enableWebGPU: true });
await greed.initialize();

const stats = greed.getStats();
console.log('WebGPU available:', stats.compute.availableStrategies.includes('webgpu'));
console.log('Compute strategies:', stats.compute.availableStrategies);
```

## 🎨 Complete Example: Neural Network

```javascript
import Greed from 'greed.js';

async function trainModel() {
    const greed = new Greed({ enableWebGPU: true });
    await greed.initialize();

    // Cell 1: Import and setup
    await greed.run(`
        import torch
        import torch.nn as nn
        import torch.optim as optim

        print("Libraries loaded")
    `);

    // Cell 2: Create dataset
    await greed.run(`
        # Generate synthetic dataset
        X_train = torch.randn(1000, 20)
        y_train = (X_train.sum(dim=1) > 0).long()

        X_test = torch.randn(200, 20)
        y_test = (X_test.sum(dim=1) > 0).long()

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
    `);

    // Cell 3: Define model
    await greed.run(`
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)

        model = Net()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print("Model architecture created")
    `);

    // Cell 4: Training loop
    const result = await greed.run(`
        epochs = 10
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == y_train).float().mean()

            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

        # Test accuracy
        with torch.no_grad():
            test_outputs = model(X_test)
            test_predictions = test_outputs.argmax(dim=1)
            test_accuracy = (test_predictions == y_test).float().mean()

        print(f"\\nTest Accuracy: {test_accuracy:.4f}")
        test_accuracy.item()
    `);

    console.log('Final test accuracy:', result);

    // Get system statistics
    const stats = greed.getStats();
    console.log('Total operations:', stats.operations);
    console.log('Memory usage:', stats.memory.memoryUsageMB, 'MB');

    // Cleanup when done
    await greed.destroy();
}

trainModel();
```

## 🔧 API Methods

### `await greed.run(code, options)` or `await greed.runPython(code, options)`
Execute Python code with state persistence.

```javascript
await greed.run(`print("Hello")`, {
    captureOutput: true,    // Capture print() output
    timeout: 5000,          // Execution timeout (ms)
    allowWarnings: false,   // Allow security warnings
    bypassSecurity: false   // Bypass security checks
});
```

### `await greed.clearState()`
Clear Python variables. Preserves imports like torch, numpy.

```javascript
await greed.clearState();
```

### `await greed.loadPackages(packages)`
Load additional Python packages.

```javascript
await greed.loadPackages(['pandas', 'matplotlib']);
```

### `greed.getStats()`
Get system statistics.

```javascript
const stats = greed.getStats();
console.log(stats);
```

### Event Monitoring

```javascript
greed.on('init:complete', (data) => {
    console.log('Initialized in', data.initTime, 'ms');
});

greed.on('operation:complete', (data) => {
    console.log('Execution time:', data.executionTime, 'ms');
});

greed.on('memory:warning', (data) => {
    console.warn('Memory usage:', data.memoryUsageMB, 'MB');
});
```

## 🌐 Browser Support

| Browser | WebGPU | State Persistence | Status |
|---------|--------|-------------------|--------|
| Chrome 113+ | ✅ | ✅ | Full Support |
| Edge 113+ | ✅ | ✅ | Full Support |
| Brave | ✅ (with flags) | ✅ | Full Support |
| Firefox | ⚠️ Experimental | ✅ | Limited |
| Safari | ❌ | ✅ | CPU Only |

## 💡 Performance Tips

1. **Enable WebGPU**: Always use `enableWebGPU: true`
2. **Use float32**: Default dtype for GPU operations
3. **Batch operations**: Larger tensors = better GPU utilization
4. **Reuse variables**: Take advantage of state persistence
5. **Monitor memory**: Use `greed.getStats()` to track usage

## 🐛 Troubleshooting

### Variables not persisting between cells?
Make sure you're using v3.0 and calling `run()` or `runPython()`:

```javascript
// ✅ Correct - variables persist
await greed.run('a = 5');
await greed.run('print(a)');  // Works!

// ❌ Wrong - don't manually clear state after each cell
await greed.run('a = 5');
await greed.clearState();  // Don't do this!
await greed.run('print(a)');  // Error: a is not defined
```

### WebGPU not working?
- Enable browser flags: `chrome://flags/#enable-unsafe-webgpu`
- Check GPU support: `greed.getStats().compute.availableStrategies`
- Update graphics drivers
- Use HTTPS or localhost (required for WebGPU)

### Import errors?
- Ensure Pyodide loads before Greed.js
- Wait for `initialize()` to complete before running code

### Memory issues?
- Monitor with `greed.getStats().memory`
- Call `greed.clearState()` between sessions
- Reduce `maxMemoryMB` in config
- Use `greed.forceGC()` for manual garbage collection

## 📚 Next Steps

- Explore the [full README](README.md) for advanced features
- Check [examples/](examples/) for more use cases
- Read about [WebGPU implementation](README.md#webgpu-implementation)
- Learn about [security features](README.md#security)

---

**Greed.js v3.0** - PyTorch + WebGPU + Notebook-Style Execution in your browser! 🚀
