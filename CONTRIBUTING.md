# Contributing to Greed.js

Thank you for considering contributing to Greed.js! We welcome contributions from the community to help make browser-based machine learning faster and more accessible.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (browser, OS, GPU)
- Code sample demonstrating the issue
- Console errors/warnings

Use the **Bug Report** issue template.

### Suggesting Features

We love new ideas! When suggesting features:

- Check if the feature already exists in PyTorch
- Describe the use case and problem it solves
- Provide example API usage
- Consider performance and browser compatibility
- Note if you're willing to implement it

Use the **Feature Request** issue template.

### Reporting Performance Issues

Performance is critical for Greed.js. When reporting performance issues:

- Include benchmark results with timing
- Compare against PyTorch or other libraries
- Provide reproducible code sample
- Share GPU/browser specifications
- Suggest potential optimizations if possible

Use the **Performance Issue** issue template.

## Development Process

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/greed.git
   cd greed
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

4. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

5. **Test your changes**
   ```bash
   npm run build
   npm test
   npm run lint
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new WebGPU shader for convolution"
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### Commit Message Guidelines

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `perf:` Performance improvement
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `style:` Code style changes (formatting)
- `chore:` Build/tooling changes

Examples:
```
feat: add GPU-accelerated conv2d layer
fix: resolve memory leak in tensor cleanup
perf: optimize matmul shader with tiled approach
docs: update WebGPU compatibility table
```

### Coding Standards

#### JavaScript/TypeScript

- Use ES6+ features
- Prefer `const` over `let`, avoid `var`
- Use meaningful variable names
- Add JSDoc comments for public APIs
- Follow existing code style
- Run `npm run lint` before committing

```javascript
/**
 * Allocates a WebGPU buffer for tensor data
 * @param {Array} data - Tensor data as nested arrays
 * @param {Array<number>} shape - Tensor dimensions
 * @param {string} dtype - Data type (float32, int32, etc.)
 * @returns {number} Buffer ID
 */
function allocateBuffer(data, shape, dtype) {
    // Implementation
}
```

#### Python (PyTorch Runtime)

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Document complex logic
- Match PyTorch API conventions
- Test against official PyTorch behavior

```python
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Applies linear transformation: y = xW^T + b

    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (*, out_features)
    """
    # Implementation
```

#### WebGPU Shaders (WGSL)

- Optimize for GPU parallelism
- Use workgroup shared memory when beneficial
- Add comments explaining algorithm
- Include performance notes
- Test on multiple GPU architectures

```wgsl
// Tiled matrix multiplication with shared memory
// Optimized for 16x16 tile size
// Achieves ~30x speedup over naive implementation
@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(...) {
    // Implementation
}
```

### Testing

#### Required Tests

- Unit tests for new features
- Integration tests for PyTorch API compliance
- Browser tests for WebGPU functionality
- Performance benchmarks for optimizations

#### Running Tests

```bash
# All tests
npm test

# Specific test suites
npm run test:unit
npm run test:browser
npm run test:performance

# Watch mode during development
npm run test:watch
```

#### Writing Tests

```javascript
describe('WebGPU Matrix Multiplication', () => {
    it('should match PyTorch output for 100x100 matrices', async () => {
        const greed = new Greed({ enableWebGPU: true });
        await greed.initialize();

        const result = await greed.runPython(`
            import torch
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            result = torch.matmul(x, y)
            result.shape
        `);

        expect(result).toEqual([100, 100]);
    });
});
```

### WebGPU Shader Development

When adding new WebGPU shaders:

1. **Research the operation**
   - Study PyTorch's implementation
   - Review CUDA kernels if available
   - Check academic papers for algorithms

2. **Design the shader**
   - Choose workgroup size (typically 64, 256, or 512)
   - Plan memory access patterns
   - Consider using shared memory for performance

3. **Implement and benchmark**
   - Start with simple implementation
   - Add optimizations incrementally
   - Benchmark against CPU fallback
   - Test on different GPUs

4. **Document performance**
   - Add comments explaining the algorithm
   - Note expected speedup
   - Document any limitations

### Performance Optimization Guidelines

- **Measure first**: Always benchmark before optimizing
- **Profile**: Use browser DevTools to identify bottlenecks
- **GPU vs CPU**: Consider when GPU overhead outweighs benefits
- **Memory**: Minimize CPU-GPU transfers
- **Caching**: Reuse GPU buffers when possible
- **Batch operations**: Combine multiple operations when feasible

### Documentation

Update documentation for:

- New PyTorch operations
- API changes
- Performance improvements
- Browser compatibility changes
- Configuration options

Documentation locations:
- `README.md` - Main documentation
- Code comments - Implementation details
- Examples - Usage demonstrations
- API reference - Function signatures

## Pull Request Process

1. **Fill out PR template** completely
2. **Ensure all tests pass** in CI
3. **Request review** from maintainers
4. **Address feedback** promptly
5. **Squash commits** if requested
6. **Wait for approval** before merging

### PR Review Checklist

Reviewers will check:

- [ ] Code quality and style
- [ ] Test coverage
- [ ] Documentation updates
- [ ] Performance impact
- [ ] Browser compatibility
- [ ] PyTorch API compliance
- [ ] No breaking changes (or properly documented)

## Areas for Contribution

We especially welcome contributions in:

### High Priority

- **WebGPU Shaders**: Optimized implementations for tensor operations
- **PyTorch API Coverage**: Missing operations and modules
- **Performance**: Optimization opportunities
- **Browser Support**: Expanding WebGPU compatibility
- **Documentation**: Examples, tutorials, API docs

### WebGPU Shader Opportunities

Operations that would benefit from GPU acceleration:

- Convolution operations (conv1d, conv3d)
- Advanced pooling (adaptive pooling, fractional pooling)
- Normalization layers (layer norm, instance norm)
- Attention mechanisms
- Custom operations

### PyTorch API Gaps

Missing functionality:

- Additional optimizers (RMSprop, AdaGrad, etc.)
- Data augmentation transforms
- Learning rate schedulers
- Advanced loss functions
- Distributed training utilities

### Testing Infrastructure

- Browser compatibility tests
- Cross-platform WebGPU tests
- Performance regression detection
- Memory leak detection
- Visual regression tests

## Getting Help

- **Questions**: Open a discussion on GitHub Discussions
- **Bugs**: Create an issue with bug report template
- **Features**: Create an issue with feature request template
- **Chat**: Join our community chat (link TBD)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Mentioned in project documentation

Significant contributions may earn:
- Collaborator access
- Recognition in project README
- Speaking opportunities at events

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (AGPL v3.0 for open source, with commercial licensing available).

---

Thank you for making Greed.js better! Every contribution, no matter how small, helps advance browser-based machine learning.
