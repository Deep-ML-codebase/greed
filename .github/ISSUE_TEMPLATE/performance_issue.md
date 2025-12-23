---
name: Performance Issue
about: Report slow performance or request optimization
title: '[PERF] '
labels: performance
assignees: ''
---

## Performance Issue Description

A clear description of the performance problem.

## Benchmark Results

### Current Performance

```
Operation: [e.g., matrix multiplication 1000x1000]
CPU Time: XXXms
GPU Time: XXXms (if applicable)
Expected Time: XXXms
```

### Comparison

How does this compare to:
- [ ] PyTorch (CPU): XXXms
- [ ] PyTorch (CUDA): XXXms
- [ ] Other browser ML libraries: XXXms

## Code Sample

```python
# Minimal reproducible code that demonstrates the performance issue
import torch
import time

# Your benchmark code
```

## Environment

- **Browser:** [e.g., Chrome 120]
- **OS:** [e.g., macOS 14.0]
- **GPU:** [e.g., Apple M2, NVIDIA RTX 4090]
- **Greed.js Version:** [e.g., 3.1.0]
- **WebGPU Status:** [Enabled/Disabled]

## Profiling Data

If available, include profiling information:
- Browser DevTools performance profile
- WebGPU shader execution times
- Memory usage patterns

## Expected Performance

What performance would you expect for this operation?

## Proposed Optimization

If you have ideas for optimization:
- [ ] Better WebGPU shader implementation
- [ ] Memory access pattern improvements
- [ ] Batching strategy changes
- [ ] Caching opportunities
- [ ] Other: [describe]

## Additional Context

- Does performance degrade with specific tensor sizes?
- Does it happen only with certain operations?
- Any patterns you've noticed?
