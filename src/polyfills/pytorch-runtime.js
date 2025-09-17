/**
 * PyTorch Runtime Polyfill - Extracted from main thread for better performance
 * Provides PyTorch-compatible API with WebGPU acceleration support
 */

/**
 * Initialize PyTorch polyfill in Python runtime
 */
export function createPyTorchPolyfill() {
  return `
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
    def __init__(self, data, device='cpu', dtype='float32', requires_grad=False, _force_webgpu=False):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        
        # Determine actual device based on tensor size and WebGPU availability
        self._original_device = device
        self._force_webgpu = _force_webgpu
        
        # TEMPORARILY DISABLED: Auto-detect WebGPU to prevent recursion
        # Auto-detect WebGPU usage for larger tensors or when forced
        # if (_force_webgpu or self._should_use_webgpu(data)) and device != 'cpu':
        #     self.device = WebGPUDevice('webgpu')
        # elif device == 'cuda' or device == 'gpu':
        #     # Map CUDA/GPU requests to WebGPU if available
        #     self.device = WebGPUDevice('webgpu')
        # else:
        #     self.device = WebGPUDevice(device) if isinstance(device, str) else device

        # Simple device assignment to prevent recursion
        self.device = device
        
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = None
        self.grad_fn = None
    
    def size(self, dim=None):
        """Return the size of the tensor or a specific dimension"""
        if dim is None:
            return self.shape
        else:
            if dim < 0:
                dim = self.ndim + dim
            if dim >= self.ndim or dim < 0:
                raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], but got {dim})")
            return self.shape[dim]
    
    def _should_use_webgpu(self, data):
        \"\"\"Determine if WebGPU should be used based on tensor characteristics\"\"\"
        try:
            # Use WebGPU for tensors with more than 1 element (very low threshold)
            if hasattr(data, 'size'):
                return data.size > 1
            elif hasattr(data, '__len__'):
                return len(data) > 1
            return False
        except:
            return False
        
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
        return WebGPUTensor(reshaped_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def transpose(self, dim0, dim1):
        transposed_data = np.swapaxes(self.data, dim0, dim1)
        return WebGPUTensor(transposed_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def sum(self, dim=None, keepdim=False):
        if dim is None:
            result_data = np.sum(self.data)
        else:
            result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            result_data = np.mean(self.data)
        else:
            result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def std(self, dim=None, keepdim=False, unbiased=True):
        """Compute standard deviation"""
        if dim is None:
            result_data = np.std(self.data, ddof=1 if unbiased else 0)
        else:
            result_data = np.std(self.data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def var(self, dim=None, keepdim=False, unbiased=True):
        """Compute variance"""
        if dim is None:
            result_data = np.var(self.data, ddof=1 if unbiased else 0)
        else:
            result_data = np.var(self.data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def to(self, device):
        new_device = WebGPUDevice(device) if isinstance(device, str) else device
        return WebGPUTensor(self.data.copy(), device=new_device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')
        
    def __repr__(self):
        return f"tensor({self.data}, device='{self.device}', dtype='{self.dtype}')"
    
    def __float__(self):
        """Convert single-element tensor to Python float"""
        if self.data.size == 1:
            return float(self.data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")
    
    def __int__(self):
        """Convert single-element tensor to Python int"""
        if self.data.size == 1:
            return int(self.data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")
    
    def __getitem__(self, key):\n        \"\"\"Support tensor indexing like tensor[indices]\"\"\"\n        if isinstance(key, WebGPUTensor):\n            # Convert WebGPUTensor indices to numpy array\n            indices = key.data.astype(int)\n            result_data = self.data[indices]\n        else:\n            result_data = self.data[key]\n        \n        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)\n    \n    # Arithmetic operators
    def __add__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data + other.data
        else:
            result_data = self.data + other

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __sub__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data - other.data
        else:
            result_data = self.data - other
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __mul__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data * other.data
        else:
            result_data = self.data * other

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __truediv__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data / other.data
        else:
            result_data = self.data / other
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def __pow__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = np.power(self.data, other.data)
        else:
            result_data = np.power(self.data, other)

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __radd__(self, other):
        result_data = other + self.data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __rmul__(self, other):
        result_data = other * self.data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __matmul__(self, other):
        \"\"\"Matrix multiplication operator (@)\"\"\"
        if isinstance(other, WebGPUTensor):
            if self.ndim == 2 and other.ndim == 2:
                result_data = np.dot(self.data, other.data)
            elif self.ndim == 1 and other.ndim == 2:
                result_data = np.dot(self.data, other.data)
            elif self.ndim == 2 and other.ndim == 1:
                result_data = np.dot(self.data, other.data)
            else:
                result_data = np.matmul(self.data, other.data)
        else:
            result_data = np.matmul(self.data, other)
        
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed
        
        return result
    
    def __rmatmul__(self, other):
        \"\"\"Reverse matrix multiplication\"\"\"
        result_data = np.matmul(other, self.data)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def retain_grad(self):
        \"\"\"Enable gradient retention for non-leaf tensor\"\"\"
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        self._retain_grad = True
        return self
    
    def zero_grad(self):
        \"\"\"Zero out the gradients\"\"\"
        if self.grad is not None:
            self.grad.data.fill(0)

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        \"\"\"Placeholder backward method - autograd disabled for stability\"\"\"
        if not self.requires_grad:
            return

        # For now, just initialize a dummy gradient to prevent errors
        # Real autograd implementation will be added later
        if self.grad is None:
            self.grad = WebGPUTensor(np.zeros_like(self.data), device="webgpu", dtype=self.dtype)

        print("Note: Autograd is currently disabled. Gradient computation skipped.")
    
    def _matmul_backward(self, grad_output, other):
        \"\"\"Backward pass for matrix multiplication\"\"\"
        if isinstance(other, WebGPUTensor):
            # d/da (a @ b) = grad_output @ b.T
            if self.grad is None:
                self.grad = WebGPUTensor(np.zeros_like(self.data), device="webgpu", dtype=self.dtype)
            self_grad = np.matmul(grad_output.data, other.data.T)
            self.grad.data += self_grad
            
            # d/db (a @ b) = a.T @ grad_output  
            if other.requires_grad:
                if other.grad is None:
                    other.grad = WebGPUTensor(np.zeros_like(other.data), device="webgpu", dtype=other.dtype)
                other_grad = np.matmul(self.data.T, grad_output.data)
                other.grad.data += other_grad

# Linear algebra operations module
class TorchLinalg:
    \"\"\"Linear algebra operations module\"\"\"
    
    def __init__(self):
        pass
    
    def det(self, input_tensor):
        \"\"\"Compute determinant\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.det() expects a 2D square tensor")
            det_value = np.linalg.det(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor([det_value], device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.det(input_tensor)
    
    def inv(self, input_tensor):
        \"\"\"Compute matrix inverse\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.inv() expects a 2D square tensor")
            inv_data = np.linalg.inv(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor(inv_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.inv(input_tensor)
    
    def norm(self, input_tensor, ord=None, dim=None, keepdim=False):
        \"\"\"Compute matrix or vector norm\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                norm_value = np.linalg.norm(input_tensor.data, ord=ord)
                return WebGPUTensor([norm_value], device="webgpu", dtype=input_tensor.dtype)
            else:
                norm_data = np.linalg.norm(input_tensor.data.reshape(input_tensor.shape), ord=ord, axis=dim, keepdims=keepdim)
                return WebGPUTensor(norm_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.norm(input_tensor, ord=ord, axis=dim, keepdims=keepdim)
    
    def eig(self, input_tensor):
        \"\"\"Compute eigenvalues and eigenvectors\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.eig() expects a 2D square tensor")
            eigenvalues, eigenvectors = np.linalg.eig(input_tensor.data.reshape(input_tensor.shape))
            return (
                WebGPUTensor(eigenvalues, device="webgpu", dtype=input_tensor.dtype),
                WebGPUTensor(eigenvectors, device="webgpu", dtype=input_tensor.dtype)
            )
        else:
            return np.linalg.eig(input_tensor)
    
    def svd(self, input_tensor, full_matrices=True):
        \"\"\"Compute singular value decomposition\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            U, S, Vh = np.linalg.svd(input_tensor.data.reshape(input_tensor.shape), full_matrices=full_matrices)
            return (
                WebGPUTensor(U, device="webgpu", dtype=input_tensor.dtype),
                WebGPUTensor(S, device="webgpu", dtype=input_tensor.dtype),
                WebGPUTensor(Vh, device="webgpu", dtype=input_tensor.dtype)
            )
        else:
            return np.linalg.svd(input_tensor, full_matrices=full_matrices)

# Neural network functional operations
class TorchNNFunctional:
    @staticmethod
    def relu(input_tensor):
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor.data, 0)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.maximum(input_tensor, 0)
    
    @staticmethod
    def sigmoid(input_tensor):
        if isinstance(input_tensor, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-input_tensor.data))
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return 1 / (1 + np.exp(-input_tensor))

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
            result = WebGPUTensor(np.dot(x.data, self.weight.data.T), device="webgpu", dtype=x.dtype)
            if self.bias is not None:
                result.data = result.data + self.bias.data
            return result
        else:
            raise TypeError("Input must be WebGPUTensor")

# Create torch module with essential functions
class TorchModule:
    def __init__(self):
        self.tensor = self._tensor
        self.zeros = self._zeros
        self.ones = self._ones
        self.randn = self._randn
        self.matmul = self._matmul
        self.sum = self._sum
        self.as_tensor = self._as_tensor
        self.arange = self._arange
        self.randperm = self._randperm
        self.nn = TorchNN()
        
        # Add Tensor class reference
        self.Tensor = WebGPUTensor
        
        # Linear algebra module
        self.linalg = TorchLinalg()
        
        # Activation functions
        self.relu = self._relu
        
        # Mathematical functions
        self.round = self.round
        
        # Data types
        self.float32 = 'float32'
        self.float64 = 'float64'
        self.double = 'float64'
        self.float = 'float32'
        self.int32 = 'int32'
        self.int64 = 'int64'
        self.long = 'int64'
        self.int = 'int32'
        self.bool = 'bool'
        self.uint8 = 'uint8'
        
        # Device types
        # self.device = self._device  # DISABLED: Potential recursion source
        
    def _tensor(self, data, **kwargs):
        # Enable WebGPU detection by default for tensor creation
        if 'device' not in kwargs:
            kwargs['device'] = 'webgpu'  # Default to WebGPU instead of CPU
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
    
    def _matmul(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(np.dot(a.data, b.data), device="webgpu")
        return WebGPUTensor(np.dot(a, b))
    
    def _device(self, device_type):
        \"\"\"Create a device object\"\"\"
        return WebGPUDevice(device_type)
    
    def _sum(self, input_tensor, dim=None, keepdim=False, dtype=None):
        \"\"\"Compute sum of tensor elements\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            return input_tensor.sum(dim=dim, keepdim=keepdim)
        else:
            # Handle numpy arrays or lists
            if dim is None:
                result_data = np.sum(input_tensor)
            else:
                result_data = np.sum(input_tensor, axis=dim, keepdims=keepdim)
            return WebGPUTensor(result_data, dtype=dtype or 'float32')
    
    def _as_tensor(self, data, dtype=None, device=None):
        \"\"\"Convert data to tensor, similar to torch.as_tensor\"\"\"
        # Determine dtype
        if dtype is None:
            if hasattr(data, 'dtype'):
                dtype = str(data.dtype)
            else:
                dtype = 'float32'
        
        # Determine device - default to WebGPU for better performance
        if device is None:
            device = 'webgpu'
        
        # Create tensor
        return WebGPUTensor(data, dtype=dtype, device=device)
    
    def eye(self, n, m=None, dtype='float32', device='webgpu'):
        \"\"\"Create identity matrix\"\"\"
        if m is None:
            m = n
        data = np.eye(n, m)
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def round(self, input_tensor, decimals=0):
        """Round tensor elements to given number of decimals"""
        if isinstance(input_tensor, WebGPUTensor):
            rounded_data = np.round(input_tensor.data, decimals=decimals)
            return WebGPUTensor(rounded_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return WebGPUTensor(np.round(input_tensor, decimals=decimals))
    
    def det(self, input_tensor):
        \"\"\"Compute determinant of square matrix\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("det() expects a 2D square tensor")
            det_value = np.linalg.det(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor([det_value], device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.det(input_tensor)
    
    def _arange(self, *args, **kwargs):
        \"\"\"Create a 1D tensor with evenly spaced values\"\"\"
        if len(args) == 1:
            # arange(end)
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            # arange(start, end)
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            # arange(start, end, step)
            start, end, step = args[0], args[1], args[2]
        else:
            raise ValueError(\"arange() takes 1 to 3 positional arguments\")
        
        data = np.arange(start, end, step)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'int64' if isinstance(start, int) and isinstance(end, int) and isinstance(step, int) else 'float32')
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def _randperm(self, n, **kwargs):
        \"\"\"Generate a random permutation of integers from 0 to n-1\"\"\"
        data = np.random.permutation(n)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'int64')
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def _relu(self, input_tensor):
        \"\"\"ReLU activation function\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor.data, 0)
            result = WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
            
            if input_tensor.requires_grad:
                def relu_backward(grad_output):
                    if input_tensor.grad is None:
                        input_tensor.grad = WebGPUTensor(np.zeros_like(input_tensor.data), device="webgpu", dtype=input_tensor.dtype)
                    relu_grad = grad_output.data * (input_tensor.data > 0).astype(input_tensor.dtype)
                    input_tensor.grad.data += relu_grad
                
                result._backward_fn = relu_backward
                result._inputs = [input_tensor]
            
            return result
        else:
            return np.maximum(input_tensor, 0)


class TorchNN:
    def __init__(self):
        self.functional = TorchNNFunctional()
        self.Linear = TorchNNLinear
        self.Module = TorchNNModule

# Global cleanup function to prevent state persistence issues
def _cleanup_global_state():
    \"\"\"Clean up global tensor state to prevent persistence issues between executions\"\"\"
    import gc
    # Force garbage collection to clean up circular references
    gc.collect()

# Install in global namespace
torch = TorchModule()
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional
sys.modules['torch.linalg'] = torch.linalg

# Clean up any existing state to prevent issues
_cleanup_global_state()
`;
}

/**
 * Get optimized PyTorch polyfill for specific operations
 */
export function getOptimizedPolyfill(operations = []) {
  const basePolyfill = createPyTorchPolyfill();
  
  // Add operation-specific optimizations
  let optimizations = '';
  
  if (operations.includes('matmul')) {
    optimizations += `
# Optimized matrix multiplication
def optimized_matmul(a, b):
    if hasattr(a, '_webgpu_buffer') and hasattr(b, '_webgpu_buffer'):
        # Use WebGPU acceleration
        return a._webgpu_matmul(b)
    return np.dot(a, b)

torch._matmul = optimized_matmul
`;
  }
  
  return basePolyfill + optimizations;
}

/**
 * Validate PyTorch polyfill before installation
 */
export function validatePolyfill(polyfillCode) {
  const dangerousPatterns = [
    /\beval\s*\(/g,
    /\bexec\s*\(/g,
    /\b__import__\s*\(/g,
    /\bsubprocess\./g,
    /\bos\.system\s*\(/g
  ];
  
  for (const pattern of dangerousPatterns) {
    if (pattern.test(polyfillCode)) {
      throw new Error(`Dangerous pattern detected in PyTorch polyfill: ${pattern}`);
    }
  }
  
  return true;
}