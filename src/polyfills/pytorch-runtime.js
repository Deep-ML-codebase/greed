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
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(dtype)
        else:
            self._data = np.array(data, dtype=dtype)
        
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
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.grad = None
        self.grad_fn = None

    @property
    def data(self):
        """Return data wrapped with PyTorch-like methods"""
        return TensorDataWrapper(self._data, self)
    
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

    def numel(self):
        """Return the total number of elements in the tensor"""
        return self._data.size

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
        return self._data

    def tolist(self):
        return self._data.tolist()

    def __str__(self):
        """String representation of tensor"""
        result = f"tensor({self._data.tolist()}, requires_grad={self.requires_grad})"
        return result

    def __repr__(self):
        """Detailed representation of tensor"""
        return f"WebGPUTensor({self._data}, device='{self.device}', requires_grad={self.requires_grad})"

    def item(self):
        """Return the value of this tensor as a standard Python number"""
        if self._data.size == 1:
            value = self._data.item()
            # Ensure we return proper Python types that can be used as indices
            if self.dtype in ['int32', 'int64', 'long']:
                return int(value)
            elif self.dtype in ['float32', 'float64', 'double']:
                return float(value)
            else:
                # For other types, try to convert appropriately
                if isinstance(value, (int, np.integer)):
                    return int(value)
                elif isinstance(value, (float, np.floating)):
                    return float(value)
                else:
                    return value
        else:
            raise ValueError("only one element tensors can be converted to Python scalars")

    def __format__(self, format_spec):
        """Support for f-string formatting"""
        if self._data.size == 1:
            return format(self._data.item(), format_spec)
        else:
            return format(str(self), format_spec)

    def view(self, *shape):
        """Reshape tensor maintaining data"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        # Handle -1 for automatic size calculation
        if -1 in shape:
            total_size = self._data.size
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
        
        reshaped_data = self._data.reshape(shape)
        return WebGPUTensor(reshaped_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def transpose(self, dim0, dim1):
        transposed_data = np.swapaxes(self._data, dim0, dim1)
        return WebGPUTensor(transposed_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def unsqueeze(self, dim):
        """Add a dimension of size 1"""
        new_shape = list(self._data.shape)
        if dim < 0:
            dim = len(new_shape) + dim + 1
        new_shape.insert(dim, 1)
        reshaped_data = self._data.reshape(new_shape)
        return WebGPUTensor(reshaped_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions"""
        if end_dim == -1:
            end_dim = self._data.ndim - 1

        shape = list(self._data.shape)
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]

        new_shape = shape[:start_dim] + [flattened_size] + shape[end_dim + 1:]
        flattened_data = self._data.reshape(new_shape)
        return WebGPUTensor(flattened_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        if dim is None:
            # Remove all dimensions of size 1
            squeezed_data = np.squeeze(self._data)
        else:
            # Remove specific dimension if it has size 1
            if dim < 0:
                dim = self._data.ndim + dim
            if self._data.shape[dim] != 1:
                return self  # No change if dimension is not size 1
            squeezed_data = np.squeeze(self._data, axis=dim)

        return WebGPUTensor(squeezed_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            result_data = np.sum(self._data)
        else:
            result_data = np.sum(self._data, axis=dim, keepdims=keepdim)
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            result_data = np.mean(self._data)
        else:
            result_data = np.mean(self._data, axis=dim, keepdims=keepdim)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def std(self, dim=None, keepdim=False, unbiased=True):
        """Compute standard deviation"""
        if dim is None:
            result_data = np.std(self._data, ddof=1 if unbiased else 0)
        else:
            result_data = np.std(self._data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def var(self, dim=None, keepdim=False, unbiased=True):
        """Compute variance"""
        if dim is None:
            result_data = np.var(self._data, ddof=1 if unbiased else 0)
        else:
            result_data = np.var(self._data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def to(self, device):
        new_device = WebGPUDevice(device) if isinstance(device, str) else device
        return WebGPUTensor(self._data.copy(), device=new_device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('webgpu')  # Map CUDA to WebGPU

    def __getitem__(self, key):
        """Support tensor slicing like X[:, 0]"""
        if isinstance(key, tuple):
            # Multi-dimensional indexing
            indexed_data = self._data[key]
        else:
            # Single dimension indexing
            indexed_data = self._data[key]


        return WebGPUTensor(indexed_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """Backward propagation through the computation graph"""
        if not self.requires_grad:
            return

        if gradient is None:
            if self._data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = WebGPUTensor(np.ones_like(self._data), device=self.device, dtype=self.dtype)

        # Simple backward pass - accumulate gradients
        if hasattr(self, '_backward_fn') and self._backward_fn:
            self._backward_fn(gradient)
        else:
            # Initialize gradient for leaf variables
            if self.grad is None:
                self.grad = WebGPUTensor(np.zeros_like(self._data), device=self.device, dtype=self.dtype)
            self.grad._data += gradient._data if hasattr(gradient, '_data') else gradient

    def __repr__(self):
        return f"tensor({self._data}, device='{self.device}', dtype='{self.dtype}')"
    
    def __float__(self):
        """Convert single-element tensor to Python float"""
        if self._data.size == 1:
            return float(self._data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")
    
    def __int__(self):
        """Convert single-element tensor to Python int"""
        if self._data.size == 1:
            return int(self._data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")

    def __len__(self):
        """Return the length of the first dimension"""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __getitem__(self, key):
        """Support tensor indexing like tensor[indices]"""
        if isinstance(key, WebGPUTensor):
            # Convert WebGPUTensor indices to numpy array
            indices = key._data.astype(int)
            result_data = self._data[indices]

            # In PyTorch, indexing with tensor indices always preserves at least 1 dimension
            # even when the index tensor has 1 element
            if result_data.ndim == 0:
                result_data = np.array([result_data])
        else:
            result_data = self._data[key]

        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)\n    \n    def __setitem__(self, key, value):\n        \"\"\"Support tensor item assignment like tensor[indices] = value\"\"\"\n        if isinstance(value, WebGPUTensor):\n            value_data = value._data\n        else:\n            value_data = value\n\n        if isinstance(key, WebGPUTensor):\n            # Convert WebGPUTensor indices to numpy array\n            indices = key._data.astype(int)\n            self._data[indices] = value_data\n        else:\n            self._data[key] = value_data\n    \n    # Comparison operators\n    def __eq__(self, other):\n        \"\"\"Element-wise equality comparison\"\"\"\n        if isinstance(other, WebGPUTensor):\n            result_data = self._data == other._data\n        else:\n            result_data = self._data == other\n        return WebGPUTensor(result_data, device=\"webgpu\", dtype='bool')\n\n    def __ne__(self, other):\n        \"\"\"Element-wise not-equal comparison\"\"\"\n        if isinstance(other, WebGPUTensor):\n            result_data = self._data != other._data\n        else:\n            result_data = self._data != other\n        return WebGPUTensor(result_data, device=\"webgpu\", dtype='bool')\n    \n    def __gt__(self, other):
        """Element-wise greater than comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self._data > other._data
        else:
            result_data = self._data > other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __lt__(self, other):
        """Element-wise less than comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self._data < other._data
        else:
            result_data = self._data < other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __ge__(self, other):
        """Element-wise greater than or equal comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self._data >= other._data
        else:
            result_data = self._data >= other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __le__(self, other):
        """Element-wise less than or equal comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self._data <= other._data
        else:
            result_data = self._data <= other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    # Arithmetic operators
    def __add__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self._data + other._data
        else:
            result_data = self._data + other

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __sub__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self._data - other._data
        else:
            result_data = self._data - other
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __mul__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self._data * other._data
        else:
            result_data = self._data * other

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __truediv__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self._data / other._data
        else:
            result_data = self._data / other
        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def __pow__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = np.power(self._data, other._data)
        else:
            result_data = np.power(self._data, other)

        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __radd__(self, other):
        result_data = other + self._data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __rmul__(self, other):
        result_data = other * self._data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result

    def __rsub__(self, other):
        result_data = other - self._data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result

    def __rtruediv__(self, other):
        result_data = other / self._data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result

    def __rpow__(self, other):
        result_data = np.power(other, self._data)
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result
    
    def __matmul__(self, other):
        \"\"\"Matrix multiplication operator (@)\"\"\"
        if isinstance(other, WebGPUTensor):
            if self.ndim == 2 and other.ndim == 2:
                result_data = np.dot(self._data, other._data)
            elif self.ndim == 1 and other.ndim == 2:
                result_data = np.dot(self._data, other._data)
            elif self.ndim == 2 and other.ndim == 1:
                result_data = np.dot(self._data, other._data)
            else:
                result_data = np.matmul(self._data, other._data)
        else:
            result_data = np.matmul(self._data, other)
        
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype,
                             requires_grad=self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad))

        # Autograd disabled for stability - backward functions removed
        
        return result
    
    def __rmatmul__(self, other):
        \"\"\"Reverse matrix multiplication\"\"\"
        result_data = np.matmul(other, self._data)
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
            self.grad._data.fill(0)

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        \"\"\"Backward pass for automatic differentiation\"\"\"

        if not self.requires_grad:
            return

        # Default gradient is ones with same shape as tensor
        if gradient is None:
            if self._data.size == 1:
                # Scalar tensor - gradient is 1.0
                gradient = WebGPUTensor([1.0], device=self.device, dtype=self.dtype)
            else:
                # Non-scalar tensor - gradient should be provided
                gradient = WebGPUTensor(np.ones_like(self._data), device=self.device, dtype=self.dtype)

        # Call the backward function if it exists
        if hasattr(self, '_backward_fn') and self._backward_fn is not None:
            self._backward_fn(gradient)
        else:
            # Initialize gradient for leaf variables
            if self.grad is None:
                self.grad = WebGPUTensor(np.zeros_like(self._data), device=self.device, dtype=self.dtype)
            self.grad._data += gradient._data if hasattr(gradient, '_data') else gradient
    
    def _matmul_backward(self, grad_output, other):
        \"\"\"Backward pass for matrix multiplication\"\"\"
        if isinstance(other, WebGPUTensor):
            # d/da (a @ b) = grad_output @ b.T
            if self.grad is None:
                self.grad = WebGPUTensor(np.zeros_like(self._data), device="webgpu", dtype=self.dtype)
            self_grad = np.matmul(grad_output._data, other._data.T)
            self.grad._data += self_grad
            
            # d/db (a @ b) = a.T @ grad_output  
            if other.requires_grad:
                if other.grad is None:
                    other.grad = WebGPUTensor(np.zeros_like(other._data), device="webgpu", dtype=other.dtype)
                other_grad = np.matmul(self._data.T, grad_output._data)
                other.grad._data += other_grad

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
            det_value = np.linalg.det(input_tensor._data.reshape(input_tensor.shape))
            return WebGPUTensor([det_value], device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.det(input_tensor)
    
    def inv(self, input_tensor):
        \"\"\"Compute matrix inverse\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.inv() expects a 2D square tensor")
            inv_data = np.linalg.inv(input_tensor._data.reshape(input_tensor.shape))
            return WebGPUTensor(inv_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.inv(input_tensor)
    
    def norm(self, input_tensor, ord=None, dim=None, keepdim=False):
        \"\"\"Compute matrix or vector norm\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                norm_value = np.linalg.norm(input_tensor._data, ord=ord)
                return WebGPUTensor([norm_value], device="webgpu", dtype=input_tensor.dtype)
            else:
                norm_data = np.linalg.norm(input_tensor._data.reshape(input_tensor.shape), ord=ord, axis=dim, keepdims=keepdim)
                return WebGPUTensor(norm_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.linalg.norm(input_tensor, ord=ord, axis=dim, keepdims=keepdim)
    
    def eig(self, input_tensor):
        \"\"\"Compute eigenvalues and eigenvectors\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.eig() expects a 2D square tensor")
            eigenvalues, eigenvectors = np.linalg.eig(input_tensor._data.reshape(input_tensor.shape))
            return (
                WebGPUTensor(eigenvalues, device="webgpu", dtype=input_tensor.dtype),
                WebGPUTensor(eigenvectors, device="webgpu", dtype=input_tensor.dtype)
            )
        else:
            return np.linalg.eig(input_tensor)
    
    def svd(self, input_tensor, full_matrices=True):
        \"\"\"Compute singular value decomposition\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            U, S, Vh = np.linalg.svd(input_tensor._data.reshape(input_tensor.shape), full_matrices=full_matrices)
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
            result_data = np.maximum(input_tensor._data, 0)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.maximum(input_tensor, 0)
    
    @staticmethod
    def sigmoid(input_tensor):
        if isinstance(input_tensor, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-input_tensor._data))
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return 1 / (1 + np.exp(-input_tensor))

# Neural network modules
class TorchNNModule:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True
        
    def parameters(self):
        params = []
        debug_id = getattr(self, '_debug_id', 'UNKNOWN')
        class_name = self.__class__.__name__
        for param in self._parameters.values():
            params.append(param)
        for module_name, module in self._modules.items():
            if hasattr(module, 'parameters'):
                subparams = module.parameters()
                params.extend(subparams)
        return params
    
    def __setattr__(self, name, value):
        """Override setattr to automatically register modules and parameters"""
        # Check if the value is a TorchNNModule (submodule)
        if isinstance(value, TorchNNModule):
            if hasattr(self, '_modules'):
                self._modules[name] = value

        # Check if the value is a WebGPUTensor with requires_grad (parameter)
        elif isinstance(value, WebGPUTensor) and getattr(value, 'requires_grad', False):
            if hasattr(self, '_parameters'):
                self._parameters[name] = value

        # Default setattr behavior
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def to(self, device):
        """Move module to specified device"""
        # Convert device string to device object if needed
        if isinstance(device, str):
            if device == 'cuda':
                device = 'webgpu'  # Map CUDA to WebGPU
            target_device = device
        else:
            target_device = device.type if hasattr(device, 'type') else str(device)

        # Move all parameters to the new device
        for param in self.parameters():
            param.device = target_device
            # Update the device reference for WebGPU acceleration
            if hasattr(param, 'data'):
                param._data = param._data  # Data stays in numpy, device is just metadata

        # Recursively move submodules
        for module in self._modules.values():
            if hasattr(module, 'to'):
                module.to(target_device)

        return self

    def cpu(self):
        """Move module to CPU"""
        return self.to('cpu')

    def cuda(self):
        """Move module to WebGPU (CUDA equivalent)"""
        return self.to('webgpu')

    def train(self, mode=True):
        """Set the module in training mode"""
        self.training = mode
        # Recursively set training mode for all submodules
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode"""
        return self.train(False)

class TorchNNLinear(TorchNNModule):
    def __init__(self, in_features, out_features, bias=True):
        import random
        self._debug_id = random.randint(1000, 9999)
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True
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
            # Linear transformation: y = xW^T + b
            result_data = np.dot(x._data, self.weight._data.T)
            if self.bias is not None:
                result_data = result_data + self.bias._data

            result = WebGPUTensor(result_data, device="webgpu", dtype=x.dtype,
                                requires_grad=(x.requires_grad or self.weight.requires_grad or
                                             (self.bias is not None and self.bias.requires_grad)))

            # Set up backward function for linear layer
            def linear_backward(grad_output):

                if x.requires_grad:
                    # Gradient w.r.t input: grad_output @ weight
                    if x.grad is None:
                        x.grad = WebGPUTensor(np.zeros_like(x._data), device="webgpu", dtype=x.dtype)
                    x.grad._data += np.dot(grad_output._data, self.weight._data)

                if self.weight.requires_grad:
                    # Gradient w.r.t weight: x^T @ grad_output
                    if self.weight.grad is None:
                        self.weight.grad = WebGPUTensor(np.zeros_like(self.weight._data), device="webgpu", dtype=self.weight.dtype)
                    weight_grad = np.dot(grad_output._data.T, x._data)
                    self.weight.grad._data += weight_grad

                if self.bias is not None and self.bias.requires_grad:
                    # Gradient w.r.t bias: sum(grad_output, axis=0)
                    if self.bias.grad is None:
                        self.bias.grad = WebGPUTensor(np.zeros_like(self.bias._data), device="webgpu", dtype=self.bias.dtype)
                    bias_grad = np.sum(grad_output._data, axis=0)
                    self.bias.grad._data += bias_grad

            result._backward_fn = linear_backward
            result._inputs = [x, self.weight] + ([self.bias] if self.bias else [])

            return result
        else:
            raise TypeError("Input must be WebGPUTensor")

class TorchNNReLU(TorchNNModule):
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True

    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result_data = np.maximum(x.data, 0)
            return WebGPUTensor(result_data, device="webgpu", dtype=x.dtype, requires_grad=x.requires_grad)
        else:
            return np.maximum(x, 0)

class TorchNNMSELoss(TorchNNModule):
    def __init__(self, reduction='mean'):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        if isinstance(input_tensor, WebGPUTensor) and isinstance(target_tensor, WebGPUTensor):
            diff = input_tensor._data - target_tensor._data
            squared_error = diff ** 2

            if self.reduction == 'mean':
                loss_value = np.mean(squared_error)
            elif self.reduction == 'sum':
                loss_value = np.sum(squared_error)
            else:  # 'none'
                loss_value = squared_error

            loss_tensor = WebGPUTensor([loss_value] if np.isscalar(loss_value) else loss_value,
                                     device="webgpu", dtype=input_tensor.dtype, requires_grad=True)

            # Set up backward function for MSE loss
            def mse_backward(grad_output):
                # MSE gradient: 2 * (input - target) / N
                N = input_tensor._data.size if self.reduction == 'mean' else 1
                grad_input = 2.0 * diff / N

                if input_tensor.requires_grad:
                    # Create gradient tensor for the input
                    grad_input_tensor = WebGPUTensor(grad_input * (grad_output._data if hasattr(grad_output, '_data') else grad_output),
                                                   device="webgpu", dtype=input_tensor.dtype)

                    # Call backward on the input tensor (linear layer output) to propagate gradients
                    input_tensor.backward(grad_input_tensor)

            loss_tensor._backward_fn = mse_backward
            loss_tensor._inputs = [input_tensor, target_tensor]

            return loss_tensor
        else:
            raise TypeError("Both input and target must be WebGPUTensor")

class TorchNNCrossEntropyLoss(TorchNNModule):
    def __init__(self, reduction='mean'):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        if isinstance(input_tensor, WebGPUTensor) and isinstance(target_tensor, WebGPUTensor):
            # Softmax
            input_data = input_tensor._data
            exp_data = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
            softmax_data = exp_data / np.sum(exp_data, axis=1, keepdims=True)

            # Cross-entropy loss
            target_indices = target_tensor._data.astype(int)
            batch_size = input_data.shape[0]

            # Extract probabilities for target classes
            target_probs = softmax_data[np.arange(batch_size), target_indices]

            # Compute negative log likelihood
            nll = -np.log(np.clip(target_probs, 1e-8, 1.0))

            if self.reduction == 'mean':
                loss_value = np.mean(nll)
            elif self.reduction == 'sum':
                loss_value = np.sum(nll)
            else:  # 'none'
                loss_value = nll

            loss_tensor = WebGPUTensor([loss_value] if np.isscalar(loss_value) else loss_value,
                                     device="webgpu", dtype=input_tensor.dtype, requires_grad=True)

            # Set up backward function for CrossEntropyLoss
            def ce_backward(grad_output):
                if input_tensor.requires_grad:
                    # Gradient: softmax - one_hot(target)
                    grad_input = softmax_data.copy()
                    grad_input[np.arange(batch_size), target_indices] -= 1.0

                    if self.reduction == 'mean':
                        grad_input = grad_input / batch_size

                    grad_input_tensor = WebGPUTensor(grad_input * (grad_output._data if hasattr(grad_output, '_data') else grad_output),
                                                   device="webgpu", dtype=input_tensor.dtype)

                    # Call backward on the input tensor to propagate gradients
                    input_tensor.backward(grad_input_tensor)

            loss_tensor._backward_fn = ce_backward
            loss_tensor._inputs = [input_tensor, target_tensor]

            return loss_tensor
        else:
            raise TypeError("Both input and target must be WebGPUTensor")

# Create torch module with essential functions
class TorchModule:
    def __init__(self):
        self.tensor = self._tensor
        self.zeros = self._zeros
        self.ones = self._ones
        self.randn = self._randn
        self.zeros_like = self._zeros_like
        self.matmul = self._matmul
        self.sum = self._sum
        self.as_tensor = self._as_tensor
        self.arange = self._arange
        self.randperm = self._randperm
        self.linspace = self._linspace
        self.nn = TorchNN()

        # Add Tensor class reference
        self.Tensor = WebGPUTensor

        # Linear algebra module
        self.linalg = TorchLinalg()

        # Data utilities module
        self.utils = TorchUtils()

        # Optimizer module
        self.optim = TorchOptim()

        # Activation functions
        self.relu = self._relu

        # Mathematical functions
        self.round = self.round
        self.sqrt = self._sqrt
        self.pow = self._pow
        self.exp = self._exp
        self.log = self._log
        self.mean = self._mean
        self.max = self._max
        self.min = self._min
        self.transpose = self._transpose

        # Advanced mathematical functions
        self.cat = self._cat
        self.stack = self._stack
        self.std = self._std
        self.abs = self._abs
        self.sin = self._sin
        self.cos = self._cos
        self.clamp = self._clamp
        self.argmax = self._argmax

        # Context managers
        self.no_grad = self._no_grad

        # Device and CUDA support
        self.device = self._device
        self.cuda = TorchCuda()
        self.manual_seed = self._manual_seed

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
            rounded_data = np.round(input_tensor._data, decimals=decimals)
            return WebGPUTensor(rounded_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return WebGPUTensor(np.round(input_tensor, decimals=decimals))
    
    def det(self, input_tensor):
        \"\"\"Compute determinant of square matrix\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("det() expects a 2D square tensor")
            det_value = np.linalg.det(input_tensor._data.reshape(input_tensor.shape))
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

    def _linspace(self, start, end, steps, **kwargs):
        \"\"\"Generate linearly spaced values\"\"\"
        data = np.linspace(start, end, steps)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'float32')
        return WebGPUTensor(data, device=device, dtype=dtype)

    def _zeros_like(self, input_tensor, **kwargs):
        \"\"\"Create a tensor of zeros with the same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            data = np.zeros_like(input_tensor._data)
            device = kwargs.get('device', input_tensor.device)
            dtype = kwargs.get('dtype', input_tensor.dtype)
            return WebGPUTensor(data, device=device, dtype=dtype)
        else:
            data = np.zeros_like(input_tensor)
            device = kwargs.get('device', 'cpu')
            dtype = kwargs.get('dtype', 'float32')
            return WebGPUTensor(data, device=device, dtype=dtype)

    def _relu(self, input_tensor):
        \"\"\"ReLU activation function\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor._data, 0)
            result = WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
            
            if input_tensor.requires_grad:
                def relu_backward(grad_output):
                    if input_tensor.grad is None:
                        input_tensor.grad = WebGPUTensor(np.zeros_like(input_tensor._data), device="webgpu", dtype=input_tensor.dtype)
                    relu_grad = grad_output._data * (input_tensor._data > 0).astype(input_tensor.dtype)
                    input_tensor.grad._data += relu_grad
                
                result._backward_fn = relu_backward
                result._inputs = [input_tensor]
            
            return result
        else:
            return np.maximum(input_tensor, 0)

    def _sqrt(self, input_tensor):
        """Square root function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sqrt(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.sqrt(input_tensor)

    def _pow(self, input_tensor, exponent):
        """Power function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.power(input_tensor._data, exponent)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.power(input_tensor, exponent)

    def _exp(self, input_tensor):
        """Exponential function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.exp(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.exp(input_tensor)

    def _log(self, input_tensor):
        """Natural logarithm function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.log(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.log(input_tensor)

    def _mean(self, input_tensor, dim=None, keepdim=False):
        """Mean function"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.mean(input_tensor._data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.mean(input_tensor._data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.mean(input_tensor, axis=dim, keepdims=keepdim)

    def _max(self, input_tensor, dim=None, keepdim=False):
        """Maximum function - returns values and indices when dim is specified"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                # Global max - return single value
                result_data = np.max(input_tensor._data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                # Max along dimension - return (values, indices) tuple like PyTorch
                max_values = np.max(input_tensor._data, axis=dim, keepdims=keepdim)
                max_indices = np.argmax(input_tensor._data, axis=dim)
                if keepdim:
                    max_indices = np.expand_dims(max_indices, axis=dim)


                values_tensor = WebGPUTensor(max_values, device="webgpu", dtype=input_tensor.dtype)
                indices_tensor = WebGPUTensor(max_indices, device="webgpu", dtype='int64')
                return values_tensor, indices_tensor
        else:
            if dim is None:
                return np.max(input_tensor)
            else:
                max_values = np.max(input_tensor, axis=dim, keepdims=keepdim)
                max_indices = np.argmax(input_tensor, axis=dim)
                if keepdim:
                    max_indices = np.expand_dims(max_indices, axis=dim)
                return max_values, max_indices

    def _min(self, input_tensor, dim=None, keepdim=False):
        """Minimum function - returns values and indices when dim is specified"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                # Global min - return single value
                result_data = np.min(input_tensor._data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                # Min along dimension - return (values, indices) tuple like PyTorch
                min_values = np.min(input_tensor._data, axis=dim, keepdims=keepdim)
                min_indices = np.argmin(input_tensor._data, axis=dim)
                if keepdim:
                    min_indices = np.expand_dims(min_indices, axis=dim)


                values_tensor = WebGPUTensor(min_values, device="webgpu", dtype=input_tensor.dtype)
                indices_tensor = WebGPUTensor(min_indices, device="webgpu", dtype='int64')
                return values_tensor, indices_tensor
        else:
            if dim is None:
                return np.min(input_tensor)
            else:
                min_values = np.min(input_tensor, axis=dim, keepdims=keepdim)
                min_indices = np.argmin(input_tensor, axis=dim)
                if keepdim:
                    min_indices = np.expand_dims(min_indices, axis=dim)
                return min_values, min_indices

    def _transpose(self, input_tensor, dim0, dim1):
        """Transpose function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.swapaxes(input_tensor._data, dim0, dim1)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.swapaxes(input_tensor, dim0, dim1)

    def _cat(self, tensors, dim=0):
        """Concatenate tensors along specified dimension"""
        if not isinstance(tensors, (list, tuple)):
            raise TypeError("tensors must be a list or tuple")

        if len(tensors) == 0:
            raise RuntimeError("cat expects a non-empty list of tensors")

        # Convert all tensors to WebGPUTensor if needed
        tensor_list = []
        for tensor in tensors:
            if isinstance(tensor, WebGPUTensor):
                tensor_list.append(tensor.data)
            else:
                tensor_list.append(tensor)

        # Use numpy concatenate
        result_data = np.concatenate(tensor_list, axis=dim)
        return WebGPUTensor(result_data, device="webgpu", dtype=tensors[0].dtype)

    def _stack(self, tensors, dim=0):
        """Stack tensors along a new dimension"""
        if not isinstance(tensors, (list, tuple)):
            raise TypeError("tensors must be a list or tuple")

        if len(tensors) == 0:
            raise RuntimeError("stack expects a non-empty list of tensors")

        # Convert all tensors to WebGPUTensor if needed
        tensor_list = []
        for tensor in tensors:
            if isinstance(tensor, WebGPUTensor):
                tensor_list.append(tensor.data)
            else:
                tensor_list.append(tensor)

        # Use numpy stack
        result_data = np.stack(tensor_list, axis=dim)
        return WebGPUTensor(result_data, device="webgpu", dtype=tensors[0].dtype)

    def _std(self, input_tensor, dim=None, keepdim=False):
        """Standard deviation function"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.std(input_tensor._data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.std(input_tensor._data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.std(input_tensor, axis=dim, keepdims=keepdim)

    def _abs(self, input_tensor):
        """Absolute value function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.abs(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.abs(input_tensor)

    def _sin(self, input_tensor):
        """Sine function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sin(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.sin(input_tensor)

    def _cos(self, input_tensor):
        """Cosine function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.cos(input_tensor._data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.cos(input_tensor)

    def _clamp(self, input_tensor, min=None, max=None):
        """Clamp function - constrain values to a range"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.clip(input_tensor._data, min, max)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.clip(input_tensor, min, max)

    def _argmax(self, input_tensor, dim=None, keepdim=False):
        """Argmax function - indices of maximum values"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.argmax(input_tensor._data)
                return WebGPUTensor([result_data], device="webgpu", dtype='int64')
            else:
                result_data = np.argmax(input_tensor._data, axis=dim)
                if keepdim:
                    result_data = np.expand_dims(result_data, axis=dim)
                return WebGPUTensor(result_data, device="webgpu", dtype='int64')
        else:
            return np.argmax(input_tensor, axis=dim)

    def _no_grad(self):
        """No gradient context manager - simplified implementation"""
        class NoGradContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        return NoGradContext()

    def _device(self, device_name):
        """Create a device object"""
        if device_name == 'cuda':
            device_name = 'webgpu'  # Map CUDA to WebGPU
        return WebGPUDevice(device_name)

    def _manual_seed(self, seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return seed

class TorchNNL1Loss(TorchNNModule):
    def __init__(self, reduction='mean'):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._backward_pre_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}
        self.training = True
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        if isinstance(input_tensor, WebGPUTensor) and isinstance(target_tensor, WebGPUTensor):
            diff = np.abs(input_tensor._data - target_tensor._data)

            if self.reduction == 'mean':
                loss_value = np.mean(diff)
            elif self.reduction == 'sum':
                loss_value = np.sum(diff)
            else:  # 'none'
                loss_value = diff

            return WebGPUTensor([loss_value] if np.isscalar(loss_value) else loss_value,
                             device="webgpu", dtype=input_tensor.dtype)
        else:
            raise TypeError("Both input and target must be WebGPUTensor")


# Data utilities for torch.utils.data
class TensorDataset:
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("At least one tensor should be given")
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(tensor.data[index] for tensor in self.tensors)
        return tuple(WebGPUTensor(tensor.data[index]) for tensor in self.tensors)

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self._dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._current_index = 0

    def __iter__(self):
        self._current_index = 0
        self.indices = list(range(len(self._dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self._current_index >= len(self._dataset):
            raise StopIteration

        batch_indices = self.indices[self._current_index:self._current_index + self.batch_size]
        batch = []

        # Get first item to determine structure
        first_item = self._dataset[batch_indices[0]]
        num_outputs = len(first_item) if isinstance(first_item, tuple) else 1

        # Initialize batch lists
        if num_outputs > 1:
            batch = [[] for _ in range(num_outputs)]
            for idx in batch_indices:
                items = self._dataset[idx]
                for i, item in enumerate(items):
                    batch[i].append(item.data if hasattr(item, 'data') else item)

            # Convert to tensors
            result = tuple(WebGPUTensor(np.array(batch_data)) for batch_data in batch)
        else:
            batch_data = []
            for idx in batch_indices:
                item = self._dataset[idx]
                batch_data.append(item.data if hasattr(item, 'data') else item)
            result = WebGPUTensor(np.array(batch_data))

        self._current_index += self.batch_size
        return result

class TorchUtilsData:
    def __init__(self):
        self.TensorDataset = TensorDataset
        self.DataLoader = DataLoader

class TorchUtils:
    def __init__(self):
        self._data = TorchUtilsData()

    @property
    def data(self):
        """Return the data module"""
        return self._data

# SGD Optimizer
class SGDOptimizer:
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        for i, param in enumerate(params):
            pass  # Parameter validation could go here
        self.param_groups = [{'params': params, 'lr': lr, 'momentum': momentum,
                             'dampening': dampening, 'weight_decay': weight_decay, 'nesterov': nesterov}]
        self.state = {}

    def zero_grad(self):
        """Clear gradients of all parameters"""
        for group in self.param_groups:
            for param in group['params']:
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad._data = np.zeros_like(param.grad._data)

    def step(self):
        """Perform one optimization step"""
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if not hasattr(param, 'grad') or param.grad is None:
                    continue

                grad = param.grad._data

                # Add weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * param._data

                # Apply momentum
                if momentum != 0:
                    param_state = self.state.setdefault(id(param), {})
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = np.zeros_like(param._data)
                    buf = param_state['momentum_buffer']
                    buf = momentum * buf + grad
                    grad = buf

                # Update parameters
                old_data = param._data.copy()
                np.copyto(param._data, param._data - lr * grad)

class TorchOptim:
    def __init__(self):
        self.SGD = SGDOptimizer

# CUDA support for WebGPU acceleration
class TorchCuda:
    def __init__(self):
        pass

    def is_available(self):
        """Check if WebGPU acceleration is available (maps CUDA to WebGPU)"""
        # In browser, we use WebGPU instead of CUDA
        try:
            # Use JavaScript to check if WebGPU is actually available
            import js
            if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                return js.navigator.gpu is not None
            else:
                return False
        except:
            return False

class TorchNN:
    def __init__(self):
        self.functional = TorchNNFunctional()
        self.Linear = TorchNNLinear
        self.Module = TorchNNModule
        self.ReLU = TorchNNReLU
        self.MSELoss = TorchNNMSELoss
        self.L1Loss = TorchNNL1Loss
        self.CrossEntropyLoss = TorchNNCrossEntropyLoss

# Global cleanup function to prevent state persistence issues
def _cleanup_global_state():
    \"\"\"Clean up global tensor state to prevent persistence issues between executions\"\"\"
    import gc
    # Force garbage collection to clean up circular references
    gc.collect()

class TensorDataWrapper:
    """Wrapper for tensor data that provides PyTorch-like methods"""
    def __init__(self, numpy_array, parent_tensor=None):
        self._array = numpy_array
        self._parent = parent_tensor

    def cpu(self):
        """Return tensor data on CPU"""
        return TensorDataWrapper(self._array.copy(), self._parent)

    def numpy(self):
        """Return the underlying numpy array"""
        return self._array

    def __getitem__(self, key):
        """Support indexing operations"""
        result = self._array[key]
        return TensorDataWrapper(result, self._parent)

    def __setitem__(self, key, value):
        """Support item assignment"""
        if isinstance(value, TensorDataWrapper):
            self._array[key] = value._array
        else:
            self._array[key] = value

    def __len__(self):
        """Return length of the array"""
        return len(self._array)

    def __iter__(self):
        """Support iteration"""
        for item in self._array:
            yield TensorDataWrapper(item, self._parent) if hasattr(item, 'shape') else item

    # Arithmetic operators
    def __add__(self, other):
        """Addition operator"""
        if isinstance(other, TensorDataWrapper):
            result = self._array + other._array
        else:
            result = self._array + other
        return TensorDataWrapper(result, self._parent)

    def __sub__(self, other):
        """Subtraction operator"""
        if isinstance(other, TensorDataWrapper):
            result = self._array - other._array
        else:
            result = self._array - other
        return TensorDataWrapper(result, self._parent)

    def __mul__(self, other):
        """Multiplication operator"""
        if isinstance(other, TensorDataWrapper):
            result = self._array * other._array
        else:
            result = self._array * other
        return TensorDataWrapper(result, self._parent)

    def __truediv__(self, other):
        """Division operator"""
        if isinstance(other, TensorDataWrapper):
            result = self._array / other._array
        else:
            result = self._array / other
        return TensorDataWrapper(result, self._parent)

    def __pow__(self, other):
        """Power operator"""
        if isinstance(other, TensorDataWrapper):
            result = self._array ** other._array
        else:
            result = self._array ** other
        return TensorDataWrapper(result, self._parent)

    # Reverse arithmetic operators
    def __radd__(self, other):
        """Reverse addition"""
        result = other + self._array
        return TensorDataWrapper(result, self._parent)

    def __rsub__(self, other):
        """Reverse subtraction"""
        result = other - self._array
        return TensorDataWrapper(result, self._parent)

    def __rmul__(self, other):
        """Reverse multiplication"""
        result = other * self._array
        return TensorDataWrapper(result, self._parent)

    def __rtruediv__(self, other):
        """Reverse division"""
        result = other / self._array
        return TensorDataWrapper(result, self._parent)

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying numpy array"""
        return getattr(self._array, name)

    def __array__(self):
        """Support numpy array interface"""
        return self._array

    def __repr__(self):
        return repr(self._array)

# Install in global namespace
torch = TorchModule()
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional
sys.modules['torch.linalg'] = torch.linalg
sys.modules['torch.utils'] = torch.utils
sys.modules['torch.utils.data'] = torch.utils.data
sys.modules['torch.optim'] = torch.optim

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
  // Basic validation without pattern matching to avoid security conflicts
  if (!polyfillCode || typeof polyfillCode !== 'string') {
    throw new Error('Invalid polyfill code provided');
  }

  if (polyfillCode.length > 1000000) { // 1MB limit
    throw new Error('Polyfill code too large');
  }

  return true;
}