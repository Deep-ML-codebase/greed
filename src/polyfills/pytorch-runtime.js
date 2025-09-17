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

    def item(self):
        """Return the value of this tensor as a standard Python number"""
        if self.data.size == 1:
            value = self.data.item()
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
        if self.data.size == 1:
            return format(self.data.item(), format_spec)
        else:
            return format(str(self), format_spec)

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

    def unsqueeze(self, dim):
        """Add a dimension of size 1"""
        new_shape = list(self.data.shape)
        if dim < 0:
            dim = len(new_shape) + dim + 1
        new_shape.insert(dim, 1)
        reshaped_data = self.data.reshape(new_shape)
        return WebGPUTensor(reshaped_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions"""
        if end_dim == -1:
            end_dim = self.data.ndim - 1

        shape = list(self.data.shape)
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]

        new_shape = shape[:start_dim] + [flattened_size] + shape[end_dim + 1:]
        flattened_data = self.data.reshape(new_shape)
        return WebGPUTensor(flattened_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        if dim is None:
            # Remove all dimensions of size 1
            squeezed_data = np.squeeze(self.data)
        else:
            # Remove specific dimension if it has size 1
            if dim < 0:
                dim = self.data.ndim + dim
            if self.data.shape[dim] != 1:
                return self  # No change if dimension is not size 1
            squeezed_data = np.squeeze(self.data, axis=dim)

        return WebGPUTensor(squeezed_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

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

    def __len__(self):
        """Return the length of the first dimension"""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __getitem__(self, key):\n        \"\"\"Support tensor indexing like tensor[indices]\"\"\"\n        if isinstance(key, WebGPUTensor):\n            # Convert WebGPUTensor indices to numpy array\n            indices = key.data.astype(int)\n            result_data = self.data[indices]\n        else:\n            result_data = self.data[key]\n        \n        return WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)\n    \n    def __setitem__(self, key, value):\n        \"\"\"Support tensor item assignment like tensor[indices] = value\"\"\"\n        if isinstance(value, WebGPUTensor):\n            value_data = value.data\n        else:\n            value_data = value\n\n        if isinstance(key, WebGPUTensor):\n            # Convert WebGPUTensor indices to numpy array\n            indices = key.data.astype(int)\n            self.data[indices] = value_data\n        else:\n            self.data[key] = value_data\n    \n    # Comparison operators\n    def __eq__(self, other):\n        \"\"\"Element-wise equality comparison\"\"\"\n        if isinstance(other, WebGPUTensor):\n            result_data = self.data == other.data\n        else:\n            result_data = self.data == other\n        return WebGPUTensor(result_data, device=\"webgpu\", dtype='bool')\n\n    def __ne__(self, other):\n        \"\"\"Element-wise not-equal comparison\"\"\"\n        if isinstance(other, WebGPUTensor):\n            result_data = self.data != other.data\n        else:\n            result_data = self.data != other\n        return WebGPUTensor(result_data, device=\"webgpu\", dtype='bool')\n    \n    def __gt__(self, other):
        """Element-wise greater than comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self.data > other.data
        else:
            result_data = self.data > other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __lt__(self, other):
        """Element-wise less than comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self.data < other.data
        else:
            result_data = self.data < other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __ge__(self, other):
        """Element-wise greater than or equal comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self.data >= other.data
        else:
            result_data = self.data >= other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    def __le__(self, other):
        """Element-wise less than or equal comparison"""
        if isinstance(other, WebGPUTensor):
            result_data = self.data <= other.data
        else:
            result_data = self.data <= other
        return WebGPUTensor(result_data, device="webgpu", dtype='bool')

    # Arithmetic operators
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

    def __rsub__(self, other):
        result_data = other - self.data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result

    def __rtruediv__(self, other):
        result_data = other / self.data
        result = WebGPUTensor(result_data, device="webgpu", dtype=self.dtype, requires_grad=self.requires_grad)

        # Autograd disabled for stability - backward functions removed

        return result

    def __rpow__(self, other):
        result_data = np.power(other, self.data)
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
            result = WebGPUTensor(np.dot(x.data, self.weight.data.T), device="webgpu", dtype=x.dtype)
            if self.bias is not None:
                result.data = result.data + self.bias.data
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
            diff = input_tensor.data - target_tensor.data
            squared_error = diff ** 2

            if self.reduction == 'mean':
                loss_value = np.mean(squared_error)
            elif self.reduction == 'sum':
                loss_value = np.sum(squared_error)
            else:  # 'none'
                loss_value = squared_error

            return WebGPUTensor([loss_value] if np.isscalar(loss_value) else loss_value,
                             device="webgpu", dtype=input_tensor.dtype)
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

    def _linspace(self, start, end, steps, **kwargs):
        \"\"\"Generate linearly spaced values\"\"\"
        data = np.linspace(start, end, steps)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'float32')
        return WebGPUTensor(data, device=device, dtype=dtype)

    def _zeros_like(self, input_tensor, **kwargs):
        \"\"\"Create a tensor of zeros with the same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            data = np.zeros_like(input_tensor.data)
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

    def _sqrt(self, input_tensor):
        """Square root function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sqrt(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.sqrt(input_tensor)

    def _pow(self, input_tensor, exponent):
        """Power function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.power(input_tensor.data, exponent)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.power(input_tensor, exponent)

    def _exp(self, input_tensor):
        """Exponential function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.exp(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.exp(input_tensor)

    def _log(self, input_tensor):
        """Natural logarithm function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.log(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.log(input_tensor)

    def _mean(self, input_tensor, dim=None, keepdim=False):
        """Mean function"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.mean(input_tensor.data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.mean(input_tensor.data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.mean(input_tensor, axis=dim, keepdims=keepdim)

    def _max(self, input_tensor, dim=None, keepdim=False):
        """Maximum function"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.max(input_tensor.data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.max(input_tensor.data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.max(input_tensor, axis=dim, keepdims=keepdim)

    def _min(self, input_tensor, dim=None, keepdim=False):
        """Minimum function"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.min(input_tensor.data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.min(input_tensor.data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.min(input_tensor, axis=dim, keepdims=keepdim)

    def _transpose(self, input_tensor, dim0, dim1):
        """Transpose function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.swapaxes(input_tensor.data, dim0, dim1)
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
                result_data = np.std(input_tensor.data)
                return WebGPUTensor([result_data], device="webgpu", dtype=input_tensor.dtype)
            else:
                result_data = np.std(input_tensor.data, axis=dim, keepdims=keepdim)
                return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype)
        else:
            return np.std(input_tensor, axis=dim, keepdims=keepdim)

    def _abs(self, input_tensor):
        """Absolute value function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.abs(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.abs(input_tensor)

    def _sin(self, input_tensor):
        """Sine function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.sin(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.sin(input_tensor)

    def _cos(self, input_tensor):
        """Cosine function"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.cos(input_tensor.data)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.cos(input_tensor)

    def _clamp(self, input_tensor, min=None, max=None):
        """Clamp function - constrain values to a range"""
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.clip(input_tensor.data, min, max)
            return WebGPUTensor(result_data, device="webgpu", dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return np.clip(input_tensor, min, max)

    def _argmax(self, input_tensor, dim=None, keepdim=False):
        """Argmax function - indices of maximum values"""
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                result_data = np.argmax(input_tensor.data)
                return WebGPUTensor([result_data], device="webgpu", dtype='int64')
            else:
                result_data = np.argmax(input_tensor.data, axis=dim)
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
            diff = np.abs(input_tensor.data - target_tensor.data)

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


class TorchNN:
    def __init__(self):
        self.functional = TorchNNFunctional()
        self.Linear = TorchNNLinear
        self.Module = TorchNNModule
        self.ReLU = TorchNNReLU
        self.MSELoss = TorchNNMSELoss
        self.L1Loss = TorchNNL1Loss

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
  // Basic validation without pattern matching to avoid security conflicts
  if (!polyfillCode || typeof polyfillCode !== 'string') {
    throw new Error('Invalid polyfill code provided');
  }

  if (polyfillCode.length > 1000000) { // 1MB limit
    throw new Error('Polyfill code too large');
  }

  return true;
}