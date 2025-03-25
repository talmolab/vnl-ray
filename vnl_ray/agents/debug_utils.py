"""Debugging utilities for the VNL-Ray codebase."""

import inspect
import sys
import types
import functools
import traceback


def trace_copy_calls(obj, max_depth=5):
    """
    Recursively inspect an object and its attributes to trace all `.copy()` method calls.

    Args:
        obj: The object to inspect
        max_depth: Maximum recursion depth

    Returns:
        A list of (object_path, has_copy_method) tuples
    """
    results = []
    visited = set()

    def _trace_recursive(current_obj, path, depth):
        if depth > max_depth or id(current_obj) in visited:
            return

        visited.add(id(current_obj))

        # Check for copy method
        has_copy = hasattr(current_obj, "copy") and callable(getattr(current_obj, "copy"))
        results.append((path, type(current_obj).__name__, has_copy))

        # Recursively inspect attributes if it's a class or module
        if isinstance(current_obj, (type, types.ModuleType)) or hasattr(current_obj, "__dict__"):
            for name, attr in inspect.getmembers(current_obj):
                # Skip special methods, callables, and modules
                if (
                    not name.startswith("__")
                    and not isinstance(attr, (types.FunctionType, types.MethodType, types.ModuleType))
                    and not callable(attr)
                ):
                    _trace_recursive(attr, f"{path}.{name}", depth + 1)

    _trace_recursive(obj, obj.__class__.__name__, 0)
    return results


def patch_tensorflow_tensor():
    """
    Patch TensorFlow's EagerTensor class to log when someone tries to call copy().
    This helps identify where in the codebase copy() is being called.
    """
    from tensorflow.python.framework.ops import EagerTensor

    # Store the original __getattr__
    original_getattr = EagerTensor.__getattr__

    @functools.wraps(original_getattr)
    def patched_getattr(self, name):
        if name == "copy":
            print("\n==== COPY METHOD ACCESSED ON TENSORFLOW TENSOR ====")
            print(f"TensorFlow tensor of shape {self.shape} and dtype {self.dtype}")
            print("Call stack:")
            for frame in traceback.extract_stack():
                filename, line, func, _ = frame
                print(f"  File {filename}, line {line}, in {func}")
            print("==================================================\n")

            # Return a function that wraps numpy's copy
            def tensor_copy():
                import numpy as np

                return np.array(self.numpy())

            return tensor_copy

        return original_getattr(self, name)

    # Apply the patch
    EagerTensor.__getattr__ = patched_getattr
    print("TensorFlow EagerTensor patched to trace copy() calls")


def install_tensor_copy_tracer():
    """
    Install the tensor copy tracer at runtime.
    Call this function early in your program to catch all copy() calls.
    """
    try:
        patch_tensorflow_tensor()
        return True
    except Exception as e:
        print(f"Failed to install tensor copy tracer: {e}")
        return False
