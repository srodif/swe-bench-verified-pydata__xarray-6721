"""
Test for the zarr chunks memory loading fix.

This test ensures that accessing chunk information doesn't trigger unnecessary data loading
for arrays that have chunk information stored in encoding (like zarr arrays).
"""
import numpy as np
import pytest

from xarray.core.common import get_chunksizes
from xarray.core.variable import Variable


class TestChunksizeMemoryFix:
    """Test that chunksizes can be accessed without loading data into memory."""

    def test_variable_chunksizes_with_encoding(self):
        """Test that Variable.chunksizes uses encoding when available."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        dims = ["x", "y"]
        encoding = {"chunks": (1, 2)}

        var = Variable(dims, data, encoding=encoding)
        chunksizes = var.chunksizes

        expected = {"x": 1, "y": 2}
        assert chunksizes == expected

    def test_variable_chunksizes_without_encoding(self):
        """Test that Variable.chunksizes returns empty dict for regular arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        dims = ["x", "y"]

        var = Variable(dims, data)
        chunksizes = var.chunksizes

        assert chunksizes == {}

    def test_get_chunksizes_with_encoding(self):
        """Test that get_chunksizes works with encoding-based chunks."""
        data1 = np.array([[1, 2], [3, 4]])
        var1 = Variable(["x", "y"], data1, encoding={"chunks": (1, 2)})

        data2 = np.array([10, 20, 30])
        var2 = Variable(["z"], data2, encoding={"chunks": (2,)})

        variables = [var1, var2]
        chunksizes = get_chunksizes(variables)

        expected = {"x": 1, "y": 2, "z": 2}
        assert dict(chunksizes) == expected

    def test_get_chunksizes_mixed_variables(self):
        """Test get_chunksizes with mix of encoding and non-encoding variables."""
        # Variable with encoding
        data1 = np.array([[1, 2], [3, 4]])
        var1 = Variable(["x", "y"], data1, encoding={"chunks": (1, 2)})

        # Variable without encoding (regular numpy array)
        data2 = np.array([10, 20, 30])
        var2 = Variable(["z"], data2)

        variables = [var1, var2]
        chunksizes = get_chunksizes(variables)

        # Only the variable with encoding should contribute to chunks
        expected = {"x": 1, "y": 2}
        assert dict(chunksizes) == expected

    def test_chunksizes_no_data_access(self):
        """Test that chunksizes doesn't trigger data access for encoding-based chunks."""

        class DataAccessTracker:
            """Array-like object that tracks when data is accessed."""

            def __init__(self, shape):
                self.shape = shape
                self.dtype = np.float32
                self.data_access_count = 0

            def __array__(self):
                self.data_access_count += 1
                return np.zeros(self.shape, dtype=self.dtype)

        # Create variable with encoding and mock data
        tracker = DataAccessTracker((10, 20))
        var = Variable(
            ["x", "y"], tracker, encoding={"chunks": (5, 10)}, fastpath=True
        )

        # Reset access count after variable creation
        initial_count = tracker.data_access_count

        # Access chunksizes - should not trigger additional data access
        chunksizes = var.chunksizes

        final_count = tracker.data_access_count

        # Verify chunksizes are correct and no additional data access occurred
        assert chunksizes == {"x": 5, "y": 10}
        assert (
            final_count == initial_count
        ), f"Data access count increased from {initial_count} to {final_count}"

    def test_get_chunksizes_consistent_dimensions(self):
        """Test that get_chunksizes validates consistent chunk sizes across variables."""
        # Two variables with same dimension but different chunk sizes
        data1 = np.array([[1, 2], [3, 4]])
        var1 = Variable(["x", "y"], data1, encoding={"chunks": (1, 2)})

        data2 = np.array([[5, 6], [7, 8]])
        var2 = Variable(["x", "y"], data2, encoding={"chunks": (2, 2)})  # Different x chunk size

        variables = [var1, var2]

        # Should raise ValueError for inconsistent chunks
        with pytest.raises(ValueError, match="inconsistent chunks along dimension x"):
            get_chunksizes(variables)