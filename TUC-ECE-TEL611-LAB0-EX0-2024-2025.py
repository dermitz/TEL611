"""
File Name: numpy_array_operations.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the creation and manipulation of NumPy arrays in different dimensions. It covers the following aspects:
    1. Creation of 1D, 2D, and 3D arrays.
    2. Accessing and slicing arrays.
    3. Applying boolean masking to filter values.
    4. Selecting specific elements using indices.
    
Key Components:
    1. **Array Creation**: Demonstrates how to create arrays of different dimensions (1D, 2D, 3D) using `np.array()`.
    2. **Element Access**: Shows how to access specific elements in 1D, 2D, and 3D arrays.
    3. **Slicing**: Demonstrates array slicing, including slicing with steps and selecting specific rows and columns.
    4. **Boolean Masking**: Filters values using a boolean mask to identify elements greater than 25.
    5. **Indexing**: Selects elements from the array using specific indices.

Usage:
    - Run the script to see how NumPy arrays are created and manipulated.
    - The script will print examples of accessing and slicing arrays, applying masks, and selecting specific elements.

Dependencies:
    - numpy

Instructions:
    - Execute the script in a Python environment.
    - The script will display various examples of working with NumPy arrays, including:
        1. Accessing elements from a 3D array.
        2. Slicing a 1D array and filtering values based on a condition.
        3. Demonstrating how to access rows and columns of a 2D array.
        4. Using a boolean mask to filter and select elements.
        5. Selecting specific elements using indices.

Notes:
    - This script demonstrates basic NumPy operations and array manipulations, which are essential for working with numerical data in Python.
    - NumPy arrays are more efficient than Python lists when performing operations on large datasets.


"""

import numpy as np
arr_1d = np.array([10, 20, 30, 40, 50]) # Create a 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])    # Create a 2D array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Create a 3D array
print("Element at [0, 1, 0]:", arr_3d[0, 1, 0])  # 3	# Access specific elements in a 3D array
print("Slice [1:4]:", arr_1d[1:4])  # [20, 30, 40]	# Slice elements from index 1 to 3 (exclusive)
print("Every second element:", arr_1d[::2])  # [10, 30, 50]	# Slice with a step
print("First two rows, all columns:\n", arr_2d[:2, :])  # [[1, 2, 3], [4, 5, 6]]	# Slice rows and columns
print("All rows, first two columns:\n", arr_2d[:, :2])  # [[1, 2], [4, 5], [7, 8]]
print("Middle section (rows 1-2, cols 1-2):\n", arr_2d[1:3, 1:3])  # [[5, 6], [8, 9]]	# Slice specific part
print("All layers, first row, all columns:\n", arr_3d[:, 0, :])  # [[1, 2], [5, 6]], Slice along the dimensions
mask = arr_1d > 25	# Boolean mask
print("Boolean Mask:", mask)  # [False, False, True, True, True]
print("Values greater than 25:", arr_1d[mask])  # [30, 40, 50]	# Use mask to filter values
indices = [0, 2, 4]	# Select specific elements using indices
print("Selected elements:", arr_1d[indices])  # [10, 30, 50]
