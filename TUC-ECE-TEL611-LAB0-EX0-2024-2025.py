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