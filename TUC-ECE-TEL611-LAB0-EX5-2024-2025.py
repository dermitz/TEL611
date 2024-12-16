#Step 1: Install and Import Libraries
# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Step 2: Work with Data Types and Control Structures
# Declare variables of different data types
integer_variable = 42
float_variable = 3.14
string_variable = "Hello, Python!"
boolean_variable = True
# Use if-else condition to check if a number is positive or negative
number = 15
if number > 0:
    print("Positive")
else:
    print("Negative")
# Create a list and iterate over it using a for loop
my_list = [10, 20, 30, 40]
for item in my_list:
    print(f"Item: {item}")

#Step 3: Work with NumPy Arrays
# Create NumPy arrays
array_1 = np.array([1, 2, 3, 4, 5])
array_2 = np.array([5, 4, 3, 2, 1])
# Perform mathematical operations on arrays
sum_array = array_1 + array_2
product_array = array_1 * array_2
print("Sum of arrays:", sum_array)
print("Product of arrays:", product_array)
# Create some data to plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
# Plot the data
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

import pandas as pd
# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [24, 27, 22, 32],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
df = pd.DataFrame(data)
# Print the first few rows of the DataFrame
print(df.head())
# Create a sample data set
data = [5, 10, 15, 20, 25, 30]
# Calculate mean, median
mean = np.mean(data)
median = np.median(data)
print(f"Mean: {mean}")
print(f"Median: {median}")
