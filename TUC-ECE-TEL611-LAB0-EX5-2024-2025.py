"""
File Name: basic_python_data_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script serves as an introduction to Python programming for data analysis. It demonstrates various fundamental programming concepts, including data types, control structures, and the use of Python libraries such as NumPy, Pandas, and Matplotlib for performing basic data analysis and visualization tasks.

Key Components:
    1. **Data Types and Control Structures**: Examples of working with different data types (integers, floats, strings, booleans) and control structures (if-else conditions, for loops).
    2. **NumPy Arrays**: Basic creation and manipulation of NumPy arrays, performing mathematical operations on arrays, and visualizing data.
    3. **Pandas DataFrames**: Creating simple DataFrames, performing basic statistical analysis, and working with sample data.
    4. **Matplotlib Visualization**: Plotting basic graphs (e.g., sine wave, bar plot) and visualizing user input data.
    5. **User Interaction**: Collecting user input, creating a DataFrame from user-provided data, and performing basic analysis and visualization on the entered data.

Usage:
    - This script provides an interactive way to learn Python programming and perform simple data analysis.
    - The user is prompted to input their personal information, and then basic operations are performed on a data set that includes this input, along with some statistical analysis and visualization.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - scipy

Instructions:
    - Run the script in a Python environment.
    - Input your name, age, and city when prompted.
    - The script will perform basic data analysis and show results such as the sum of numbers, mean, and a simple plot.

Notes:
    - The script is designed to be a hands-on exercise to introduce the basic concepts of Python programming and data analysis.
    - It provides a foundation for more advanced data analysis tasks, such as working with larger data sets and performing more complex visualizations and analyses.

"""

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

#Final Task: Create a Data Analysis Script
#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Example for input and data processing
name = input("Enter your name: ")
age = int(input("Enter your age: "))
city = input("Enter your city: ")
# Create DataFrame from user input
data = {'Name': [name], 'Age': [age], 'City': [city]}
df = pd.DataFrame(data)
# Print DataFrame
print("\nYour DataFrame:")
print(df)
# Example of using NumPy array
numbers = np.array([10, 20, 30, 40])
print("\nSum of numbers:", np.sum(numbers))
# Visualize data
plt.plot(numbers)
plt.title('Numbers Plot')
plt.show()
# Basic statistical analysis
mean = np.mean(numbers)
print("\nMean of numbers:", mean)
