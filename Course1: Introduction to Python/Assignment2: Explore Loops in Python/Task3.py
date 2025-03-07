# Task 3 - Factorial Calculation
# Ask the user for a number
num = int(input("Enter a number: "))

# Initialize the factorial variable
factorial = 1

# Use a for loop to calculate the factorial
for i in range(1, num + 1):  # Loop from 1 to num
    factorial *= i  # Multiply factorial by i

# Print the result
print(f"The factorial of {num} is {factorial}.")
