# Task 2 - Multiplication Table
# Ask the user to input a number
num = int(input("Enter a number: "))

# Use a for loop to print the multiplication table
for i in range(1, 11):  # Loop from 1 to 10
    print(f"{num} x {i} = {num * i}")
