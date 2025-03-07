# Task 1 - Understanding Python Exceptions
try:
    num = float(input("Enter a number: "))
    result = 100 / num
    print(f"100 divided by {num} is {result}.")
except ZeroDivisionError:
    print("Oops! You cannot divide by zero.")
except ValueError:
    print("Invalid input! Please enter a valid number.")
