# Task 3 - Using try...except...else...finally
try:
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))
    result = num1 / num2
except ZeroDivisionError:
    print("Oops! You cannot divide by zero.")
except ValueError:
    print("Invalid input! Please enter valid numbers.")
else:
    print(f"The result is {result}.")
finally:
    print("This block always executes.")
