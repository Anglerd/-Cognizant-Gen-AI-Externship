Hey there, Python Troubleshooter!
Exceptions and errors are essential topics for writing robust and fault-tolerant code. This assignment will guide you in handling errors gracefully and understanding Python exceptions. Let’s begin! 
Task 1 - Understanding Python Exceptions

Write a Python program that:

    Prompts the user to enter a number.
    Tries to divide 100 by the number.
    Handles the following exceptions:
        ZeroDivisionError (when dividing by zero).
        ValueError (when the user enters non-numeric input).
    Prints appropriate error messages for each exception.

Example Output:
Enter a number: 0 Oops! You cannot divide by zero. Enter a number: abc Invalid input! Please enter a valid number. Enter a number: 4 100 divided by 4 is 25.0
Task 2 - Types of Exceptions

Create a program that intentionally raises and handles the following exceptions:

    IndexError by accessing an invalid list index.
    KeyError by trying to access a non-existent key in a dictionary.
    TypeError by adding a string and an integer.

Explain in comments how each error occurs and how it is handled.

Example Output:
IndexError occurred! List index out of range. KeyError occurred! Key not found in the dictionary. TypeError occurred! Unsupported operand types.
Task 3 - Using try...except...else...finally

Write a program that:

    Prompts the user to enter two numbers.
    Tries to divide the first number by the second number.
    Implements the following:
        try block to attempt the division.
        except block to handle exceptions.
        else block to display the result if no exceptions occur.
        finally block to print a closing message regardless of exceptions.

Example Output:
Enter the first number: 10 Enter the second number: 2 The result is 5.0. This block always executes.
