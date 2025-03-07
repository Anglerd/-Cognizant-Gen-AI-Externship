In this project, you will create a Python calculator that gracefully handles exceptions and provides helpful feedback to users.
Step 1: Menu of Operations

Create a program that displays a menu with the following operations:

    Addition
    Subtraction
    Multiplication
    Division
    Exit

The user should select an option by entering a number.
Step 2: Input Validation

Add input validation to ensure the user enters numbers only for the calculations. If the input is invalid, catch the exception and prompt the user again.

Example Output:
plaintext
Copy code
Enter the first number: abc Invalid input! Please enter a valid number.
Step 3: Division with Exception Handling

Handle the following exceptions specifically for the division operation:

    ZeroDivisionError: Print a friendly message when dividing by zero.
    ValueError: Ensure the inputs are numbers.

Example Output:
plaintext
Copy code
Enter the first number: 10 Enter the second number: 0 Oops! Division by zero is not allowed.
Step 4: Logging Errors (Bonus)

Use the logging module to log errors to a file named error_log.txt whenever an exception occurs.
Example Log Entry:
plaintext
Copy code
ERROR:root:ZeroDivisionError occurred: division by zero.
Step 5: User-Friendly Interface

Enhance the program with the following features:

    Clear prompts and instructions.
    A try...except...else...finally structure to ensure robustness.
    Informative error messages for each exception.

Example Run:
plaintext
Copy code
Welcome to the Error-Free Calculator! Choose an operation: 1. Addition 2. Subtraction 3. Multiplication 4. Division 5. Exit > 4 Enter the first number: 10 Enter the second number: 0 Oops! Division by zero is not allowed. Choose an operation: 5 Goodbye!

Happy debugging and exception handling!
