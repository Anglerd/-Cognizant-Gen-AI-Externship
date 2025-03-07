# Task 4 - Understanding Recursion
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
factorial_result = factorial(5)
fibonacci_result = fibonacci(6)
print(f"Factorial of 5 is {factorial_result}.")
print(f"The 6th Fibonacci number is {fibonacci_result}.")
