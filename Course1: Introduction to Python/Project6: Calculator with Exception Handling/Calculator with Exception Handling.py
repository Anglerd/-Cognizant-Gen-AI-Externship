# Step 1: Menu of Operations
def display_menu():
    print("Welcome to the Error-Free Calculator!")
    print("Choose an operation:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")

# Step 2: Input Validation
def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a valid number.")

# Step 3: Division with Exception Handling
def divide_numbers(num1, num2):
    try:
        return num1 / num2
    except ZeroDivisionError:
        print("Oops! Division by zero is not allowed.")
        return None

# Step 4: Logging Errors (Bonus)
import logging

logging.basicConfig(filename="error_log.txt", level=logging.ERROR, format="%(levelname)s:%(name)s:%(message)s")
    
# Step 5: User-Friendly Interface
def main():
    while True:
        display_menu()
        choice = input("> ")

        if choice == "1":
            num1 = get_number("Enter the first number: ")
            num2 = get_number("Enter the second number: ")
            result = num1 + num2
            print(f"The result is {result}.")

        elif choice == "2":
            num1 = get_number("Enter the first number: ")
            num2 = get_number("Enter the second number: ")
            result = num1 - num2
            print(f"The result is {result}.")

        elif choice == "3":
            num1 = get_number("Enter the first number: ")
            num2 = get_number("Enter the second number: ")
            result = num1 * num2
            print(f"The result is {result}.")

        elif choice == "4":
            num1 = get_number("Enter the first number: ")
            num2 = get_number("Enter the second number: ")
            result = divide_numbers(num1, num2)
            if result is not None:
                print(f"The result is {result}.")

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice! Please select a valid option (1-5).")

# Run the program
if __name__ == "__main__":
    main()
