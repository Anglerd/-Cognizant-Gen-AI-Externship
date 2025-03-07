# Step 5: User-Friendly Program
def main():
    while True:
        display_menu()
        choice = input("> ")

        if choice == "1":
            try:
                num = int(input("Enter a number to find its factorial: "))
                if num < 0:
                    print("Please enter a non-negative integer.")
                else:
                    print(f"The factorial of {num} is {factorial(num)}.")
            except ValueError:
                print("Invalid input! Please enter a valid integer.")

        elif choice == "2":
            try:
                num = int(input("Enter the position of the Fibonacci number: "))
                if num < 0:
                    print("Please enter a non-negative integer.")
                else:
                    print(f"The {num}th Fibonacci number is {fibonacci(num)}.")
            except ValueError:
                print("Invalid input! Please enter a valid integer.")

        elif choice == "3":
            print("Drawing a fractal tree... 🌳")
            fractal_tree()

        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice! Please select a valid option (1-4).")

# Run the program
if __name__ == "__main__":
    main()
