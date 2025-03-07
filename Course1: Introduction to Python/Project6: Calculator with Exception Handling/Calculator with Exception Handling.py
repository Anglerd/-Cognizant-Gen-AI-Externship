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
