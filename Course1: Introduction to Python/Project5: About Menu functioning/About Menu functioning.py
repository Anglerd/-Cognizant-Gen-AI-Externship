# Step 1: Menu of Recursive Functions
def display_menu():
    print("Welcome to the Recursive Artistry Program!")
    print("Choose an option:")
    print("1. Calculate Factorial")
    print("2. Find Fibonacci")
    print("3. Draw a Recursive Fractal (Bonus)")
    print("4. Exit")

# Step 2: Factorial Function
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Step 3: Fibonacci Function
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Step 4: Recursive Fractal Pattern (Bonus)
import turtle

def draw_tree(branch_length, t):
    if branch_length > 5:
        t.forward(branch_length)
        t.right(20)
        draw_tree(branch_length - 15, t)
        t.left(40)
        draw_tree(branch_length - 15, t)
        t.right(20)
        t.backward(branch_length)

def fractal_tree():
    window = turtle.Screen()
    window.bgcolor("white")

    t = turtle.Turtle()
    t.speed(0)
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("green")

    draw_tree(75, t)
    window.mainloop()

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
