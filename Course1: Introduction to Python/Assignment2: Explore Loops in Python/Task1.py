# Task 1 - Countdown Timer
# Ask the user for a starting number
start = int(input("Enter the starting number: "))

# Use a while loop to count down
while start >= 1:
    print(start, end=" ")  # Print the current number
    start -= 1  # Decrease the number by 1

# Print the celebratory message
print("Blast off! 🚀")
