import random

# Generate a random number between 1 and 100
number_to_guess = random.randint(1, 100)

# Initialize variables
attempts = 0
max_attempts = 10
guess = None

print("Welcome to the Number Guessing Game! 🎲")
print(f"I'm thinking of a number between 1 and 100. Can you guess it in {max_attempts} attempts?")

# Use a while loop to keep prompting the user
while guess != number_to_guess and attempts < max_attempts:
    # Ask the user for their guess
    guess = int(input("Guess the number (between 1 and 100): "))
    attempts += 1  # Increment the number of attempts

    # Provide feedback based on the guess
    if guess < number_to_guess:
        print("Too low! Try again.")
    elif guess > number_to_guess:
        print("Too high! Try again.")
    else:
        print(f"Congratulations! You guessed it in {attempts} attempts! 🎉")

# If the user runs out of attempts
if attempts == max_attempts and guess != number_to_guess:
    print(f"Game over! Better luck next time! The number was {number_to_guess}.")
