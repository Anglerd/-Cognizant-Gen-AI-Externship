Let’s Make Loops Fun! 🎲
For this project, you’ll create a simple yet addictive number-guessing game. Here’s how it works:

Step 1: Generate a Random Number
Use Python’s random module to generate a random number between 1 and 100.
import random
number_to_guess = random.randint(1, 100)

Step 2: Prompt the User for Guesses
Use a while loop to let the user keep guessing until they get the correct answer. After each guess:

    If the guess is too high, print: "Too high! Try again."
    If the guess is too low, print: "Too low! Try again."
    If the guess is correct, print: "Congratulations! You guessed it!"

Step 3: Count the Attempts
Keep track of how many guesses the user has made and display the total number of attempts when they win.

Example Run:
Guess the number (between 1 and 100): 50 Too high! Try again.
Guess the number (between 1 and 100): 25 Too low! Try again.
Guess the number (between 1 and 100): 37 Congratulations! You guessed it in 3 attempts!

Bonus Task:

    Limit the number of attempts to 10. If the user fails to guess in 10 attempts, end the game with a message like "Game over! Better luck next time!"

Make your program friendly, interactive, and fun to play! 😊
