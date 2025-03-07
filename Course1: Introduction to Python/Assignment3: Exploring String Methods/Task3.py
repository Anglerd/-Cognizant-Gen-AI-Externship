# Task 3 - Check for Palindromes
# Ask the user to input a word
word = input("Enter a word: ")

# Use slicing to reverse the word
reversed_word = word[::-1]

# Check if the word is a palindrome
if word == reversed_word:
    print(f"Yes, '{word}' is a palindrome! 🎉")
else:
    print(f"No, '{word}' is not a palindrome. Better luck next time! 😊")
