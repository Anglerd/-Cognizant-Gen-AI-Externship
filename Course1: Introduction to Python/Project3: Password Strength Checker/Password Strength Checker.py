# Password Strength Checker
# Step 1: Input the password
password = input("Enter a password: ")

# Step 2: Evaluate the password
# Initialize variables to track password strength
has_upper = False
has_lower = False
has_digit = False
has_special = False
special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/"

# Check each condition
if len(password) >= 8:
    for char in password:
        if char.isupper():
            has_upper = True
        elif char.islower():
            has_lower = True
        elif char.isdigit():
            has_digit = True
        elif char in special_chars:
            has_special = True
else:
    print("Your password must be at least 8 characters long.")

# Bonus Challenge: Password Strength Meter
# Initialize a score variable
score = 0

# Check each condition and add to the score
if len(password) >= 8:
    score += 2  # Length contributes 2 points
if has_upper:
    score += 2  # Uppercase contributes 2 points
if has_lower:
    score += 2  # Lowercase contributes 2 points
if has_digit:
    score += 2  # Digit contributes 2 points
if has_special:
    score += 2  # Special character contributes 2 points

# Print the score and feedback
print(f"Password strength: {score}/10")
if score == 10:
    print("Your password is strong! 💪")
elif score >= 6:
    print("Your password is decent, but it could be stronger.")
else:
    print("Your password is weak. Please improve it.")
