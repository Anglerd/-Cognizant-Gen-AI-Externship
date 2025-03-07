# Voter Eligibility Checker
try:
    # Ask the user's age
    age = int(input("How old are you? "))
    
    # Check for invalid age (negative or unrealistic)
    if age < 0:
        print("Oops! Age can't be negative. Are you a time traveler? 🕰️")
    elif age > 120:
        print("Wow, you're either a legend or entered an invalid age! 😲")
    else:
        # Decide the eligibility
        if age >= 18:
            print("Congratulations! You are eligible to vote. Go make a difference! 🎉")
        else:
            years_to_wait = 18 - age
            print(f"Oops! You’re not eligible yet. But hey, only {years_to_wait} more years to go! 😊")
except ValueError:
    print("Invalid input! Please enter a valid age (numbers only). 🤔")
