# Task 3 - Using Tuples
# Create a tuple with favorite things
favorite_things = ("Inception", "Bohemian Rhapsody", "1984")
print("Favorite things:", favorite_things)

# Try to change one of the elements (this will raise an error)
try:
    favorite_things[0] = "The Matrix"
except TypeError as e:
    print("Oops! Tuples cannot be changed. Error:", e)

# Print the length of the tuple
print("Length of tuple:", len(favorite_things))
