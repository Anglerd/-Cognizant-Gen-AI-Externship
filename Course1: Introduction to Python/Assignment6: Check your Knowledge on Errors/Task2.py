# Task 2 - Types of Exceptions
# IndexError
try:
    my_list = [1, 2, 3]
    print(my_list[5])  # Accessing an invalid index
except IndexError:
    print("IndexError occurred! List index out of range.")

# KeyError
try:
    my_dict = {"name": "Alice", "age": 25}
    print(my_dict["address"])  # Accessing a non-existent key
except KeyError:
    print("KeyError occurred! Key not found in the dictionary.")

# TypeError
try:
    result = "Hello" + 42  # Adding a string and an integer
except TypeError:
    print("TypeError occurred! Unsupported operand types.")
