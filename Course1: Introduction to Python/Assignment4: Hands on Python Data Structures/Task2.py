# Task 2 - Exploring Dictionaries
# Create a dictionary with personal information
my_info = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}
print("Original dictionary:", my_info)

# Add a new key-value pair for "favorite color"
my_info["favorite color"] = "Blue"
print("After adding favorite color:", my_info)

# Update the "city" key with a new value
my_info["city"] = "San Francisco"
print("After updating city:", my_info)

# Print all keys and values using a loop
print("Keys and Values:")
for key, value in my_info.items():
    print(f"{key}: {value}")
