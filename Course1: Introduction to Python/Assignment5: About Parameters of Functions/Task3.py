# Task 3 - Functions with Variable Arguments
def make_sandwich(*ingredients):
    print("Making a sandwich with the following ingredients:")
    for ingredient in ingredients:
        print(f"- {ingredient}")

# Example usage
make_sandwich("Lettuce", "Tomato", "Cheese")
