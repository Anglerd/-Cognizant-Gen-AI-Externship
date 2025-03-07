# Inventory Management Program

# Step 1: Create the Inventory
inventory = {
    "apple": (10, 2.5),
    "banana": (20, 1.2)
}

# Step 2: Add, Remove, and Update Items
def add_item(item_name, quantity, price):
    inventory[item_name] = (quantity, price)

def remove_item(item_name):
    if item_name in inventory:
        del inventory[item_name]
    else:
        print(f"{item_name} not found in inventory.")

def update_item(item_name, quantity=None, price=None):
    if item_name in inventory:
        current_quantity, current_price = inventory[item_name]
        if quantity is not None:
            current_quantity = quantity
        if price is not None:
            current_price = price
        inventory[item_name] = (current_quantity, current_price)
    else:
        print(f"{item_name} not found in inventory.")

# Step 3: Display the Inventory
def display_inventory():
    print("Current inventory:")
    for item, (quantity, price) in inventory.items():
        print(f"Item: {item}, Quantity: {quantity}, Price: ${price:.2f}")

# Step 4: Bonus - Calculate Total Value
def calculate_total_value():
    total_value = 0
    for item, (quantity, price) in inventory.items():
        total_value += quantity * price
    return total_value

# Main Program
print("Welcome to the Inventory Manager!")
display_inventory()

# Add a new item
add_item("mango", 15, 3.0)
print("\nAdding a new item: mango")
display_inventory()

# Update an item
update_item("apple", quantity=15)
print("\nUpdating quantity of apple to 15")
display_inventory()

# Remove an item
remove_item("banana")
print("\nRemoving banana from inventory")
display_inventory()

# Calculate total value
total_value = calculate_total_value()
print(f"\nTotal value of inventory: ${total_value:.2f}")
