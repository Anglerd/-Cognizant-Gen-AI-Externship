try:
    from openai import OpenAI
except ImportError:
    print("Prerequisites not installed, please run the following code to install the necessary files")
    print("pip install openai")
import os
import json

# Configuration file path
CONFIG_FILE = "config.json"

# Step 1: Set up the environment
def setup_openai():
    """
    Set up the OpenAI API key by either loading it from a config file or prompting the user.
    """
    # Check if the config file exists
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            api_key = config.get("api_key")
            if api_key:
                print("API key loaded from config file.")
                return OpenAI(api_key=api_key)

    # If the config file doesn't exist or the API key is missing, prompt the user
    print("You can find your API key at https://platform.openai.com/account/api-keys")
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: API key is required.")
        exit(1)

    # Save the API key to the config file
    with open(CONFIG_FILE, "w") as f:
        json.dump({"api_key": api_key}, f)
    print("API key saved to config file.")

    return OpenAI(api_key=api_key)

# Step 2: Build the text completion application
def generate_completion(client, prompt, max_tokens=50, temperature=0.7):
    """
    Generate text completions using the OpenAI GPT model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use the GPT-3.5 model
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,  # Control the length of the response
            temperature=temperature,  # Control creativity (0 = deterministic, 1 = creative)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Step 3: Debugging and handling common issues
def validate_input(prompt):
    """
    Validate user input to ensure it is not empty or irrelevant.
    """
    if not prompt.strip():
        print("Please enter a valid prompt.\n")
        return False
    return True

# Step 4: Evaluating the model's behavior
def evaluate_model(client):
    """
    Test the model with various prompts and parameters.
    """
    test_prompts = [
        "Write a short poem about the ocean.",
        "Explain quantum computing in simple terms.",
        "Summarize the benefits of exercise.",
        "What is the capital of France?",
        "Tell me a joke."
    ]

    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        response = generate_completion(client, prompt, max_tokens=100, temperature=0.5)
        print(f"AI Response: {response}\n")

# Main application loop
def text_completion_app(client):
    """
    Main loop for the text completion application.
    """
    print("Welcome to the AI-Powered Text Completion App!")
    print("Type 'exit' to quit the application.\n")

    while True:
        # Get user input
        prompt = input("Enter your prompt: ")

        # Exit condition
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Validate input
        if not validate_input(prompt):
            continue

        # Generate and display the completion
        print("\nGenerating response...\n")
        completion = generate_completion(client, prompt)
        print(f"AI Response: {completion}\n")

# Run the program
if __name__ == "__main__":
    # Step 1: Set up the environment and obtain the API key
    client = setup_openai()

    # Step 4: Evaluate the model's behavior
    print("Evaluating the model with test prompts...")
    evaluate_model(client)

    # Step 2: Run the text completion application
    text_completion_app(client)
