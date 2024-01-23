# Script for interacting with OpenAI API

import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# Initialize OpenAI client with API key
def initialize_openai_client():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to display all available models from OpenAI
def display_available_models(client):
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(model.id)

# Function to generate a joke using a specified OpenAI model
def create_joke(client, model_name="gpt-3.5-turbo-0301"):
    prompt = "Tell me a joke"
    response = client.completions.create(model=model_name, prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

# Main function to execute the script
def main():
    try:
        client = initialize_openai_client()
        display_available_models(client)
        joke = create_joke(client)
        print("\nJoke:\n" + "-" * 20)
        print(joke)

    except RateLimitError as e:
        print("API quota exceeded. Please check your OpenAI usage and limits at https://platform.openai.com/usage")
        print("Error message:", e)

if __name__ == "__main__":
    main()
