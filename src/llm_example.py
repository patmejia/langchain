import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI as LangChainOpenAI

# Function to load environment variables
def load_environment_variables():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

# Function to initialize Langchain OpenAI LLM
def initialize_llm(api_key):
    return LangChainOpenAI(temperature=0.9, api_key=api_key)

# Function to generate movie plot summary
def generate_plot_summary(llm, text):
    try:
        return llm.invoke(input=text, max_tokens=100)
    except Exception as e:
        print(f"Error in LangChain text generation: {e}")
        exit()

# Function to check coherence of generated text
def check_coherence(generated_text):
    try:
        return openai.Completion.create(
            engine="davinci",
            prompt=generated_text,
            max_tokens=100
        )
    except Exception as e:
        print(f"Error from OpenAI API: {e}")
        exit()

# Function to print movie plot summary
def print_plot_summary(response):
    print("\nMovie Plot Summary:\n" + "-" * 20)
    print(response.choices[0].text.strip())

# Main function to orchestrate the generation of movie plot summary
def main():
    api_key = load_environment_variables()
    openai.api_key = api_key
    llm = initialize_llm(api_key)
    text = input("Enter a prompt for a movie plot summary: ")
    generated_text = generate_plot_summary(llm, text)
    response = check_coherence(generated_text)
    print_plot_summary(response)

if __name__ == "__main__":
    main()
