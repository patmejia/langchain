from langchain.llms import OpenAI

# Initialize the OpenAI LLM with a temperature of 0.9
llm = OpenAI(temperature=0.9)

# Generate a company name based on the input prompt
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))

# Generate a description for a product based on the input prompt
text = "Write a short description of a smartwatch for a fitness enthusiast."
print(llm(text))

# Generate a movie plot summary based on the input prompt
text = "Write a plot summary for a movie about a group of friends on a road trip."
print(llm(text))
