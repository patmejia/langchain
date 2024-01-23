# Multion Integration Script for AlphaCodium Explanation

import os
import openai  # Import OpenAI module
import multion
from dotenv import load_dotenv
from langchain_community.agent_toolkits import MultionToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

# Setup and Initialization
def initialize_multion_toolkit():
    toolkit = MultionToolkit()
    return toolkit

def multion_setup():
    multion.login()  # Login to MultiOn

def get_multion_tools(toolkit):
    return toolkit.get_tools()

# Agent Setup and Execution
def setup_agent(toolkit):
    # Load OpenAI API key
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    instructions = "You are an assistant."
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=False,
    )
    return agent_executor

def execute_agent(agent_executor, input_text):
    return agent_executor.invoke({"input": input_text})

# Main Function
def main():
    try:
        toolkit = initialize_multion_toolkit()
        multion_setup()
        tools = get_multion_tools(toolkit)
        agent_executor = setup_agent(toolkit)
        input_text = "Use multion to explain how AlphaCodium works, a recently released code language model."
        result = execute_agent(agent_executor, input_text)
        print(result)
    except openai.RateLimitError as e:
        print("Rate limit exceeded: ", e)
        # Handle rate limit error, possibly wait or inform the user
    except Exception as e:
        print("An error occurred: ", e)
        # Handle other potential errors

if __name__ == "__main__":
    main()
