# 🦜🔗 LangChain - Comprehensive Guide

## Overview

LangChain now integrates with Multion API, enhancing its NLP application development capabilities. This addition complements the existing OpenAI API, offering advanced functionalities for chatbots and automated writing assistants.

For detailed information, visit: [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction).

![LangChain Architecture Diagram](img/langchain_art.png)

## Workflow in LangChain

1. **Data Acquisition**: Gather various types of data.
2. **Data Processing**: Includes vectorization, model development, training, testing, and deployment.
3. **Machine Learning Application**: Use machine learning for dataset queries.
4. **Model Interaction**: Submit queries for responses.
5. **Receive and Utilize Outputs**: Use the outputs received from the model.

## Setup

### Python Configuration

#### Environment Setup

```bash
# Set up project environment
cd your_project_directory
echo "OPENAI_API_KEY=YOUR_API_KEY_HERE" > .env
source .env
echo "OPENAI_API_KEY=${OPENAI_API_KEY:0:5}..."
```

##### API Key: [Your API Key](https://platform.openai.com/account/api-keys)

#### Conda Environment

```bash
# Create and activate a Conda environment
conda create --name langchain_env python=3.11
conda activate langchain_env

# Install dependencies
pip install -r requirements.txt
```

#### Script Execution

```bash
# Run OpenAI, LangChain, and Multion scripts
python3 src/my_openai.py
python3 src/llm_example.py
python3 src/multion_integration.py
```

#### Simplified Single Command Execution

```bash
cd your_project_directory && \
echo "OPENAI_API_KEY=YOUR_API_KEY_HERE" > .env && \
source .env && \
conda create --name langchain_env python=3.11 && \
conda activate langchain_env && \
pip install -r requirements.txt && \
python3 src/my_openai.py && \
python3 src/llm_example.py && \
python3 src/multion_integration.py
```

### Troubleshooting Tips

- Confirm `.env` file placement in project root.
- Verify Python version in Conda environment.
- For package installation issues, review `requirements.txt`.
- Address OpenAI library issues with `openai migrate`.

For comprehensive troubleshooting, refer to the OpenAI Python library [README](https://github.com/openai/openai-python) and [migration guide](https://github.com/openai/openai-python/discussions/742).

## LangChain Application Development

### Building Language Model Applications

#### Filename: `openai_llm.py`

```python
# Import OpenAI from langchain.llms
from langchain.llms import OpenAI

# Initialize OpenAI with model name and parameters
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

# Generate a joke using the language model
llm("Tell me a joke")
# Output: "Why did the chicken cross the road? To get to the other side."

# Generate multiple outputs using the language model
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

# Print the length of the generations
print(len(llm_result.generations)) # Output: 30

# Print the first generation
print(llm_result.generations[0])
```

#### Running the Language Model Application

```bash
# Run the openai_llm.py script
python openai_llm.py
```

#### Alternatively, use Hugging Face

```bash
# Install transformers
pip install transformers
```

### Introduction to "modular-abstraction/chains"

```python
# Import PromptTemplate and OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Initialize OpenAI and PromptTemplate
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Running the Chain
# Import LLMChain
from langchain.chains import LLMChain

# Initialize LLMChain with llm and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Execute the chain by specifying only the input variable.
print(chain.run("colorful socks"))
```

### Using Chat Models

```python
# Import ChatOpenAI, ChatPromptTemplate, and HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

# Initialize HumanMessagePromptTemplate with a prompt
human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is a good name for a company that makes {product}

?",
        input_variables=["product"],
    )
)

# Initialize ChatPromptTemplate with messages
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

# Initialize ChatOpenAI
chat = ChatOpenAI(temperature=0.9)

# Initialize LLMChain with llm and prompt
chain = LLMChain(llm=chat, prompt=chat_prompt_template)

# Execute the chain by specifying only the input variable.
print(chain.run("colorful socks"))
```

### Creating a Second Chain

```python
# Initialize a second PromptTemplate
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Compose a catchphrase for the following company: {company_name}",
)

# Initialize a second LLMChain with llm and the second prompt
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```

### Using SimpleSequentialChain

```python
# Import SimpleSequentialChain
from langchain.chains import SimpleSequentialChain

# Initialize SimpleSequentialChain with chains and verbose mode
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Execute the chain by specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")

# Print the catchphrase
print(catchphrase)
```

### Defining a New Chain Class: ConcatenateChain

```python
# Import LLMChain, Chain, Dict, and List
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from typing import Dict, List

# Define a new chain class: ConcatenateChain
class ConcatenateChain(Chain):
    def __init__(self, chain_1: LLMChain, chain_2: LLMChain):
        self.chain_1 = chain_1
        self.chain_2 = chain_2

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.chain_1.input_keys).union(set(self.chain_2.input_keys)))

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}
```

### Using ConcatenateChain

```python
# Initialize two PromptTemplates
prompt_1 = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
prompt_2 = PromptTemplate(
    input_variables=["product"],
    template="What is a catchy slogan for a company that makes {product}?",
)

# Initialize two LLMChains with llm and the prompts
chain_1 = LLMChain(llm=llm, prompt=prompt_1)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

# Initialize ConcatenateChain with the two chains
concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)

# Run the ConcatenateChain with an input
concat_output = concat_chain.run("colorful socks")

# Print the concatenated output
print(f"Concatenated output:\n{concat_output}")
```

## References

Here are some resources that have been invaluable in the development of LangChain:

- [LangChain Quickstart Guide](https://python.langchain.com/)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [GPT-3 Creative Applications](https://openai.com/blog/gpt-3-apps/)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)