
## 🦜🔗 LangChain - Overview
### Quick Start and Advanced Usage
![LangChain Architecture Diagram](img/langchain_art.png)
LangChain, leveraging OpenAI's API, offers a modular architecture for NLP application development. It simplifies the integration of pre-trained language models and data storage systems, expediting the development of chatbots and automated writing assistants.

For more information, visit the [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction).

### LangChain Workflow: 
- Step 1: Gather a diverse range of data including job postings, articles, code snippets, tweets, and any new articles you can find.
- Step 2: Vectorize the collected data, create a model, train the model, test the model, and finally deploy the model.
- Step 3: Utilize machine learning to search the dataset using a language model.
- Step 4: Interact with the model by asking questions about the data.

## OpenAI Integration with Node.js

```bash
# Install OpenAI
npm install openai

# Fetch models from OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Organization: org-9PKnn7txwxhPDTNvTe3ZL164"

# Test chat completions with OpenAI
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

## OpenAI and LangChain Integration with Python

```bash
# Install OpenAI
pip install openai
```

> Navigate to `/account/api-keys`: [Your API Key-here](https://platform.openai.com/account/api-keys)

```bash
# Navigate to your project directory
cd your_project_directory
# Create a new .env file
touch .env
# Open the .env file
code .env
```

```
# filename: .env
# This file contains environment variables for the OpenAI API key.

OPENAI_API_KEY=YOUR_API_KEY_HERE
```

> Replace YOUR_API_KEY_HERE with your actual API key from OpenAI.

> Save and close the file.

```bash
# Source the .env file
source .env
# Print the first 5 characters of the API key to verify it's loaded
echo "OPENAI_API_KEY=${OPENAI_API_KEY:0:5}..."
# echo $OPENAI_API_KEY
```

## Execution

```bash
# Activate the langchain environment
conda activate langchain
# Run the my_openai.py script
python src/my_openai.py
# Run the llm_example.py script
python src/llm_example.py
# Run the langchain_examples.py script
python src/langchain_examples.py
```



## Acknowledgements

- LangChain is inspired by [Hugging Face](https://huggingface.co/), [OpenAI](https://openai.com/), and [GPT-3](https://openai.com/blog/gpt-3-apps/).
- OpenAI's [API](https://beta.openai.com/docs/api-reference/introduction) is used to generate text.

---

## Building Language Model Applications with LLMs

```python
# filename: openai_llm.py
# Import OpenAI from langchain.llms
from langchain.llms import OpenAI
# Initialize OpenAI with model name and parameters
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

# Generate a joke using the language model
llm("Tell me a joke")
# '\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'

# Generate multiple outputs using the language model
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

# Print the length of the generations
len(llm_result.generations)
30
# Print the first generation
llm_result.generations[0]
```

## Running the Language Model Application

```bash
# Run the openai_llm.py script
python openai_llm.py
```

> Alternatively, use Hugging Face

```bash
# Install transformers
pip install transformers
```

---

## Introduction to "modular-abstraction/chains"

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
```

## Running the Chain

```python
# Import LLMChain
from langchain.chains import LLMChain
# Initialize LLMChain with llm and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Execute the chain by specifying only the input variable.
print(chain.run("colorful socks"))
```

## Using Chat Models

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
            template="What is a good name for a company that makes {product}?",
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

## Creating a Second Chain

```python
# Initialize a second PromptTemplate
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Compose a catchphrase for the following company: {company_name}",
)
# Initialize a second LLMChain with llm and the second prompt
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```

## Using SimpleSequentialChain

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

## Defining a New Chain Class: ConcatenateChain

```python
# Import LLMChain, Chain, Dict, and List
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List


# Define a new chain class: ConcatenateChain
class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}
```

## Using ConcatenateChain

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
# References

Here are some resources that have been invaluable in the development of LangChain:

- [LangChain Quickstart Guide](https://python.langchain.com/docs/get_started/quickstart)
- [OpenAI API Documentation](https://beta.openai.com/docs/api-reference/introduction)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [GPT-3 Creative Applications](https://openai.com/blog/gpt-3-apps/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
