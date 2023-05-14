LangChain 🦜 🔗

LangChain uses OpenAI's API and a modular architecture that enables developers to easily combine different components, such as pre-trained language models and data storage systems. With LangChain, you can build chatbots, automated writing assistants, and other natural language processing applications with ease.

## LangChain Workflow: Collect, Vectorize, Model, Search

- Step 1: Collect a large amount of job data, articles, code, and tweets, as well as any new articles you can find.
- Step 2: Vectorize the data, create a model, train the model, test the model, and then deploy the model.
- Step 3: Use machine learning to search the data using a language model.
- Step 4: Ask the model questions about the data.

# Start OpenAI with Node.js

```sh

npm install openai

curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Organization: org-9PKnn7txwxhPDTNvTe3ZL164"

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

# Start OpenAi and LangChain with Python

```sh
pip install openai
```

> Link to `/account/api-keys`: [Your API Key-here](https://platform.openai.com/account/api-keys)

```sh
cd your_project_directory
touch .env
code .env
```

```
# filename: .env
# This file contains environment variables for the OpenAI API key.

OPENAI_API_KEY=YOUR_API_KEY_HERE
```

> Replace your_api_key_here with your actual API key from OpenAI.

> Save and close the file.

```sh
source .env
echo "OPENAI_API_KEY=${OPENAI_API_KEY:0:5}..."
# echo $OPENAI_API_KEY
```

# Run

```sh
conda activate langchain
python src/my_openai.py
python src/llm_example.py
```

### The End

# Acknowledgements

- LangChain is inspired by [Hugging Face](https://huggingface.co/), [OpenAI](https://openai.com/), and [GPT-3](https://openai.com/blog/gpt-3-apps/).
- OpenAI's [API](https://beta.openai.com/docs/api-reference/introduction) is used to generate text.

---

wip...

# Building a Language Model Application: LLMs

```python
# filename: openai_llm.py
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

llm("Tell me a joke")
# '\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

len(llm_result.generations)
30
llm_result.generations[0]
```

```sh
python openai_llm.py
```

> or, Hugging Face

```sh
pip install transformers
```

---

# Getting Started with "modular-abstraction/chains"

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))
```

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run("colorful socks"))
```

```python
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```

```python
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)
```

```python
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List


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

```python
prompt_1 = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["product"],
    template="What is a good slogan for a company that makes {product}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.run("colorful socks")
print(f"Concatenated output:\n{concat_output}")
```
