# 🦜🔗 LangChain - Comprehensive Guide

## Overview

LangChain has expanded its NLP application development capabilities by integrating with the Multion API. This integration enhances the functionality of LangChain when used in conjunction with the existing OpenAI API, particularly for the development of chatbots and automated writing assistants.

For detailed information, visit: [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction).

![LangChain Architecture Diagram](img/langchain_art.png)

## Workflow in LangChain

1. **Data Acquisition**: Collect a variety of data types.
2. **Data Processing**: Process the data through vectorization, model development, training, testing, and deployment.
3. **Machine Learning Application**: Employ machine learning techniques to query datasets.
4. **Model Interaction**: Interact with the model by submitting queries and receiving responses.
5. **Receive and Utilize Outputs**: Implement the outputs obtained from the model in your applications.

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

```python
# Language model application example
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

# Generate content
llm("Tell me a joke")
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

# Display outputs
len(llm_result.generations)
llm_result.generations[0]
```

### Running Language Model Applications

```bash
# Execute the language model application script
python3 openai_llm.py
```

### Advanced Usage: Modular Abstraction and Chains

```python
# Modular abstraction using chains
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Initialize OpenAI and PromptTemplate
llm = OpenAI(temperature=0
```

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [OpenAI](https://openai.com/)
- [GPT-3](https://openai.com/blog/gpt-3-apps/)
- [Multion API](https://api.multion.ai/)
- [OpenAI API](https://beta.openai.com/docs/api-reference/introduction)