# Claude LangChain Example App

A comprehensive example demonstrating how to use LangChain with Anthropic's Claude models.

## What This Example Shows

This script ([claude_example.py](claude_example.py)) demonstrates 7 different ways to interact with Claude using LangChain:

1. **Basic Chat Completion** - Simple question and answer
2. **Streaming Response** - Real-time token-by-token output
3. **Prompt Templates** - Reusable prompts with variables
4. **Conversation Memory** - Multi-turn conversations
5. **Function/Tool Calling** - Claude calling external tools
6. **Structured Output** - Parsing responses into Pydantic models
7. **Multi-turn Reasoning** - Complex logical reasoning

## Prerequisites

### 1. Install Dependencies

```bash
pip install langchain-anthropic python-dotenv pydantic
```

### 2. Get API Key

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Get your API key from the dashboard
3. Add to your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Running the Example

```bash
python claude_example.py
```

## Model Availability

The script automatically detects which Claude model you have access to:

| Model | Tier | Features |
|-------|------|----------|
| Claude 3.5 Sonnet | Paid | Most capable, best for complex tasks |
| Claude 3 Haiku | Free | Fast, cost-effective, good for simple tasks |

**Note:** Your current API key has access to **Claude 3 Haiku**.

To use Claude 3.5 Sonnet, you need to:
1. Upgrade your API tier at [Anthropic Console](https://console.anthropic.com/)
2. The script will automatically use it once available

## Example Output

```
🔍 Finding available Claude model...
✓ Using model: claude-3-haiku-20240307

================================================================================
Example 1: Basic Chat Completion
================================================================================

📤 Sending: 'What is LangChain? Explain in 2 sentences.'

📥 Claude responds:
LangChain is a framework for building applications with large language models...
```

## Key Code Patterns

### Basic Usage
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?")
])
print(response.content)
```

### Streaming
```python
for chunk in llm.stream([HumanMessage(content="Count to 5")]):
    print(chunk.content, end="", flush=True)
```

### Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {expertise}."),
    ("human", "{question}")
])

chain = prompt | llm
response = chain.invoke({"expertise": "Python", "question": "What is Python?"})
```

### Tool Calling
```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    param: str = Field(description="Parameter description")

@tool(args_schema=ToolInput)
def my_tool(param: str) -> str:
    """Tool description for the LLM"""
    return f"Result: {param}"

llm_with_tools = llm.bind_tools([my_tool])
response = llm_with_tools.invoke([HumanMessage(content="Use the tool")])

if response.tool_calls:
    result = my_tool.invoke(response.tool_calls[0]['args'])
    print(result)
```

## Integration with Your RAG App

To use Claude in your RAG application:

### 1. Update Model Configuration

The example shows how to detect available models. Apply this pattern to your `rag_setup.py` and `rag_app.py`:

```python
# Test which model is available
CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307"
]

for model in CLAUDE_MODELS:
    try:
        test_llm = ChatAnthropic(model=model, max_tokens=5)
        test_llm.invoke([HumanMessage(content="Hi")])
        MODEL_CLAUDE = model
        break
    except:
        continue
```

### 2. Update Vectorstore

Since model names changed, rebuild your vectorstore:

```bash
# Build vectorstore for the model you have access to
python rag_setup.py claude-3-haiku-20240307
```

### 3. Update .env

```bash
MODEL_DEFAULT="claude-3-haiku-20240307"
```

### 4. Run Your App

```bash
streamlit run rag_app.py
```

## Troubleshooting

### 404 Model Not Found Error

**Problem:** `Error code: 404 - model: claude-3-5-sonnet-20240620`

**Solution:** Your API key doesn't have access to that model. The script will automatically fall back to Claude 3 Haiku if it's available.

### API Key Not Set

**Problem:** `ANTHROPIC_API_KEY not set`

**Solution:**
```bash
# Add to .env file
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' >> .env

# Or export in terminal
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Rate Limiting

**Problem:** `Error code: 429 - rate_limit_error`

**Solution:**
- Free tier has rate limits
- Wait a moment between requests
- Consider upgrading your API tier

## Differences Between Models

| Feature | Claude 3.5 Sonnet | Claude 3 Haiku |
|---------|-------------------|----------------|
| Speed | Moderate | Very Fast |
| Cost | Higher | Lower |
| Context Window | 200K tokens | 200K tokens |
| Best For | Complex reasoning | Quick responses |
| API Tier | Paid | Free + Paid |

## Next Steps

1. **Run the example**: `python claude_example.py`
2. **Experiment**: Modify the examples to test different prompts
3. **Integrate**: Apply patterns to your RAG application
4. **Upgrade**: Consider upgrading to Claude 3.5 Sonnet for production

## Resources

- [LangChain Anthropic Docs](https://python.langchain.com/docs/integrations/chat/anthropic)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Claude Model Cards](https://docs.anthropic.com/en/docs/about-claude/models)
- [Anthropic Console](https://console.anthropic.com/)

## License

This example is part of the InsightForge RAG Agent project.
