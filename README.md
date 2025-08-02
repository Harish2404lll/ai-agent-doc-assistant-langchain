# AI Agent-Based Document Assistant using LangChain

## Abstract:

This project demonstrates a LangChain-built autonomous agent with AI capabilities that can read documents and utilize reasoning to respond to user inquiries.  It shows how intelligent agents may integrate reasoning stages with tool usage to complete difficult tasks by interacting with various tools, such as web search, document retrieval, and calculation, using a ReAct (Reasoning and Acting) agent.  This serves as the foundation for developing increasingly complex AI task-solving systems that can interact with external APIs and multi-modal inputs.

## Program: 

```

from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=4) #increased number of results
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """
You are an intelligent and efficient research assistant integrated with a web search tool (TavilySearchResults).
Your goal is to find accurate, up-to-date information to answer complex, multi-step user queries.

Guidelines:
1. You can use the search tool to find information on the internet.
2. You are allowed to make multiple calls, either simultaneously or sequentially.
3. Do not hallucinate. If uncertain, search first before answering.
4. Always explain your reasoning if the question requires multiple steps.
5. Be concise, factual, and structured in your answers.

Output Format:
- First, clearly state the answer.
- Then briefly explain how you found it (if needed).
- Cite the source if available from the tool result.

Example:
Q: Who won the Super Bowl in 2024, and what is the GDP of the winner’s home state?

A:
1. The Kansas City Chiefs won the Super Bowl in 2024.
2. Their home state is Missouri.
3. The GDP of Missouri is approximately $390 billion (as of 2023).

(Source: [URL or summary])
"""
model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
abot = Agent(model, [tool], system=prompt)

from IPython.display import Image

Image(abot.graph.get_graph().draw_png())

messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})
print("HARISH G")
print("212222243001")

result

result['messages'][-1].content

messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})

result['messages'][-1].content

# A new, complex query to leverage the detailed prompt.
# This requires the agent to find a capital, then a person, then a population number.
query = """What is the capital of Australia? Who is the current prime minister of that country, and what is the approximate population of the capital city? Answer each question clearly."""
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})

print(result['messages'][-1].content)
```
## OUTPUT: 
<img width="966" height="206" alt="image" src="https://github.com/user-attachments/assets/9b4ef6a3-4ea7-4121-be58-80af2faad4f6" />

