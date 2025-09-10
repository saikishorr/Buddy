from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the state
class State(TypedDict): 
    messages: list
 
# Use Ollama with gemma:2b
ollama_model = ChatOllama(model="gemma:2b")

# Define LangGraph workflow
workflow = StateGraph(State)

def chatbot_node(state: State):
    response = ollama_model.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Compile graph
app = workflow.compile()

# Test the chatbot
result = app.invoke({"messages": ["what is the fullform of SIP?"]})
# print(result)
 
# Get the last AI message
ai_message = result["messages"][-1]
print(ai_message.content)
 
