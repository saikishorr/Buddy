import sqlite3
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
 
class State(TypedDict):
    messages: list
 
llm = ChatOllama(model="gemma:2b")
app = (
    StateGraph(State)
    .add_node("chatbot", lambda s: {"messages": s["messages"] + [llm.invoke(s["messages"])]})
    .set_entry_point("chatbot")
    .add_edge("chatbot", END)
    .compile()
)
 
def save_chat(q, a):
    with sqlite3.connect("chat_history.db") as conn:
        conn.execute("INSERT INTO chat_logs (question, answer) VALUES (?, ?)", (q, a))
        conn.commit()
 
question = input("Enter your question: ")
result = app.invoke({"messages": [question]})
answer = result["messages"][-1].content
 
print(f"👤 {question}\n🤖 {answer}")
save_chat(question, answer)
print("💾 Chat saved!")
