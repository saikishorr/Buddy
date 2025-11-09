import sqlite3
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state
class State(TypedDict):
    messages: list

ollama_model = ChatOllama(model="gemma:2b")

workflow = StateGraph(State)

def chatbot_node(state: State):
    response = ollama_model.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Function to store Q&A in DB
def save_to_db(question, answer):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_logs (question, answer) VALUES (?, ?)",
        (question, answer)
    )
    conn.commit()
    conn.close()

# --- Take input ---
question = input("Enter your question: ")
print(f"👤 Question: {question}")

# Run workflow
app = workflow.compile()
result = app.invoke({"messages": [question]})
ai_message = result["messages"][-1]
answer = ai_message.content

print(f"🤖 Answer: {answer}")

# --- Save to DB ---
save_to_db(question, answer)
print("💾 Chat saved to database!")


