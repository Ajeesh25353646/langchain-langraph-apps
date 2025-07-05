import os
from langchain_ollama import OllamaLLM
from typing import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import streamlit as st

# Importing a model locally
model = OllamaLLM(model="llama3.1:8b")

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define chatbot Node
def chatbot(state: State):
    return {"messages": model.invoke(state["messages"])}


# Create the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Define the system prompt
sys_message = SystemMessage(content="You are a helpful assistant")

# Streamlit Interface (App building)
st.title("AI Chatbot without memory")
st.subheader("What would you like to know today?")

user_input = st.text_area("Write something")
# user_input = st.text_input("Write something")

if st.button("Send") or user_input:
    if user_input:
        with st.spinner("Thinking....."):  # Displays model is thinking
            messages = [sys_message, HumanMessage(content=user_input)]
            response = graph.invoke({"messages": messages})

            # Extract the true content of answer
            answer = response["messages"][-1].content
            
            # Display the Ai response
            st.write("**AI:**")
            st.write(answer)

    else:
        st.warning("Please enter a message first")

# display footer
st.markdown("Powered by Ollama")