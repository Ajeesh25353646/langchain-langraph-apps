import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import streamlit as st

# Load environment variables
load_dotenv()
my_google_api_key = os.getenv("GOOGLE_API_KEY")

# Checking if key is present or not
if not my_google_api_key:
    print("GOOGLE_API_KEY secret not found or not attached!")
else:
    # Checking model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=my_google_api_key)
    print("Model loaded - API is working")


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


# Main loop to get user input
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting chatbot.")
#         break
#     try:
#         response = graph.invoke({"messages": [sys_message, HumanMessage(content=user_input)]})
#         print(f"AI: {response['messages'][-1].content}")
#     except Exception as e:
#         print(f"An error occurred: {e}")


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
st.markdown("Powered by Gemini")