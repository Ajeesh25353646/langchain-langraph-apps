import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core import memory
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from typing import Annotated
from typing_extensions import TypedDict
import streamlit as st

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Define the model
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    )


# Define the state that holds messages
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the chatbot node
def chatbot_node(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the graph
graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile and build the graph
memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)

# Run the graph
st.title("LangGraph Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.write(f"**You ({i+1}):** {user_msg}")
        st.write(f"**Bot ({i+1}):** {bot_msg}")
        st.write("---")

input_message = st.text_input("Enter your message:")

if st.button("Generate Answer") and input_message:
    config = {"configurable": {"thread_id": "1"}}
    response = compiled_graph.invoke({"messages": [{"role": "user", "content": input_message}]}, config)
    # st.write(response["messages"][-1].content)

     # Add to chat history
    st.session_state.chat_history.append((input_message, response["messages"][-1].content))
    
    # Display current response
    st.success(f"**Bot Response:** {response}")
    
    # Clear the input by rerunning the app
    st.rerun()

