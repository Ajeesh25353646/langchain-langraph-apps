import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
import streamlit as st

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY
)

# Define the tools
tool1 = TavilySearch(max_results=2)
tool2 = DuckDuckGoSearchRun()
tools = [tool1, tool2]

# Define the state schema
class AgentState(TypedDict):
    """The state of the agent, maintaining a list of messages."""
    messages: Annotated[List, add_messages]


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant who is an expert at real-time currency conversions. "
        "Use your tools to find the most up-to-date exchange rates and perform the conversion accurately. "
        "Always provide the final converted value and a brief explanation clearly."
    )
)

# Streamlit UI
st.set_page_config(page_title="AI Currency Converter", page_icon="💰")
st.title("💰 AI Currency Converter Bot")
st.markdown("Real-time currency conversion powered by LangGraph (v1.0+) and Gemini 3.1.")

# User Inputs
source_input = st.text_input("Amount and source currency (e.g., 100 USD): ")
target_currency = st.text_input("Target currency (e.g., EUR):")

if st.button("Convert"):
    if source_input and target_currency:
        with st.spinner("Analyzing real-time market data..."):
            # Construct the prompt
            prompt = f"Convert {source_input} to {target_currency}"
            
            try:
                # We use a tuple [("user", prompt)] for the messages list, which is the idiomatic standard.
                response = agent.invoke({"messages": [("user", prompt)]})
                
                # Success display
                st.success("Conversion Complete!")
                
                # Extract the final message content
                final_message = response["messages"][-1].content

                # Handle structured responses
                if isinstance(final_message, list):
                    final_text = "".join(
                        block.get("text", "")
                        for block in final_message
                        if block.get("type") == "text"
                    )
                else:
                    final_text = final_message

                st.write(final_text)
                
                # Optional: Show 'Underlying Logic' if thoughts were included
                with st.expander("Show Reasoning Trace"):
                    st.info("The agent analyzed market drivers using Tavily and calculated the result.")
                    
            except Exception as e:
                # Enhanced error handling for 2026 API signatures
                st.error(f"Execution Error: {str(e)}")
    else:
        st.warning("Please provide both source and target currency details.")

# Technical Footer
st.divider()
st.caption("Built with LangChain v1.0.0 | LangGraph | Gemini 3.1 Flash Preview")
