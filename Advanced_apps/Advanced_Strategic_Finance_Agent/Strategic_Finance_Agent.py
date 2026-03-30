import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def get_api_key(key_name: str, env_var: str) -> str:
    """Get API key from .env or Streamlit secrets."""
    if st.runtime.exists():
        try:
            # Check if key_name is in secrets
            if key_name in st.secrets:
                secret = st.secrets[key_name]
                if isinstance(secret, str):
                    return secret
                # Check for nested keys like 'api_key' or 'google_api_key'
                for k in ["api_key", "google_api_key", key_name]:
                    if k in secret:
                        return secret[k]
        except Exception:
            pass
    return os.getenv(env_var)


# Initialize Model with API Key
google_api_key = get_api_key("google_genai", "GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=google_api_key) if google_api_key else ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


# --- Define Tools ---
tavily_api_key = get_api_key("tavily", "TAVILY_API_KEY")
tavily_tool = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
ddg_tool = DuckDuckGoSearchRun()

# --- Define Agent State ---
class AgentState(TypedDict):
    source_currency: str
    target_currency: str
    amount: float
    business_context: str
    current_rate: str
    market_news: str
    financial_analysis: str
    final_recommendation: str

# --- Define Nodes ---

def currency_data_fetcher(state: AgentState):
    """Fetches the latest real-time exchange rate from a reliable API."""
    api_key = get_api_key("exchangerate_api", "EXCHANGERATE_API_KEY")
    source = state['source_currency'].upper()
    target = state['target_currency'].upper()
    
    if api_key:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{source}/{target}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get("result") == "success":
                rate = data.get("conversion_rate")
                return {"current_rate": f"1 {source} = {rate} {target}"}
        except Exception as e:
            print(f"API Error: {e}")

    # Fallback to search ONLY if API fails
    query = f"current exchange rate {source} to {target} price today"
    try:
        result = tavily_tool.invoke(query)
    except Exception:
        result = ddg_tool.invoke(query)
    
    return {"current_rate": f"[Search Fallback] {result}"}

def financial_news_analyst(state: AgentState):
    """Searches for high-impact financial news and economic events."""
    queries = [
        f"{state['source_currency']} {state['target_currency']} forecast next 30 days",
        f"central bank interest rate decisions {state['source_currency']} {state['target_currency']} recent news",
        f"major economic events affecting {state['target_currency']} this week"
    ]
    
    news_summary = ""
    for q in queries:
        try:
            res = tavily_tool.invoke(q)
            news_summary += f"Query: {q}\nResult: {res}\n\n"
        except Exception:
            pass
            
    return {"market_news": news_summary}

def senior_strategist(state: AgentState):
    """Synthesizes data and news into a strategic recommendation."""
    
    prompt = f"""
    You are a Chief Financial Officer (CFO) level advisor.
    
    **Context:**
    User wants to convert {state['amount']} {state['source_currency']} to {state['target_currency']}.
    Business Scenario: {state['business_context']}
    
    **Market Data:**
    {state['current_rate']}
    
    **News & Drivers:**
    {state['market_news']}
    
    **Instructions:**
    1. EXTRACT the likely current exchange rate from the data.
    2. ANALYZE the short-term trend (Bullish/Bearish) based on the news (e.g., Central Bank speeches, inflation data).
    3. ASSESS VOLATILITY: Is the market stable or expecting a shock?
    4. FORMULATE A STRATEGY:
       - Should they execute the full amount now?
       - Should they wait?
       - Should they hedge (split the transaction)?
    
    **Output Style:**
    Professional, concise, bullet-proof logic. Start with the "Executive Decision" first.
    """
    response = model.invoke([HumanMessage(content=prompt)])

    content = response.content
    if isinstance(content, list):
        # Extract and join all text blocks from the response
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = content  # Already a plain string

    return {"final_recommendation": text}    


# --- Build Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("market_research", currency_data_fetcher)
workflow.add_node("news_intel", financial_news_analyst)
workflow.add_node("cfo_advisor", senior_strategist)

# Set Entry Point
workflow.set_entry_point("market_research")

# Define Edges (Linear flow for high reliability)
workflow.add_edge("market_research", "news_intel")
workflow.add_edge("news_intel", "cfo_advisor")
workflow.add_edge("cfo_advisor", END)

# Compile
app = workflow.compile()

# --- Streamlit UI ---
# Check if running within a streamlit context to avoid errors in tests
if st.runtime.exists():
    st.set_page_config(page_title="Strategic FX Advisor", page_icon="📈")

    st.title("📈 Strategic Forex Intelligence Agent")
    st.markdown("A CFO-level AI agent that analyzes market conditions to recommend the *optimal* time to move capital.")

    with st.sidebar:
        st.header("Transaction Details")
        source_curr = st.text_input("Source Currency (e.g., USD)", "USD")
        target_curr = st.text_input("Target Currency (e.g., EUR)", "EUR")
        amount = st.number_input("Amount", min_value=100.0, value=1000000.0, step=1000.0)
        context = st.text_area("Business Context", "We need to pay a supplier invoice in 15 days. We can pay anytime between now and then.")
        
        run_btn = st.button("Generate Strategy")

    if run_btn:
        # Check for required keys before proceeding
        if not google_api_key:
            st.error("⚠️ Google GenAI API Key is missing. Please check your .env or Streamlit Secrets.")
            st.stop()
        
        inputs = {
            "source_currency": source_curr,
            "target_currency": target_curr,
            "amount": amount,
            "business_context": context,
            "current_rate": "",
            "market_news": "",
            "financial_analysis": "",
            "final_recommendation": ""
        }
        
        with st.status("🤖 Running FX Intelligence Agent...", expanded=True) as status:
            st.write("🔍 Agent 1: Fetching real-time exchange rates...")
            st.write("📰 Agent 2: Analyzing market news & central bank signals...")
            st.write("🧠 Agent 3: CFO Strategist formulating recommendation...")
            result = app.invoke(inputs)
            status.update(label="✅ Analysis complete!", state="complete")
        
        # Display Results
        st.divider()
        st.subheader("Executive Strategy Report")
        st.write(result["final_recommendation"])
        
        with st.expander("See Underlying Data Sources"):
            st.write("### Raw Market Data")
            st.write(result["current_rate"])
            st.write("### Market Intelligence")
            st.write(result["market_news"])
