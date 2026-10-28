import sqlite3
import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

LOGS_DB = "support_logs.db"
LOGS_JSONL = "ticket_logs.jsonl"

def init_db():
    """Initialize the SQLite database for AI logs."""
    conn = sqlite3.connect(LOGS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            timestamp DATETIME,
            intent TEXT,
            priority TEXT,
            sentiment TEXT,
            refund_amount REAL,
            issue_description TEXT,
            resolution TEXT,
            summary TEXT,
            chat_history TEXT,
            full_state_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_ticket_log(state_values: dict, ticket_id: str):
    """Save the ticket data to both SQLite and JSONL."""
    init_db()
    
    timestamp = datetime.now().isoformat()
    
    # 1. Prepare human-readable history
    formatted_history = []
    for msg in state_values.get("messages", []):
        if isinstance(msg, HumanMessage):
            role = "Customer"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = "Other"
        formatted_history.append(f"{role}: {msg.content}")
    
    history_str = "\n".join(formatted_history)
    
    # 2. Prepare serializable state for JSON
    # We strip the complex message objects and replace them with simple strings
    serializable_state = state_values.copy()
    serializable_state["messages"] = formatted_history
    state_json = json.dumps(serializable_state)
    
    # --- Save to SQLite ---
    try:
        conn = sqlite3.connect(LOGS_DB)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ticket_logs (
                ticket_id, timestamp, intent, priority, sentiment, 
                refund_amount, issue_description, resolution, 
                summary, chat_history, full_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            timestamp,
            state_values.get("intent"),
            state_values.get("priority"),
            state_values.get("sentiment"),
            state_values.get("refund_amount"),
            state_values.get("issue_description"),
            state_values.get("resolution"),
            state_values.get("summary"),
            history_str,
            state_json
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to SQLite: {e}")

    # --- Save to JSONL ---
    try:
        log_entry = {
            "ticket_id": ticket_id,
            "timestamp": timestamp,
            "data": serializable_state
        }
        with open(LOGS_JSONL, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error saving to JSONL: {e}")
