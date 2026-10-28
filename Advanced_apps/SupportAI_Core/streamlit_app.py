import streamlit as st
from main import app
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from logger import save_ticket_log

# --- Page Config & Styling ---
st.set_page_config(page_title="SupportAI Core", page_icon="🤖", layout="wide")

# Custom CSS for warm amber/copper theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

    /* ── Base ── */
    .stApp {
        background: #F8F4F0;
    }
    .main > div {
        background: #F8F4F0;
    }
    .block-container {
        background: #F8F4F0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #F8F4F0 !important;
        border-right: 1px solid #E0D5C8;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1gv3huu,
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        background: #F8F4F0 !important;
    }
    section[data-testid="stSidebar"] .stTitle {
        color: #B87333 !important;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #E0D5C8 !important;
        margin: 12px 0;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #C9A96E !important;
        letter-spacing: 1px;
        font-size: 11px;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: #FFFFFF;
        border: 1px solid #E0D5C8;
        color: #8B7355;
        border-radius: 10px;
        font-size: 13px;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #D4874A;
        color: #D4874A;
    }

    /* ── Status / st.status ── */
    div[data-testid="stStatusWidget"] {
        background: #F8F4F0 !important;
        border: 1px solid #EDE5DC !important;
        border-radius: 12px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #D4874A !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8B7355 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Containers / Cards ── */
    .stContainer {
        background: #FFFFFF !important;
        border: 1px solid #EDE5DC !important;
        border-radius: 12px !important;
    }
    .st-emotion-cache-1wrcr25 {
        background: #FFFFFF !important;
        border: 1px solid #EDE5DC !important;
        border-radius: 12px !important;
    }

    /* ── Resolution Card ── */
    .res-card {
        background: linear-gradient(135deg, #FDF6EE, #FAEBD7) !important;
        border: 1px solid #E8D5C4 !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        text-align: center !important;
        margin-top: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important;
    }

    /* ── Expander ── */
    .stExpander {
        border: 1px solid #EDE5DC !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #EDE5DC !important;
    }

    /* ── Headers ── */
    h1, h2, h3 {
        color: #1E1E1E !important;
    }
    .stHeader {
        color: #1E1E1E !important;
    }

    /* ── Buttons (general) ── */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stButton button[kind="primary"] {
        background: #D4874A !important;
        color: white !important;
        border: none !important;
    }
    .stButton button[kind="primary"]:hover {
        background: #B87333 !important;
    }

    /* ── Success / Info / Error messages ── */
    .stSuccess {
        background: #EDF5F0 !important;
        border-color: #2D6A4F !important;
        color: #1E1E1E !important;
    }
    .stInfo {
        background: #FFF8F0 !important;
        border-color: #D4874A !important;
        color: #1E1E1E !important;
    }
    .stError {
        background: #FFF5F3 !important;
        border-color: #C85A48 !important;
        color: #C85A48 !important;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background: #F8F4F0 !important;
        border: 1px solid #EDE5DC !important;
        border-radius: 10px !important;
        color: #1E1E1E !important;
    }
    .stTextArea textarea:focus {
        border-color: #D4874A !important;
    }

    /* ── Toast ── */
    .stToast {
        background: #F8F4F0 !important;
        border: 1px solid #EDE5DC !important;
        color: #1E1E1E !important;
    }

    /* ── Input widgets ── */
    .stTextInput input, .stSelectbox, .stRadio {
        background: #FFFFFF !important;
        border: 1px solid #EDE5DC !important;
        border-radius: 10px !important;
        color: #1E1E1E !important;
    }

    /* ── Streamlit background overrides ── */
    #root > div:nth-child(1) > div.withScreencast > div {
        background: #F8F4F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_resolved" not in st.session_state:
    st.session_state.is_resolved = False
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Customer Portal"
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "user_rating" not in st.session_state:
    st.session_state.user_rating = None

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ SupportAI Core")
    
    # View Switcher
    st.session_state.view_mode = st.radio(
        "Interface Mode",
        ["Customer Portal", "Executive Dashboard"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Ticket Details
    st.caption(f"TICKET ID: {st.session_state.thread_id[:8].upper()}")
    
    if st.session_state.is_resolved:
        st.success("STATUS: RESOLVED")
    else:
        st.info("STATUS: ACTIVE")

    st.markdown("---")
    
    # Controls
    if st.button("🔄 Reset Chat", use_container_width=True):
        for key in ["thread_id", "chat_history", "is_resolved", "feedback_submitted", "user_rating"]:
            st.session_state.pop(key, None)
        st.rerun()

# --- 1. Customer Portal View ---
if st.session_state.view_mode == "Customer Portal":
    st.header("👋 Help Center")
    st.caption("Chat with our AI assistant for instant support.")
    
    # Chat Container
    chat_container = st.container()
    
    # Display History
    with chat_container:
        if not st.session_state.chat_history:
            st.chat_message("assistant").write("Hello! How can I help you today?")
            
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

    # --- Resolution & Feedback Flow ---
    if st.session_state.is_resolved:
        with chat_container:
            st.markdown('<div class="res-card">', unsafe_allow_html=True)
            st.subheader("Thank you for contacting our support team today!")
            st.write("We hope your issue was resolved to your satisfaction. Your feedback helps us improve our service.")
            
            if not st.session_state.feedback_submitted:
                col1, col2 = st.columns(2)
                if col1.button("👍 Satisfied", use_container_width=True):
                    st.session_state.user_rating = "thumbs_up"
                    st.session_state.feedback_submitted = True
                    # Log final state
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    state = app.get_state(config)
                    final_data = state.values.copy()
                    final_data["user_rating"] = "thumbs_up"
                    save_ticket_log(final_data, st.session_state.thread_id)
                    st.rerun()
                
                if col2.button("👎 Unsatisfied", use_container_width=True):
                    st.session_state.user_rating = "thumbs_down"
                    st.rerun()

                if st.session_state.user_rating == "thumbs_down":
                    st.divider()
                    feedback_text = st.text_area("We're sorry to hear that. What went wrong?", placeholder="Tell us more about your experience...")
                    if st.button("Submit Feedback", type="primary"):
                        st.session_state.feedback_submitted = True
                        # Log final state
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        state = app.get_state(config)
                        final_data = state.values.copy()
                        final_data["user_rating"] = "thumbs_down"
                        final_data["user_feedback"] = feedback_text
                        save_ticket_log(final_data, st.session_state.thread_id)
                        st.rerun()
            else:
                st.success("✅ Thank you! Your feedback has been recorded.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Active Chat Logic ---
    else:
        # Manual Escalation Option (after 4 turns / 8 messages)
        if len(st.session_state.chat_history) >= 8:
            if st.button("🆘 Talk to a Human Representative", use_container_width=True):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                app.update_state(config, {"intent": "technical_issue"}, as_node="triage")
                st.toast("Transferring you to a human agent...")
                st.rerun()

        # Input Area
        if prompt := st.chat_input("Type your message..."):
            # User Message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                st.chat_message("user").write(prompt)

            # AI Processing — stream node progress + token-by-token response
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response_text = ""
            agent_node_names = ("technical", "inquiry")
            try:
                msg_placeholder = st.chat_message("assistant").empty()

                with st.status("🤔 Analyzing your query...", expanded=False) as status:
                    for chunk in app.stream(
                        {"messages": [HumanMessage(content=prompt)]},
                        config,
                        stream_mode=["updates", "messages"],
                        version="v2",
                    ):
                        if chunk["type"] == "updates":
                            for node_name, node_state in chunk["data"].items():
                                if node_name == "guardrail" and node_state.get("is_blocked"):
                                    status.update(label="⛔ Blocked by safety system", state="error")
                                elif node_name == "triage":
                                    intent = node_state.get("intent", "...")
                                    status.update(label=f"📋 Intent: {intent}", state="running")
                                elif node_name in agent_node_names:
                                    status.update(label="📝 Generating response...", state="running")
                                elif node_name == "off_topic":
                                    status.update(label="🚫 Off-topic query", state="running")
                                elif node_name == "escalation":
                                    status.update(label="🔄 Escalating...", state="running")
                                elif node_name == "respond":
                                    status.update(label="✅ Response ready", state="complete")

                        elif chunk["type"] == "messages":
                            msg, metadata = chunk["data"]
                            if msg.content and metadata.get("langgraph_node") in agent_node_names:
                                response_text += msg.content
                                msg_placeholder.markdown(response_text + "▌")

                if response_text:
                    msg_placeholder.markdown(response_text)
                else:
                    # Fallback for nodes that don't emit LLM tokens (off_topic, escalation)
                    final_state = app.get_state(config)
                    if final_state and "messages" in final_state.values:
                        response_text = final_state.values["messages"][-1].content
                    if response_text:
                        msg_placeholder.markdown(response_text)

                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"System Error: {str(e)}")
            st.rerun()

        # Mark as Resolved Button (Moved to bottom of chat area)
        if len(st.session_state.chat_history) >= 4:
            st.divider()
            if st.button("✅ I'm satisfied, Mark as Resolved", use_container_width=True):
                st.session_state.is_resolved = True
                st.rerun()

# --- 2. Executive Dashboard View ---
else:
    st.header("📊 Supervisor Dashboard")
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        current_state = app.get_state(config)
        state_values = current_state.values if current_state else {}
    except:
        state_values = {}

    if not state_values:
        st.info("No active ticket data found. Start a conversation in the Customer Portal.")
        st.stop()

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.subheader("📝 Live Transcript")
        with st.container(height=500, border=True):
            for msg in state_values.get("messages", []):
                role = "👤 Customer" if isinstance(msg, HumanMessage) else "🤖 AI Agent"
                avatar = "👤" if isinstance(msg, HumanMessage) else "🤖"
                with st.chat_message(role, avatar=avatar):
                    st.write(msg.content)

    with right_col:
        st.subheader("🔍 Insights")
        with st.container(border=True):
            m1, m2 = st.columns(2)
            m1.metric("Intent", state_values.get("intent", "N/A").title())
            m2.metric("Sentiment", state_values.get("sentiment", "N/A").title())
            m3, m4 = st.columns(2)
            m3.metric("Priority", state_values.get("priority", "N/A").title())
            refund_amt = state_values.get("refund_amount", 0)
            m4.metric("Value", f"₹{refund_amt}" if refund_amt else "N/A")

        st.subheader("⚡ Actions & Evaluation")
        
        # Internal Supervisor Notes
        with st.container(border=True):
            st.caption("INTERNAL EVALUATION")
            sup_notes = st.text_area("Supervisor Notes", value=state_values.get("supervisor_notes", ""), placeholder="Rate AI performance or add internal context...")
            if st.button("Save Evaluation"):
                app.update_state(config, {"supervisor_notes": sup_notes})
                st.success("Evaluation saved!")

        # Escalation Handling
        if current_state.next and "human_review" in current_state.next:
            with st.status("🚨 Escalation Pending Review", expanded=True, state="error"):
                st.write("**Reason:**")
                notes = "Requires Manager Approval"
                if current_state.tasks and current_state.tasks[0].interrupts:
                    notes = current_state.tasks[0].interrupts[0].value.get("escalation_notes", notes)
                st.info(notes)
                
                comment = st.text_input("Approval/Rejection Note:", placeholder="e.g., Refund approved per policy.")
                b1, b2 = st.columns(2)
                if b1.button("✅ Approve", use_container_width=True):
                    app.update_state(config, {"is_approved": True, "comment": comment}, as_node="human_review")
                    st.success("Approved!")
                    st.rerun()
                if b2.button("⛔ Reject", use_container_width=True):
                    app.update_state(config, {"is_approved": False, "comment": comment}, as_node="human_review")
                    st.error("Rejected!")
                    st.rerun()
        else:
            with st.container(border=True):
                st.success("✅ System Autonomous")
                st.caption("Agent is handling this ticket.")

        with st.expander("🛠️ Raw State Data"):
            st.json(state_values)
