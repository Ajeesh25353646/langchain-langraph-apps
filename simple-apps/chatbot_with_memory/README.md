# Simple LangGraph Chatbot

This is a simple chatbot application which supports multiturn conversations.

## Usage

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up environment variables:**

    Create a `.env` file in the root directory and add your Google API key:

    ```
    GOOGLE_API_KEY=your_api_key
    ```

3.  **Run the application:**

    ```bash
    streamlit run Simple_chatbot.py
    ```

## Code Overview

The `Simple_chatbot.py` file contains the following:

- **Model Initialization:** Initializes the chat model using.
- **State Definition:** Defines the state of the graph, which holds the messages.
- **Chatbot Node:** Defines the node that interacts with the chat model.
- **Graph Definition:** Creates the graph, adds the chatbot node, and defines the edges.
- **Graph Compilation:** Compiles the graph with a memory saver.
- **Streamlit UI:** Creates a simple Streamlit UI for interacting with the chatbot.
