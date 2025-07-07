# AI Chatbot without Memory (Streamlit App)

This project provides a simple, chatbot built using **Python**, **Streamlit**, and **LangGraph**. The chatbot does **not retain memory** of past conversations, treating each query as a new interaction.

Two versions of the chatbot are made:

* `chatbot_without_memory.py`: Uses the **Google Gemini API** for responses.
* `chatbot_local.py`: Runs a **local large language model** using **Ollama**. Such models can be extremely useful to deal with any confidential information as they are run offline.

---

## Features

* Clean, minimal UI built with **Streamlit**
* Stateless conversation flow managed via **LangGraph**
* Support for both **cloud-based (Gemini)** and **local (Ollama)** LLMs

---

## How to Run

Use `streamlit run` to launch either version of the chatbot:

* **To run the Google Gemini chatbot:**

  ```bash
  streamlit run chatbot_without_memory.py
  ```

* **To run the local Ollama chatbot:**

  ```bash
  streamlit run chatbot_local.py
  ```

---

## File Descriptions

* `chatbot_without_memory.py`: Main app for Gemini-based chatbot
* `chatbot_local.py`: Main app for Ollama-based chatbot
* `README.md`: This file — contains setup and usage instructions

---

## Environment Configuration

### For the Google Gemini version (`chatbot_without_memory.py`)

Create a `.env` file in the project root and add your Gemini API key:

```env
GOOGLE_API_KEY="your_google_api_key_here"
```

### For the local Ollama version (`chatbot_local.py`)

Make sure **Ollama** is installed and running.
Download it from: [https://ollama.com](https://ollama.com)

Then pull the required model:

```bash
ollama pull llama3.1:8b
```


