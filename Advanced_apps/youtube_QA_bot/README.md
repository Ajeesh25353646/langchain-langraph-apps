# YouTube Q&A Bot

A lot of times, I had to watch super long videos just to discover that video actually never answered my question. Identifying this problem, I have created this app, you can ask questions about any YouTube video. It uses Google's Gemini model for answering questions based on the youtube video's transcript. It additionally offers different retrieval strategies for fetching relevant information from the video transcript. This can be a great way to cross check the accuracy of the answer. Additionally, the code can be modified to use proxies incase youtube starts blocking the requests due to some reason. 

## Features

*   **YouTube Transcript Retrieval**: Automatically fetches the transcript of a given YouTube video using the provided URL.
*   **Question Answering**: Answers user questions based on the content of the video transcript and provide timestamp referrences alongwith its answer. if the answer is not found in the video, it just replies "not found" saving your precious time. 
*   **Multiple Retriever Options**: Choose between `MultiQueryRetriever`, `VectorStoreRetriever`, and `ContextualCompressionRetriever` to experiment and check the consistency of the answer as consistent answers indicates that more likely a good answer.
*   **Local Embeddings**: Uses `Qwen/Qwen3-Embedding-0.6B` for local embeddings, eliminating the need for external embedding APIs. The choice of embeddings are based on MTEB rankings and these embeddings have proved themselves in terms of both speed and performance.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ajeesh25353646/langchain-langgraph-apps.git
cd langchain-langgraph-apps/Advanced_apps/youtube_QA_bot
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Google API Key

Obtain a Google API Key from the [Google AI Studio](https://aistudio.google.com/app/apikey). Google's model has been used since, it has been seen that gemini performs slightly better on youtube videos as they both belong to the same company.

Create a `.env` file in the `youtube_QA_bot` directory and add your API key:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## How to Run

1.  Navigate to the `youtube_QA_bot` directory in your terminal.
2.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your web browser.

## Usage

1.  **Enter YouTube Video URL**: Paste the URL of the YouTube video you want to query into the "Enter the Video URL" text box.
2.  **Ask a Question**: Type your question related to the video content into the "What question would you like to Ask?" text box.
3.  **Choose a Retriever**: Select your preferred retriever strategy from the dropdown menu.
4.  **Generate Answer**: Click the "Generate Answer" button to get the answer based on the video transcript.

## Example

**Video URL**: `https://www.youtube.com/watch?v=your_video_id` (Replace with an actual YouTube video URL)
**Question**: "What are the main topics discussed in the video?"

The bot will then process the transcript and provide an answer.
