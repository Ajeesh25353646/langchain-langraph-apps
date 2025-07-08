# PDF RAG Bot: Advanced Question-Answering for Your Documents

This project is a Streamlit application that allows you to ask questions from your textual PDF documents. It leverages Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based on the content of your uploaded files. It additionally offers multiple Retrieval options, this give you a chance to cross validate and verify the correctness of your answer.

## ⭐ Key Features

### 1. **Confidential Mode**
The application allows users to switch to confidential mode. In this mode, the latest open source model is imported from Ollama and used for the task. Since, the model is local, this way, the highest level of confidentiality of documents is maintained. In naive terms, it doesn't matter if internet is on or off, the model can be effectively used. 


### 2. **Multiple Retrieval Strategies**
The core of any RAG system is its ability to find the right information. This application lets you choose from three distinct and powerful retrieval methods from the LangChain library:

*   **MultiQueryRetriever**: Perfect for complex questions. It uses the LLM to automatically generate multiple variations of your original question from different perspectives, casting a wider net to find the most relevant document chunks.
*   **ContextualCompressionRetriever**: Focuses on quality over quantity. It initially fetches a larger number of documents and then uses the LLM to filter out and discard any that aren't directly relevant to your question. This cleans the context, leading to more precise answers.
*   **VectorStoreRetriever**: The classic, efficient semantic search. It finds the document chunks that are most similar to your question based on their vector embeddings.

### 3. **Local & Private Models + Embeddings**
Privacy and cost are critical. Instead of relying on a paid, external API to create vector embeddings, the embeddings has also been generated locally, using open-source model (`Qwen/Qwen3-Embedding-0.6B`) imported from Hugging Face. These embeddings were selected based on MTEB leaderboard of HuggingFace. 

### 4. **Advanced Caching Mechanisms allowing multiturn conversations**
This application further includes some certain caching mechanisms. Once a file is uploaded, you can keep asking multiple questions and the cached embeddings will be reused, saving a lot of your time.

### 5. **Show more updates**
It shows many updates about the current status for your questioning task.

*   **Why this is good:** Your document's content is never sent to a third-party service for anything. The entire application runs on your own device. This enhances privacy and makes the application suitable for sensitive or confidential documents. It also significantly reduces operational costs as no API costs are involved.

*  Additionally, This prompt was tested extensively and has been found to work really well.it makes sure that, even if the answer isn't in the document, the application will tell you, ensuring a high degree of trust and reliability.

### 4. **Interactive & Easy-to-Use Interface**
Built with Streamlit, the application is incredibly user-friendly. No command-line knowledge is needed. Simply:
1.  Upload your PDF.
2.  Select your preferred retrieval strategy from a dropdown menu.
3.  Ask your question.

## 🚀 Why Use This Application?

*   **Unlock Knowledge**: Go beyond simple keyword searching. Understand the actual meaning and context within your PDFs, from dense academic papers to technical manuals.
*   **Enhance Productivity**: Get answers instantly without having to manually skim through hundreds of pages.
*   **Trustworthy AI**: The combination of advanced retrieval, local embeddings, and strict, context-aware prompting creates a reliable assistant you can count on for accurate information.
*   **Flexible & Experimental**: The ability to switch between different retrieval methods allows you to experiment and find the optimal approach for your specific documents and query types.

## Further Notes
This version of pdf_rag is meant for text heavy PDFs only. For PDFs with many images or tables. I have designed an even more advanced multimodal RAG system suitable for that.Stay tuned for that.

All of these RAG systems can be combined later on to form a full blown RAG app as well.