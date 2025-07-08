# PDF RAG Bot: Advanced Question-Answering for Your Documents

This project is an Streamlit application that allows you to ask questions from your PDF documents. It leverages Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based on the content of your uploaded files. It additionally offers multiple Retrieval options, this give you a chance to cross validate and verify the correctness of your answer.

## ⭐ Key Features


### 1. **Multiple Retrieval Strategies**
The core of any RAG system is its ability to find the right information. This application lets you choose from three distinct and powerful retrieval methods from the LangChain library:

*   **MultiQueryRetriever**: Perfect for complex questions. It uses the LLM to automatically generate multiple variations of your original question from different perspectives, casting a wider net to find the most relevant document chunks.
*   **ContextualCompressionRetriever**: Focuses on quality over quantity. It initially fetches a larger number of documents and then uses the LLM to filter out and discard any that aren't directly relevant to your question. This cleans the context, leading to more precise answers.
*   **VectorStoreRetriever**: The classic, efficient semantic search. It finds the document chunks that are most similar to your question based on their vector embeddings.

### 2. **Local & Private Embeddings**
Privacy and cost are critical. Instead of relying on a paid, external API to create vector embeddings, this application uses a powerful open-source model (`Qwen/Qwen3-Embedding-0.6B`) run locally from Hugging Face. In a very similar way, the main model can also be run offline without the need of APIs to further strengthen confidentiality.

*   **Why this is good:** Your document's content is never sent to a third-party service for embedding. This enhances privacy and makes the application suitable for sensitive or confidential documents. It also significantly reduces operational costs.

### 3. **High-Fidelity Answers with Google Gemini**
The final answer generation is powered by Google's `gemini-2.5-flash` model. A carefully engineered prompt instructs the model to answer **only** from the information provided in the retrieved document excerpts.

*   **Why this is good:** This prompt dramatically reduces the risk of the model "hallucinating" or making up information. If the answer isn't in the document, the application will tell you, ensuring a high degree of trust and reliability.

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
This version of pdf_rag is meant for text heavy PDFs. For PDFs having lot of images, tables. I have designed an advanced multimodal RAG system.
