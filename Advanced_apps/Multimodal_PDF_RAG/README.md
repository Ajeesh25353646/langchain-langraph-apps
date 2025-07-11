# PDF RAG with Advanced Multimodal Support

This project is an intelligent, multimodal assistant that allows you to have a conversation with your PDF documents. It goes beyond simple text-based Q&A by deeply understanding the content of images, tables, and figures within your files, providing you with more accurate and comprehensive answers by further using various RAG techniques. Not only it provides the answer but it also provides some refrence regarding where the answer comes from, so you can quickly look up the document if you wish to.

## Key Features & Capabilities

This PDF Tool can do everything you would wanna do with a PDF:

*   **True Multimodal Understanding:** The RAG application possesses enhanced multimodal capabilities that most typical models lack. It doesn't just read the text; it *sees* the images and tables in your document. It generates rich descriptions of them, ensuring you get answers that rely on the complete context of the document.
*   **Confidential Mode:** Worried about data privacy? You can process highly sensitive documents with complete peace of mind. Simply toggle on "Confidential Mode" to use a powerful local model (`gemma3:4b`), ensuring your data never leaves your machine.
*   **Instant Image Extraction:** Ever needed to grab a chart for a presentation from a PDF? This tool lets you view and download any image or table from your document with a single click and those images are really good quality ones!
*   **Accurate, Referenced Answers:** Get clear answers to your questions, alongwith page number references. This makes it incredibly easy to go back and verify the source of the information in the original document.
*   **Blazing-Fast Performance:** Initally the PDF processing can take time but once it answers the first question. A smart caching mechanism is used i.e. your document is processed only once. After the initial setup, you can ask as many follow-up questions as you want and you will recieve answers within seconds.
*   **Advanced Search Options:** The tool offers multiple retrieval strategies (`MultiQueryRetriever`, `ContextualCompressionRetriever`, etc.). This allows you to experiment and find the best way to search your documents, ensuring you get the most stable and relevant results every time.
*   **Multiple model supports:** Under confidential toggle, when its off, a gemini model is used but once, you turn on the toggle, the model uses a local model useful for condifential documents.

## How It Works

The system uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline:

1.  **Document Loading:** When you upload a PDF, it's processed using `unstructured` to extract text, tables, and images, preserving the document's structure.
2.  **Image Captioning:** In multimodal mode, each extracted image is passed to a vision-language model (either Gemini or a local Gemma model) which generates a detailed, accurate description of the visual content.
3.  **Content Injection:** These generated descriptions are seamlessly injected back into the document's text, replacing the original images. Now, the document is purely text-based, but it contains rich descriptions of its original visual elements.
4.  **Vectorization:** The entire document, now enriched with image descriptions, is split into chunks and embedded into a `FAISS` vector store using `Qwen/Qwen3-Embedding-0.6B` embeddings.
5.  **Retrieval & Generation:** When you ask a question, the system retrieves the most relevant chunks from the vector store and feeds them, along with your question, to the chosen language model (Gemini or local Gemma) to generate a final, context-aware answer.
