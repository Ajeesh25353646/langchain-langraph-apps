import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import streamlit as st


#############  Getting the api key from the .env file-----------------------------------------------------------
load_dotenv()
my_google_api_key = os.getenv("GOOGLE_API_KEY")

# Checking if key is present or not
if not my_google_api_key:
    print("GOOGLE_API_KEY secret not found or not attached!")
else:
    # Checking model
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=my_google_api_key)
    print("Model loaded - API is working")


# Importing the local modal
@st.cache_resource(show_spinner="loaing local model...")
def cache_local_model():
    """Loads Model Locally"""
    gemma_modal = OllamaLLM(model="gemma3:4b")
    return gemma_modal


@st.cache_resource(show_spinner="preprocessing..")
def create_vector_store(file_bytes):
    """caches the entire file loading process - splitting, embedding"""
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    
    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


######### Loading the pdf  ---------------------------------------------------------------------
# Define the title for my bot
st.title("PDF RAG bot")

#################### Setting up the App Interface -------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Document Retrieval query
user_question = st.text_input("What question would you like to Ask?")


# Retriever choice
retriever_choice = st.selectbox(
    "Choose a retriever:",
    ("MultiQueryRetriever", "VectorStoreRetriever", "ContextualCompressionRetriever")
)

# Toggle to choose between local model or gemini
local_model = st.toggle("confidential documents")


# --- Button --------------------------------------------------
if st.button("Generate Answer"):
    # chooses between gemini or local model
    if local_model:
        # st.info("Using local model...")
        model = cache_local_model()
    else:
        # st.info("Using Gemini model...")
        model = gemini_model
    
    if uploaded_file is not None and user_question is not None:
            file_bytes = uploaded_file.getvalue()
            vector_store = create_vector_store(file_bytes)

            if vector_store is None:
                st.warning("No text could be extracted from the PDF. Please try another file.")
                st.stop()
            
            print("Vector store created/retrieved from cache.")

            # --- Retriever Selection ---
            st.write(f"Using {retriever_choice}...")
            with st.spinner("Retrieving useful information...."):
                if retriever_choice == "MultiQueryRetriever":
                    retriever = MultiQueryRetriever.from_llm(
                        retriever=vector_store.as_retriever(search_kwargs={"k":5}), llm=model
                    )
                elif retriever_choice == "VectorStoreRetriever":
                    retriever = vector_store.as_retriever(search_kwargs={"k":5})
                elif retriever_choice == "ContextualCompressionRetriever":
                    # I'll get more documents (e.g. 10) and let the compressor filter them
                    base_retriever = vector_store.as_retriever(search_kwargs={"k":10})
                    compressor = LLMChainFilter.from_llm(model)
                    retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=base_retriever
                    )


    ############# User query Prompt Template-----------------------------------------------------------------------------------------------------------------------
            prompt = PromptTemplate(
                template = '''
            You are an expert assistant helping answer questions about a PDF document, using only the provided document excerpt.

            Document excerpt:
            --------------------
            {Docs}
            --------------------

            Instructions:
            - Answer the user's question using only the information from the document above.
            - If the answer is found, quote or reference the relevant part(s) of the document.
            - If the answer cannot be found in the document, reply: "Not enough information in the provided document."
            - Be clear and concise.

            Question: {user_question}
            Answer:
            ''',
                input_variables = ["user_question", "Docs"]
            )

    ###### Retrieval Query and Retrieve documents based on Retrieval Query---------------------------------------------------------------------------------------------
            with st.spinner("Generating Answer..."):
                # user_question = "what embedding model is being used in the video and what is the data used for building the coding assistant?"
                retrieval_query = user_question
                
                # For better understanding
                retrieved_docs = retriever.invoke(retrieval_query)

                # Concatenting text as strings
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # Retrieved document's text concatenated
                retrieve_docs = format_docs(retrieved_docs)

                # Output Parser
                parser = StrOutputParser()

                # Chain making with user query prompt
                chain = prompt|model|parser
                result = chain.invoke({"user_question": user_question, "Docs": retrieve_docs})
            st.write(result)