import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
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
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=my_google_api_key)
    print("Model loaded - API is working")

######### Getting the Transcript ---------------------------------------------------------------------
# Define the title for my bot
st.title("Youtube Q/A Bot")

def get__transcript(video_url):
    # parse this url to extract id
    video_id = None
    parsed_url = urlparse(video_url)

    # Handle standard youtube.com URLs
    if parsed_url.hostname and 'youtube.com' in parsed_url.hostname:
        query = parsed_url.query
        params = parse_qs(query)
        if 'v' in params:
            video_id = params['v'][0]
    # Handle shortened youtu.be URLs
    elif parsed_url.hostname and 'youtu.be' in parsed_url.hostname:
        video_id = parsed_url.path.lstrip('/')

    if not video_id:
        raise ValueError("Could not extract video ID from the URL.")

    print(f"Video ID: {video_id}")

    # use proxies in case my request to Yt blocked
    # proxies to counter blocking - https://free-proxy-list.net/
    # proxy = {"http": "http://47.90.149.238:5060"}

    # Get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    print(f"How data is loaded: {type(transcript)}")

    return transcript

#################### Setting up the App Interface -------------------------------------------------------------
# Getting the url
input_link = st.text_input("Enter the Video URL")

# Document Retrieval query
user_question = st.text_input("What question would you like to Ask?")

# Retriever choice
retriever_choice = st.selectbox(
    "Choose a retriever:",
    ("MultiQueryRetriever", "VectorStoreRetriever", "ContextualCompressionRetriever")
)


# --- Button --------------------------------------------------
if st.button("Generate Answer"):
    if input_link and user_question is not None:
        with st.spinner("working on it.."):
            # Invalid Link handling
            try: 
                transcript = get__transcript(input_link)
            except Exception as e:
                st.write(f"Link not valid: {e}")
                st.stop()


    ##### Splitting the text -------------------------------------------------------------------------------------------------------------
            ##### Splitting the text -------------------------------------------------------------------------------------------------------------
            # Manually chunking the transcript to include metadata
            docs = []
            current_chunk_text = ""
            current_chunk_start_time = None
            # This size is in characters, similar to chunk_size in splitters
            chunk_size_limit = 1000 

            for item in transcript:
                if current_chunk_start_time is None:
                    current_chunk_start_time = item['start']

                # If adding the next item exceeds chunk size, finalize the current chunk
                if len(current_chunk_text) + len(item['text']) > chunk_size_limit and current_chunk_text:
                    docs.append(Document(page_content=current_chunk_text.strip(), metadata={'start': current_chunk_start_time}))
                    # Start a new chunk
                    current_chunk_text = item['text']
                    current_chunk_start_time = item['start']
                else:
                    current_chunk_text += " " + item['text']

            # Add the last remaining chunk
            if current_chunk_text:
                docs.append(Document(page_content=current_chunk_text.strip(), metadata={'start': current_chunk_start_time}))

            print(f"Numer of Chunks: {len(docs)}")
            print(docs[1])

            # Inspecting my chunks 
            print(len(docs))
            print(docs[0])  # inspecting a random chunked document


    ######## Generating Embeddings & Storing the vectors -----------------------------------------------------------------------------------------------------
            # Running embeddings locally
            embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

            # Vector embedding stored 
            vector_store = FAISS.from_documents(docs, embeddings)
            print(vector_store)

            # --- Retriever Selection ---
            st.write(f"Using {retriever_choice}...")
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
            You are an expert assistant helping answer questions about a YouTube video, using only the provided transcript excerpt.

            Transcript excerpt:
            --------------------
            {Docs}
            --------------------

            Instructions:
            - Answer the user's question using only the information from the transcript above.
            - If the answer is found, quote or reference the relevant part(s) of the transcript.
            - If the answer cannot be found in the transcript, reply: "Not enough information in the provided transcript."
            - Be clear and concise.

            Question: {user_question}
            Answer:
            ''',
                input_variables = ["user_question", "Docs"]
            )

    ###### Retrieval Query and Retrieve documents based on Retrieval Query---------------------------------------------------------------------------------------------
            # user_question = "what embedding model is being used in the video and what is the data used for building the coding assistant?"
            retrieval_query = user_question

            
            # For better understanding
            retrieved_docs = retriever.invoke(retrieval_query)

            # Concatenting text & timstamps as strings
            def format_docs(docs):
                return "\n\n".join(f"[{doc.metadata.get('start', '?')}] {doc.page_content}" for doc in docs)

            # Retrieved document's text concatenated
            retrieve_docs = format_docs(retrieved_docs)

            # Output Parser
            parser = StrOutputParser()

            # Chain making with user query prompt
            chain = prompt|model|parser
            result = chain.invoke({"user_question": user_question, "Docs": retrieve_docs})
            st.write(result)