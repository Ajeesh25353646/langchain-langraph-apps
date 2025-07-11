import os
import base64
from IPython.display import display, Image
from dotenv import load_dotenv
from sqlalchemy import true
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# import json
from google import genai
import time
import streamlit as st


start = time.time()
# Load the API Key
load_dotenv()
my_google_api_key = os.getenv("GOOGLE_API_KEY")

#############  Getting the api key from the .env file-----------------------------------------------------------
# Checking if key is present or not
def use_gemini_model(path, prompt):
    if not my_google_api_key:
        print("GOOGLE_API_KEY secret not found or not attached!")
    else:
        client = genai.Client(api_key=my_google_api_key)
        my_file = client.files.upload(file=path)
        response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[my_file, prompt],
    )
        print("Model loaded - API is working")
        return response


#############  Getting the api key from the .env file---------------------------------------------------------------------------------------------
# Checking if key is present or not
if not my_google_api_key:
    print("GOOGLE_API_KEY secret not found or not attached!")
else:
    # Checking model
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=my_google_api_key)
    print("Model loaded - API is working")


# Importing the local modal
@st.cache_resource(show_spinner="loaing local model...")
def cache_local_model():
    """Loads Model Locally"""
    gemma_modal = OllamaLLM(model="gemma3:4b")
    return gemma_modal



######### Loading the pdf  ---------------------------------------------------------------------
# pdf without images
@st.cache_data(show_spinner="Loading pdf...")
def load_pdf(file_bytes):
    "Loads the pdf file as LangChain document object"

    with open("random.pdf", "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader("random.pdf")
    docs = loader.load()
    return docs


@st.cache_data(show_spinner="Loading PDF with images in a structured format...")
def load_pdf_with_images(file_path):
    loader = UnstructuredLoader(file_path=file_path,
                                Strategy="hi_res",
                                Languages=["eng"],
                                infer_table_structure = True,
                                extract_images_in_pdf=True, 
        extract_image_block_types=["Image", "Table", "Figure"],
        extract_image_block_to_payload=True,
        )

    # Load the file 
    docs = loader.load()

    end_time = time.time()
    print(f"Time Taken: {end_time - start}")
    print("#########################################")
    print(f"Length of docs: {len(docs)}")
    return docs


@st.cache_data(show_spinner="Extracting images and its metadata...")
def extract_images_and_captions(_Documents):
    """
    Extract base64 images and their captions from documents.
    later to be used for image captioning model.
    """
    
    # First, collecting all images with base64 data & their captains
    images_data = []
    img_caption = []
    
    for doc in _Documents:
        if 'image_base64' in doc.metadata:
            images_data.append({
                'page_number': doc.metadata.get('page_number'),
                'category': doc.metadata.get('category'),
                'image_base64': doc.metadata.get('image_base64'),
                'image_mime_type': doc.metadata.get('image_mime_type', 'image/png'),
                'text_content': doc.page_content,
                'coordinates': doc.metadata.get('coordinates')
            })
        if doc.metadata.get('category') == 'FigureCaption':   # Caption may get saved in some other place e.g. ImageCaption, I need to test
            img_caption.append(doc.page_content)     
            # print(doc.page_content)

    # # verifying all images extracted had captains
    if len(images_data) == len(img_caption):
        for i in range(len(images_data)):
            images_data[i]['img_caption'] = img_caption[i]
    return images_data

    


##### Prompt ---------------------------------------------------------------
# This part generate text descriptions for each image and then inject them back into the documents text.

# Initialising genai client 
client = genai.Client(api_key=my_google_api_key)

# images_data = images_data[0]
@st.cache_resource(show_spinner="LOading Local model")
def get_local_multimodal_model(image_base64_data):
        local_llm = OllamaLLM(model="gemma3:4b")
        local_llm_with_image = local_llm.bind(images=[image_base64_data])
        return local_llm_with_image


# Takes in dict with image & and its metadata
@st.cache_data(show_spinner="performing image captioning...")
def docs_with_image_text(_images_data, _Docs, use_local_model=False):
    """
    images_data: dict with image and its metadata
    docs: Loaded docs with images in it
    use_local_model: Flag to use local model for image captioning

    The task of this function is to generate image captions 
    for all the tables and images in the pdf and then those
    image captions are fed back into the pdf replacing images
    """

    for i, image_data in enumerate(_images_data):
        path = f'./my_image_{i}.jpeg'

        category = image_data.get('category', None)
        ocr_text = image_data.get('text_content', None)
        existing_caption = image_data.get('img_caption', None)
        
        prompt = f"""
            You are an expert at describing visual content.
            I will provide:
            - 1. An image 
            - 2. A category label: {category}
            - 3. Extracted text from the image:\n {ocr_text}
            - 4. A pre-existing image caption:\n {existing_caption}
            
            

            Your task is to write an improved, accurate, and detailed caption that clearly describes the visual content. Take the following into account:
            - Use the extracted text only if it adds clarity.
            - If the category is 'table', the caption can be quite noisy & can be safely ignored if thats the case.Further on the structure, headers, or patterns and describe it.
            - If the category is 'image', describe the scene, objects, or layout.
            """

        # decoding the base64 encoded image data & feeding it to model to generate text description
        image_base64 = image_data['image_base64']
        pic = base64.b64decode(image_base64)
        with open(path, 'wb') as f:
            f.write(pic)


        # Selects the moddle to choose based on toggle
        if use_local_model:
            model_ = get_local_multimodal_model(image_base64)
            response = model_.invoke(prompt)
            generated_description = response
        else:
            # Upload the file and generate content
            my_file = client.files.upload(file=path)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[my_file, prompt],
            )
            generated_description = response.text

        description = f'A visual figure/diagram/table described in text: {generated_description}'
        print(f"Generated description for image {i}:\n{description}\n")

        # Now, find the original document and inject the image description there
        for doc in _Docs:
            if doc.metadata.get('image_base64') == image_base64:   # matching based on image
                doc.page_content = description
                new_metadata = {'page_number': doc.metadata.get('page_number'), 'category': 'GeneratedDescription'}
                doc.metadata = new_metadata
                print(f"Injected description back into docs for page {doc.metadata['page_number']}")
                break 
           
        os.remove(path)
    return _Docs


@st.cache_resource(show_spinner="Generating Embedding...")
def create_vector_store(_document):
    """
    Takes in the Langchain Document loader object
    
    Task: Does text splitting, then makes embedding and embeddings are stored in vector store
    Bonus: Everything done is then cached for next time use
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(_document)
    
    if not splitted_docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = FAISS.from_documents(splitted_docs, embeddings)
    return vector_store


#################### Setting up the App Interface -------------------------------------------------------------
# Define the title for my bot
st.title("PDF RAG with multimodal support")

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

# Select between multimodal and normal mode
multimodal_mode = st.toggle("Multimodal mode")

# Toggle to save extracted images
save_images = st.toggle("Save Images")




# --- Button --------------------------------------------------
if st.button("Generate Answer"):

    # chooses between gemini or local model
    if local_model:
        # st.info("Using local model...")
        model = cache_local_model()
    else:
        # st.info("Using Gemini model...")
        model = gemini_model
# ---------------------------------------------------------

    if uploaded_file is not None and user_question is not None:
            file_bytes = uploaded_file.getvalue()
            with open("temp.pdf", "wb") as f:
                f.write(file_bytes)

            
            # This part of code is st.fragment makes code run indepently from
            # the rest of the app, this way I dont need to do anythihng else to
            # generate extract the images - just do this image toggle
            docs_with_images = load_pdf_with_images("temp.pdf")
            images_data = extract_images_and_captions(docs_with_images)
                 
            @st.fragment
            def save_image(_images_data): 
                if save_images:
                    for i, image_data in enumerate(images_data):
                        image_base64 = image_data['image_base64']
                        pic = base64.b64decode(image_base64)
                        image_format = (image_data["image_mime_type"].split('/')[-1])
                        print(st.image(pic, caption=f"Extracted image {i+1} from page {image_data.get('page_number')}"))
                        st.download_button(
                            label=f"Download Image {i+1}",
                            data=pic,
                            file_name=f"image_{i+1}.{image_format}",
                            mime=image_data['image_mime_type']
                        )  
            save_image(images_data)    

            if multimodal_mode:
                # load pdf
                docs_with_images = load_pdf_with_images("temp.pdf")

                # loads images data and its metadata in a dictionary
                images_data = extract_images_and_captions(docs_with_images)
                          

                docs = docs_with_image_text(_images_data=images_data, _Docs=docs_with_images, use_local_model=local_model)
                
            
            else:
                # Directly load pdf as a document loader - no formalities needed
                docs = load_pdf(file_bytes=file_bytes)


            vector_store = create_vector_store(docs)

            if vector_store is None:
                st.warning("No text could be extracted from the PDF. Please try another file.")
                st.stop()
            
            print("Vector store created/retrieved from cache.")

            # --- Retriever Selection ---
            st.write(f"Using {retriever_choice}...")
            with st.spinner("Retrieving useful information...."):
                if retriever_choice == "MultiQueryRetriever":
                    retriever = MultiQueryRetriever.from_llm(
                        retriever=vector_store.as_retriever(search_kwargs={"k":15}), llm=model
                    )
                elif retriever_choice == "VectorStoreRetriever":
                    retriever = vector_store.as_retriever(search_kwargs={"k":20})
                elif retriever_choice == "ContextualCompressionRetriever":
                    # I'll get more documents (e.g. 10) and let the compressor filter them
                    base_retriever = vector_store.as_retriever(search_kwargs={"k":20})
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
            - Be clear and concise and try to provide page number as reference if possible

            Question: {user_question}
            Answer with reference:
            ''',
                input_variables = ["user_question", "Docs"]
            )

    ###### Retrieval Query and Retrieve documents based on Retrieval Query---------------------------------------------------------------------------------------------            
            with st.spinner("Generating Answer..."):
                # user_question = "what embedding model is being used in the video and what is the data used for building the coding assistant?"
                retrieval_query = user_question
                
                # For better understanding
                retrieved_docs = retriever.invoke(retrieval_query)

                
                print(retrieved_docs)
                # Concatenting text as strings
                def format_docs(docs):
                    format_docs_list = []
                    for doc in docs:
                        page_num = doc.metadata.get("page_number", "Unknown")
                        category = doc.metadata.get("category", "Text")
                        format_docs_list.append(f"----Page Number: {page_num} and category: {category}---\n{doc.page_content}")
                    return "\n\n".join(format_docs_list)

                # Retrieved document's text concatenated
                retrieve_docs = format_docs(retrieved_docs)
                print(retrieve_docs)

                # Output Parser
                parser = StrOutputParser()

                # Chain making with user query prompt
                chain = prompt|model|parser
                result = chain.invoke({"user_question": user_question, "Docs": retrieve_docs})
            st.write(result)



