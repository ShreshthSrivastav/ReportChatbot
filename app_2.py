import streamlit as st
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
import os
import uuid
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone
from dotenv import load_dotenv, find_dotenv

import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import PodSpec
from pinecone import ServerlessSpec

# load_dotenv(find_dotenv(), override=True)


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        # from langchain.document_loaders import PyPDFLoader
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data
  
# Chunk document into smaller pieces
def chunk_data(data, chunk_size=256, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# Embedding the chunks into a vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(
        # model='text-embedding-ada-002'
        )
    # vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Insert embeddings into Pinecone or fetch existing ones
def insert_or_fetch_embeddings(chunks, index_name):
    # importing the necessary libraries and initializing the Pinecone client
    # import pinecone
    # from langchain_community.vectorstores import Pinecone
    # from langchain_openai import OpenAIEmbeddings
    # from pinecone import PodSpec
    # from pinecone import ServerlessSpec

    
    pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

    # loading from existing index
    if index_name in pc.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,  #1536
            metric='cosine',
            spec=ServerlessSpec(
            # spec= PodSpec(
            cloud="aws",
            region="us-east-1"
            # environment="us-west1-gcp",
            )

        # deletion_protection="disabled" 
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store
    
# Ensure this memory object is initialized once and stored in session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Ask a question and get an answer from the vector store
def ask_and_get_answer(vector_store, q, k=3):

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})


    # # Create a memory buffer to track the conversation
    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
    llm = llm,  # Link the ChatGPT LLM
    retriever = retriever,  # Link the vector store based retriever
    memory = st.session_state.memory,  # Link the conversation memory
    chain_type ='stuff',  # Specify the chain type
    verbose = False  # Set to True to enable verbose logging for debugging
    )

    try:
        answer = chain.invoke({'question': q})
        # if answer.strip():  # If an answer is found, return it
        return answer.get('answer', "I'm sorry, I couldn't find an answer.")
    except Exception as e:
        print(f"Retrieval error: {e}")


    general_prompt = f"Answer this question as an intelligent assistant: {q}"
    return llm(general_prompt).content

    # answer = chain.invoke({'question': q})
    # return answer['answer']

def calculate_embedding_cost(texts):
    import tiktoken
    # enc = tiktoken.encoding_for_model('text-embedding-3-small')
    enc = tiktoken.encoding_for_model('text-embedding-3-small') #   text-embedding-ada-002
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens,   total_tokens / 1000 * 0.0004

def delete_pinecone_index(index_name='all'):
    # import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')
    

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":

    
    st.subheader("Petroleum Report Chatbot using RAG-Pinecone-LLM 🤖")
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password') 
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        pine_api_key = st.text_input('PINECONE API Key:', type='password') 
        if pine_api_key:
            os.environ['PINECONE_API_KEY'] = pine_api_key

        st.session_state.uploaded_file = st.file_uploader("Upload the report:", type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', value=512, min_value=100, max_value=2048, on_change=clear_history)
        k = st.number_input('k (Number of top documents):', value=3, min_value=1, max_value=20, on_change = clear_history)   
        add_data = st.button('Add Data to PINECONE', on_click=clear_history)

        if st.session_state.uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = st.session_state.uploaded_file.read()
                file_name = os.path.join('./', st.session_state.uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')


                index_name = f"chat-{uuid.uuid4()}"
                vector_store = insert_or_fetch_embeddings(chunks, index_name)

                st.session_state.vs = vector_store
                st.session_state.file_name = file_name
                st.write("Index name: ", index_name)
                st.success('File uploaded, chunked and embedded successfully.')


    del_index = st.sidebar.button('Delete all PINECONE index')
    if del_index:
        delete_pinecone_index('all')
        st.sidebar.success('All indexes deleted successfully.')

        
    # If the file is uploaded, load the PDF
    if st.session_state.uploaded_file:
        import fitz  # Ensure fitz (PyMuPDF) is imported

        # Load the PDF only once and store it in session state
        if 'doc' not in st.session_state:
            st.session_state.doc = fitz.open(stream=st.session_state.uploaded_file.read(), filetype="pdf")
            st.session_state.total_pages = st.session_state.doc.page_count
            st.session_state.page_number = 1  # Default to the first page

        # View PDF controls
        st.session_state.page_number = st.slider(
            "Page", 
            1, 
            st.session_state.total_pages, 
            st.session_state.page_number
        )

        # Render the selected page
        page = st.session_state.doc.load_page(st.session_state.page_number - 1)
        image = page.get_pixmap(dpi=150)  # Render the page as an image
        st.image(image.tobytes("png"), 
                caption=f"Page {st.session_state.page_number} of {st.session_state.total_pages}", 
                use_column_width=True)

        



    q = st.text_input('Ask a question about the uploaded report:')
    if q and 'vs' in st.session_state:
        vector_store = st.session_state.vs
        # st.write(f'k: {k}')
        answer = ask_and_get_answer(vector_store, q, k)
        st.text_area('LLM Answer:', value = answer)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-" *100} \n {st.session_state.history}'
        h = st.session_state.__hash__
        # h = st.session_state.history

        st.text_area(label="Chat History", value = h, key = 'history', height=400)
