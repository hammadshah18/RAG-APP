# import asyncio
# import nest_asyncio
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---- Fix for Streamlit + async gRPC ----
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
# nest_asyncio.apply()
# # ----------------------------------------

# Step 1: Configurations
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv('GOOGLE_API_KEY'))
# Using free local embeddings to avoid API quota limits
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2: Steamlit App
st.set_page_config(page_title="RAG with LangChain", page_icon=":)", layout="wide")
st.title("📂 RAG Q&A with Gemini")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt",'docx'])

if uploaded_file:
    with st.spinner("Processing file... "):
        # Save uploaded file locally
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load file
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embeddings + Vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        system_prompt = (
    "You are an intelligent AI assistant. Use the retrieved context only when it is relevant, "
    "but you are also allowed to answer using your own internal knowledge. "
    "If the context does not contain useful information, answer from your general knowledge. "
    "If both context and your own knowledge are useful, combine them:\n\n"
    "{context}"
)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        qa = create_retrieval_chain(retriever, question_answer_chain)

        st.success(" File processed successfully!")

        # Ask a question
        user_query = st.text_input("Ask a question ")
        if st.button("Get Answer"):
            if user_query.strip() != "":
                with st.spinner("Fetching answer... "):
                    response = qa.invoke({"input": user_query})
                    
                    # Show Answer
                    st.subheader(" Answer")
                    st.write(response["answer"])

                    # Show Metadata (Sources)
                    st.subheader("📎 Sources / Metadata")
                    for i, doc in enumerate(response["context"], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata}")
                        st.write(doc.page_content[:200] + "...")