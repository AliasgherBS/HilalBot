# # Required imports
# # import streamlit as st
# # import re
# # import os
# # from langchain import load_qa_chain
# # from langchain.document_loaders import Docx2txtLoader
# # from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain.schema import Document, PromptTemplate
# # from langchain.vectorstores import FAISS
# # from langchain.memories import ConversationBufferMemory
# # from dotenv import load_dotenv
# # from langchain.chains import ChatOpenAI
# import re
# import os
# from langchain.document_loaders import Docx2txtLoader
# from langchain.schema import Document
# from dotenv import load_dotenv
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# import streamlit as st  
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI


# # Load API keys 
# from constants import openai_key
# from constants import org_key
# import openai
# openai.organization = org_key
# openai.api_key = openai_key




# # Define the function to process each file
# def process_file(file_path):
#     loader = Docx2txtLoader(file_path)
#     data = loader.load()
#     data_text = data[0].page_content
#     protocols = []
#     sections = re.split(r'\n(\d+\.\s*Name of the Protocol:\s*)', data_text)
#     for i in range(0, len(sections)-1, 2):
#         content_section = sections[i] + sections[i+1]
#         metadata_patterns = {
#             # Metadata patterns here
#         }
#         metadata = {}
#         for key, pattern in metadata_patterns.items():
#             match = re.search(pattern, content_section)
#             if match:
#                 metadata[key] = match.group(1).strip() if match.group(1) else "NULL"
#                 content_section = re.sub(pattern, '', content_section)
#         content_section = re.sub(r'\n{2,}', '\n\n', content_section).strip()
#         document = Document(page_content=content_section, metadata=metadata)
#         protocols.append(document)
#     return protocols

# # # Load environment variables
# # load_dotenv()
# # open_ai_key = os.getenv("openai_key")

# # Initialize embeddings and vector store
# embeddings = OpenAIEmbeddings(model='text-embedding-3-small',openai_api_key=openai.api_key)
# report_db = None  # Placeholder for the FAISS vector store

# # Streamlit UI
# st.title("Crypto Chatbot")

# # # File upload
# # uploaded_files = st.file_uploader("Upload protocol documents", accept_multiple_files=True, type=['docx'])
# # if uploaded_files:
# #     all_protocols = []
# #     for uploaded_file in uploaded_files:
# #         # Assuming the uploaded file is saved temporarily for processing
# #         with open(uploaded_file.name, "wb") as f:
# #             f.write(uploaded_file.getbuffer())
# #         protocols = process_file(uploaded_file.name)
# #         all_protocols.extend(protocols)
# #     # Process and initialize vector store after documents are uploaded and processed
# #     report_db = FAISS.from_documents(all_protocols, embeddings)
# #     report_db.save_local("faiss_index_report")
# #     report_db = FAISS.load_local("faiss_index_report", embeddings)


# report_db = FAISS.load_local("faiss_index_report", embeddings)

# # Chat interface
# user_input = st.text_input("Ask me about crypto protocols:")

# if user_input and report_db:
#     # Perform search and response generation
#     docs = report_db.similarity_search(user_input, k=3)
#     template = """Please provide accurate and intelligent responses to our clients as a virtual assistant specializing in Crypto Currencies. Begin the conversation with a friendly greeting. When discussing information or providing recommendations about 
# Crypto Currencies, make sure to use accurate information from the knowledge base provided in {context}.If a question is asked that 
# is not related to Crypto Currencies or falls outside the scope of this document, kindly reply with the response, \"I'm sorry, 
# but the available information is limited as I am an AI assistant.\" Please refer to the chat history in {chat_history} and 
# respond to the human input as follows: \"Human: {human_input} Virtual Assistant:\""""
#     prompt = PromptTemplate(
#         input_variables=["chat_history", "human_input", "context"], template=template
#     )
#     memory_report = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", max_history=2)
#     chain_report = load_qa_chain(ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2000,openai_api_key=openai.api_key), verbose=False, chain_type="stuff", memory=memory_report, prompt=prompt)
#     output = chain_report({"input_documents": [docs[0]], "human_input": user_input}, return_only_outputs=False)
#     st.write(output['output_text'])
# else:
#     st.write("Please upload files and enter a query to get started.")

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
# --------------------------------------------------- 

import re
import os
from langchain.document_loaders import Docx2txtLoader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st  
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Load API keys 
from constants import openai_key
from constants import org_key
import openai
# openai.organization = org_key
openai.api_key = openai_key

# Define the function to process each file
def process_file(file_path):
    loader = Docx2txtLoader(file_path)
    data = loader.load()
    data_text = data[0].page_content
    protocols = []
    sections = re.split(r'\n(\d+\.\s*Name of the Protocol:\s*)', data_text)
    for i in range(0, len(sections)-1, 2):
        content_section = sections[i] + sections[i+1]
        metadata_patterns = {
            # Metadata patterns here
        }
        metadata = {}
        for key, pattern in metadata_patterns.items():
            match = re.search(pattern, content_section)
            if match:
                metadata[key] = match.group(1).strip() if match.group(1) else "NULL"
                content_section = re.sub(pattern, '', content_section)
        content_section = re.sub(r'\n{2,}', '\n\n', content_section).strip()
        document = Document(page_content=content_section, metadata=metadata)
        protocols.append(document)
    return protocols

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai.api_key)
report_db = None  # Placeholder for the FAISS vector store initialization

# Streamlit UI
st.title("Crypto Chatbot")

report_db = FAISS.load_local("faiss_index_report", embeddings)

# Chat interface
user_input = st.text_input("Ask me about crypto protocols:")

if user_input and report_db:
    # Perform search and response generation
    docs = report_db.similarity_search(user_input, k=3)
    template = """Please provide accurate and intelligent responses to our clients as a virtual assistant specializing in Crypto Currencies. Begin the conversation with a friendly greeting. When discussing information or providing recommendations about Crypto Currencies, make sure to use accurate information from the knowledge base provided in {context}. If a question is asked that is not related to Crypto Currencies or falls outside the scope of this document, kindly reply with the response, "I'm sorry, but the available information is limited as I am an AI assistant." Please refer to the chat history in {chat_history} and respond to the human input as follows: "Human: {human_input} Virtual Assistant:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    memory_report = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", max_history=2)
    chain_report = load_qa_chain(ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2000,openai_api_key=openai.api_key), verbose=True, chain_type="stuff", memory=memory_report, prompt=prompt)
    output = chain_report({"input_documents": [docs[0]], "human_input": user_input}, return_only_outputs=False)
    st.write(output['output_text'])
else:
    st.write("Please upload files and enter a query to get started.")





