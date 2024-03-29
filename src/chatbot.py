import re
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')


# Define the function to process each file
def process_file(file_path):
    try:
        loader = Docx2txtLoader(file_path)
        data = loader.load()
        data_text = data[0].page_content
        protocols = []

        # Splitting with capturing group to include the protocol names
        sections = re.split(r'\n(\d+\.\s*Name of the Protocol:\s*)', data_text)

        # Skip the first empty section, then iterate by 2 to take the protocol name and content
        for i in range(0, len(sections)-1, 2):
            content_section = sections[i] + sections[i+1]
            metadata_patterns = {
                'Name of the Protocol': r'Name of the Protocol:\s*:?([^\n]*)',
                'Name of the Token': r'Name of the Token:\s*:?([^\n]*)',
                'Official Website': r'Official Website:\s*:?([^\n]*)',
                'Official Documentation Link': r'Official Documentation Link:\s*:?([^\n]*)',
                'CoinMarketCap Link': r'CoinMarketCap Link:\s*:?([^\n]*\bhttps?://\S+)',
                'CoinGecko Link': r'CoinGecko Link:\s*:?([^\n]*)',
                'Initial Assessment': r'Initial Assessment\s*:?([^\n]*)',
                'Initial Assessment Date': r'Initial Assessment Date\s*:?([^\n]*)',
                'Reviewer': r'Reviewer\s*:?([^\n]*)',
                'Review Date': r'Review Date\s*:?([^\n]*)',
                'Report Expiry Date': r'Report Expiry Date\s*:?([^\n]*)'
            }
            
            # Initialize metadata
            metadata = {}
            for key, pattern in metadata_patterns.items():
                match = re.search(pattern, content_section)
                if match:
                    metadata[key] = match.group(1).strip() if match.group(1) else "NULL"
                    # Remove the matched metadata from the content
                    content_section = re.sub(pattern, '', content_section)

            # Further clean up content_section
            content_section = re.sub(r'\n{2,}', '\n\n', content_section).strip()

            # Create Document object
            document = Document(page_content=content_section, metadata=metadata)
            protocols.append(document)

        return protocols, True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], False
    

def crypto_chatbot(user_input, report_db):
    try:
        # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai.api_key)
        # report_db = FAISS.load_local("faiss_index_report", embeddings)

        print("Crypto Chatbot")
        user_input = user_input

        # if user_input and report_db:
        #     docs = report_db.similarity_search(user_input, k=3)
        #     template = ("You are my Virtual Assistant specializing in Crypto Currencies.\n\n"
        #                 "Instructions for Virtual Assistant specializing in Crypto Currencies:\n\n"
        #                 "- Begin the conversation with a friendly greeting.\n"
        #                 "- Use only information from the knowledge base provided in {context}.\n"
        #                 "- If a question is asked that is not related to Crypto Currencies or falls outside the scope "
        #                 "of this document, reply with \"I'm sorry, but the available information is limited as I am an AI "
        #                 "assistant.\"\n"
        #                 "- Refer to the chat history in {chat_history} and respond to the human input as follows: \n"
        #                 "   \"Human: {human_input} \n    Virtual Assistant:\" \n\n"
        #                 "It is important to note that the bot DOES NOT makeup answers and only provides information from "
        #                 "the context provided. And when user greetings the bot like 'Hi', etc. Then it must reply the "
        #                 "greetings professionally.")
        #     prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=template)
        #     memory_report = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", max_history=2)
        #     chain_report = load_qa_chain(ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, max_tokens=2000,
        #                                             openai_api_key=openai.api_key), verbose=True, chain_type="stuff",
        #                                 memory=memory_report, prompt=prompt)
        #     output = chain_report({"input_documents": [docs[0]], "human_input": user_input}, return_only_outputs=False)
        #     return output['output_text'], True

        output_parser = StrOutputParser()

        # report_db = FAISS.load_local("faiss_index_report", embeddings)

        # # Chat interface
        # user_input = st.text_input("Ask me about crypto protocols:")

        if user_input and report_db:
            # Perform search and response generation
            # docs = report_db.similarity_search(user_input, k=3)
            
            # will be used to take the question, and compare it with all the numeric vetors in the database and return the most similar chunks of text
            retriever = report_db.as_retriever()

            # Adding Memory
            instruction_to_system = """
            Given the chat history and the latest user question
            which might reference context in the chat history, formulate a standalone question
            which can be understood without the chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as it is.
            """

            question_maker_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", instruction_to_system),
                    MessagesPlaceholder(variable_name = "chat_history"),
                    ("human","{question}"),
                ]
            )

            llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, max_tokens=2000,openai_api_key = openai_key)

            question_chain = question_maker_prompt | llm | StrOutputParser
            
            # prompt
            qa_system_prompt = """You are a virtual assistant specializing in Crypto Currencies.\
                Use the following pieces of retrieved context to answer the question.\
                If you don't know the answer, provide a summary of the context. Do not generate your answer.\
                
                
                {context}"""
            
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human","{question}"),
                ]
            )


            # Which question to pass to LLM?
            # We define a function that looks at the chat history,
            # if there is a history: it will pass the question chain (that reformulates user's question)
            # if chat history is empty, it will pass user's question directly

            def contextualized_question(input: dict):
                if input.get("chat_history"):
                    return question_chain
                else:
                    return input['question']
                

            # Retriever Chain 
            # We need a chain to pass the following to the llm:
                # context: use the vector retriever and get the most relevant chunks into PDF
                # question: reformulated or the original user's question depending on the history 
                # chat_history: python lists of chats

            # we use the assign function which adds the context to whatever it gets as input and pass it to the next link of the chain.

            from langchain_core.runnables import RunnablePassthrough
            retriever_chain = RunnablePassthrough.assign(
                context = contextualized_question | retriever #| format_docs
            ) 


            # RAG Chain

            rag_chain = (
                retriever_chain 
                | qa_prompt
                | llm
                | output_parser
            )

            question = user_input

            chat_history = []

            ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=question), ai_msg])
            return ai_msg
    except Exception as e:
        print(f"Error in crypto_chatbot: {e}")
        return "An error occurred while processing your request.", "pass"
    


def main():


    # report_db = None  # Placeholder for the FAISS vector store initialization

    # # Check if the file 'faiss_index_report' exists
    # if not os.path.exists("faiss_index_report"):

    #       # File names to process
    #     file_names = [
    #         '24 Tokens Review Reports.docx',
    #         '45 Tokens Review Reports.docx',
    #         '50 Token Review Reports.docx',
    #         '52Tokens Review Reports.docx'
    #     ]

    #     # Process each file
    #     all_protocols = []
    #     for file_name in file_names:
    #         protocols = process_file(file_name)
    #         all_protocols.extend(protocols)



    #     # Initialize the vector store
    #     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)
    #     report_db = FAISS.from_documents(all_protocols, embeddings)
    #     # Save the vector store
    #     report_db.save_local("faiss_index_report")
    # else:
    #     # Load existing vector store
    #     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)
    #     report_db = FAISS.load_local("faiss_index_report", embeddings)


    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)
    report_db = FAISS.load_local("faiss_index_report", embeddings, allow_dangerous_deserialization=True)

    # Here you can call any functions you need to run when the script starts
    # For example, to run the chatbot:
    user_input = input("Ask me about crypto protocols: ")
    response= crypto_chatbot(user_input, report_db)
    # print(success)
    print(response)
    

if __name__ == "__main__":
    main()
