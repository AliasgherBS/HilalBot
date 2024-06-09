import re
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough

class CryptoChatbot:
    def __init__(self, openai_key, report_db_path="faiss_index_report"):
        self.openai_key = openai_key
        self.report_db_path = report_db_path
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)
        self.report_db = self._initialize_report_db()

    def _initialize_report_db(self):
        """
        Initialize the FAISS vector store for document embeddings.
        """
        if not os.path.exists(self.report_db_path):
            # File names to process
            file_names = [
                'data/24 Tokens Review Reports.docx',
                'data/45 Tokens Review Reports.docx',
                'data/50 Token Review Reports.docx',
                'data/52Tokens Review Reports.docx',
                'data/Conceptual Data.docx'
            ]

            # Process each file
            all_protocols = []
            for file_name in file_names:
                protocols, success = self.process_file(file_name)
                if success:
                    all_protocols.extend(protocols)
                else:
                    print(f"Failed to process {file_name}")

            # Create the FAISS vector store
            report_db = FAISS.from_documents(all_protocols, self.embeddings)
            report_db.save_local(self.report_db_path)
            return report_db
        else:
            # Load existing vector store
            return FAISS.load_local(self.report_db_path, self.embeddings)

    def process_file(self, file_path):
        """
        Process a DOCX file to extract protocol information and create Document objects.
        """
        try:
            loader = Docx2txtLoader(file_path)
            data = loader.load()
            data_text = data[0].page_content
            protocols = []

            # Splitting with capturing group to include the protocol names
            sections = re.split(r'\n(\d+\.\s*Name of the Protocol:\s*)', data_text)

            # Skip the first empty section, then iterate by 2 to take the protocol name and content
            for i in range(0, len(sections) - 1, 2):
                content_section = sections[i] + sections[i + 1]
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

    def chat(self, user_input):
        """
        Handle user input to the chatbot and generate a response.
        """
        try:
            output_parser = StrOutputParser()

            if user_input and self.report_db:
                retriever = self.report_db.as_retriever()

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
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}"),
                    ]
                )

                llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, max_tokens=2000, openai_api_key=self.openai_key)

                question_chain = question_maker_prompt | llm | output_parser

                # Prompt for the QA system
                qa_system_prompt = """You are a virtual assistant specializing in Crypto Currencies.\
                    Use the following pieces of retrieved context to answer the question.\
                    If you don't know the answer, just say 'I do not know'. Do not generate new knowledge.\

                    {context}"""

                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", qa_system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}"),
                    ]
                )

                def contextualized_question(input):
                    if input.get("chat_history"):
                        return question_chain
                    else:
                        return input['question']

                retriever_chain = RunnablePassthrough.assign(
                    context=contextualized_question | retriever
                )

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
            else:
                return "I do not know"
        except Exception as e:
            print(f"Error in chat: {e}")
            return "An error occurred while processing your request.", "pass"


def main():
    openai_key = 'open-ai key'

    chatbot = CryptoChatbot(openai_key)
    user_input = input("Ask me about crypto protocols: ")
    response = chatbot.chat(user_input)
    print(response)

if __name__ == "__main__":
    main()
