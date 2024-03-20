import re
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Assuming .env file exists and has OPENAI_KEY and ORG_KEY
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
# openai.organization = os.getenv('ORG_KEY')

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
    

def crypto_chatbot(user_input):
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai.api_key)
        report_db = FAISS.load_local("faiss_index_report", embeddings)

        print("Crypto Chatbot")
        user_input = user_input

        if user_input and report_db:
            docs = report_db.similarity_search(user_input, k=3)
            template = ("You are my Virtual Assistant specializing in Crypto Currencies.\n\n"
                        "Instructions for Virtual Assistant specializing in Crypto Currencies:\n\n"
                        "- Begin the conversation with a friendly greeting.\n"
                        "- Use only information from the knowledge base provided in {context}.\n"
                        "- If a question is asked that is not related to Crypto Currencies or falls outside the scope "
                        "of this document, reply with \"I'm sorry, but the available information is limited as I am an AI "
                        "assistant.\"\n"
                        "- Refer to the chat history in {chat_history} and respond to the human input as follows: \n"
                        "   \"Human: {human_input} \n    Virtual Assistant:\" \n\n"
                        "It is important to note that the bot DOES NOT makeup answers and only provides information from "
                        "the context provided. And when user greetings the bot like 'Hi', etc. Then it must reply the "
                        "greetings professionally.")
            prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=template)
            memory_report = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", max_history=2)
            chain_report = load_qa_chain(ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, max_tokens=2000,
                                                    openai_api_key=openai.api_key), verbose=True, chain_type="stuff",
                                        memory=memory_report, prompt=prompt)
            output = chain_report({"input_documents": [docs[0]], "human_input": user_input}, return_only_outputs=False)
            return output['output_text'], True
    except Exception as e:
        print(f"Error in crypto_chatbot: {e}")
        return "An error occurred while processing your request.", False
    


def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_KEY')
    openai.organization = os.getenv('ORG_KEY')

    # Here you can call any functions you need to run when the script starts
    # For example, to run the chatbot:
    user_input = input("Ask me about crypto protocols: ")
    response, success = crypto_chatbot(user_input)
    print(success)
    print(response)

if __name__ == "__main__":
    main()
