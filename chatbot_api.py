

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk
import re
import os
import json
from dotenv import load_dotenv
import csv
import uuid
from typing import Optional, List, Dict
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough

# Load the .env file
load_dotenv()

app = FastAPI()

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("synthesized", exist_ok=True)
os.makedirs("chat_history", exist_ok=True)

class Message(BaseModel):
    role: str  # 'human' or 'ai'
    content: str

class ChatHistory(BaseModel):
    messages: List[Message]

class CryptoChatbot:
    def __init__(self, openai_key, azure_key, azure_region, report_db_path="faiss_index_report"):
        self.openai_key = openai_key
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.report_db_path = report_db_path
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)
        self.report_db = self._initialize_report_db()
        self.chat_history = []
        self.session_id = str(uuid.uuid4())
        self.debug_mode = False

        # Initialize Azure Speech SDK with English as default
        self.speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        self.language_locale_map = {
            'urdu': 'ur-PK',
            'arabic': 'ar-SA',
            'turkish': 'tr-TR',
            'french': 'fr-FR',
            'spanish': 'es-ES',
            'english': 'en-US',
            'chinese': 'zh-CN'
        }
        self.selected_language = 'english'
        self.speech_config.speech_recognition_language = self.language_locale_map['english']
        self.speech_config.speech_synthesis_language = self.language_locale_map['english']

    def set_language(self):
        """
        Set the language for speech recognition and synthesis.
        """
        print("Available languages:")
        for i, lang in enumerate(self.language_locale_map.keys()):
            print(f"{i + 1}. {lang}")

        choice = int(input("Choose a language by entering the number: ")) - 1
        if 0 <= choice < len(self.language_locale_map):
            self.selected_language = list(self.language_locale_map.keys())[choice]
            print(f"Language selected: {self.selected_language}")
        else:
            print("Invalid choice. Defaulting to English.")
            self.selected_language = 'english'

        # Update the speech config with the selected language
        self.speech_config.speech_recognition_language = self.language_locale_map[self.selected_language]
        self.speech_config.speech_synthesis_language = self.language_locale_map[self.selected_language]

    def _initialize_report_db(self):
        """
        Initialize the FAISS vector store for document embeddings.
        """
        if not os.path.exists(self.report_db_path):
            file_names = [
                'data/24 Tokens Review Reports.docx',
                'data/45 Tokens Review Reports.docx',
                'data/50 Token Review Reports.docx',
                'data/52Tokens Review Reports.docx',
                'data/Conceptual Data.docx'
            ]

            all_protocols = []
            for file_name in file_names:
                protocols, success = self.process_file(file_name)
                if success:
                    all_protocols.extend(protocols)
                else:
                    print(f"Failed to process {file_name}")

            report_db = FAISS.from_documents(all_protocols, self.embeddings)
            report_db.save_local(self.report_db_path)
            return report_db
        else:
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

            sections = re.split(r'\n(\d+\.\s*Name of the Protocol:\s*)', data_text)

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

                metadata = {}
                for key, pattern in metadata_patterns.items():
                    match = re.search(pattern, content_section)
                    if match:
                        metadata[key] = match.group(1).strip() if match.group(1) else "NULL"
                        content_section = re.sub(pattern, '', content_section)

                content_section = re.sub(r'\n{2,}', '\n\n', content_section).strip()

                document = Document(page_content=content_section, metadata=metadata)
                protocols.append(document)

            return protocols, True
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return [], False
        
    def set_debug_mode(self, debug: bool):
        """Enable or disable debug mode"""
        self.debug_mode = debug
        if debug:
            print(f"Debug mode enabled for session: {self.session_id}")

    def set_chat_history(self, history: List[Message]):
        """Set chat history from a list of messages"""
        self.chat_history = [
            HumanMessage(content=msg.content) if msg.role == 'human'
            else AIMessage(content=msg.content)
            for msg in history
        ]

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get current chat history as a list of dictionaries"""
        return [
            {
                'role': 'human' if isinstance(msg, HumanMessage) else 'ai',
                'content': msg.content
            }
            for msg in self.chat_history
        ]


    def chat(self, user_input: str, history: Optional[List[Message]] = None) -> tuple[str, Optional[List[Dict[str, str]]]]:
        """
        Process a chat message and return the response along with updated history
        """
        try:
            if history:
                self.set_chat_history(history)

            output_parser = StrOutputParser()
            
            if user_input and self.report_db:
                retriever = self.report_db.as_retriever()

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

                qa_system_prompt = """You are a virtual assistant specializing in Crypto Currencies.\
                    Use the following pieces of retrieved context to answer the question.\
                    If you don't know the answer, just say 'I do not know'. Do not generate new knowledge.\
                    If you are being asked for financial advice, or prediction of tokens future price simply say My role is to provide information on the Shariah compliance of cryptocurrencies.\
                    If you get a argumentative response, like i dont agree or i dont think so, rephrase your reponse in a convincing manner.\
                    
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

                ai_msg = rag_chain.invoke({"question": user_input, "chat_history": self.chat_history})
                self.chat_history.extend([HumanMessage(content=user_input), AIMessage(content=ai_msg)])
                
                return ai_msg, self.get_chat_history() if self.debug_mode else None
            else:
                return "I do not know", None
        except Exception as e:
            print(f"Error in chat: {e}")
            return "An error occurred while processing your request.", None

    def speak(self, text):
        """
        Convert text to speech using Azure Speech SDK.
        """
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized successfully.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("Speech synthesis canceled: {}".format(result.cancellation_details.reason))

    def listen(self):
        """
        Listen to audio input and convert it to text using Azure Speech SDK.
        """
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        print("Listening...")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {result.text}")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
            return None
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("Speech recognition canceled: {}".format(result.cancellation_details.reason))
            return None

    def recognize_from_file(self, file_path):
            """
            Recognize speech from an audio file using Azure Speech SDK.
            """
            audio_config = speechsdk.AudioConfig(filename=file_path)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

            result = speech_recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return "No speech could be recognized."
            elif result.reason == speechsdk.ResultReason.Canceled:
                return f"Speech recognition canceled: {result.cancellation_details.reason}"

    def synthesize_to_file(self, text, output_file):
        """
        Synthesize speech to an audio file using Azure Speech SDK.
        """
        audio_config = speechsdk.AudioConfig(filename=output_file)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True
        else:
            print(f"Speech synthesis failed: {result.cancellation_details.reason}")
            return False


crypto_chatbot = CryptoChatbot(
    openai_key=os.getenv("openai_key"),
    azure_key=os.getenv("azure_key"),
    azure_region=os.getenv("azure_region")
)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = None
    debug: Optional[bool] = False

class LanguageRequest(BaseModel):
    language: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if request.debug:
            crypto_chatbot.set_debug_mode(True)
        
        response, history = crypto_chatbot.chat(request.message, request.history)
        output_file = f"synthesized/{uuid.uuid4()}.wav"
        success = crypto_chatbot.synthesize_to_file(response, output_file)
        
        result = {
            "success": True,
            "response": response,
            "audio_file": output_file if success else None
        }
        
        if history:
            result["history"] = history
            
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/chat_audio")
async def chat_audio(audio: UploadFile = File(...)):
    try:
        file_path = f"uploads/{audio.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await audio.read())
        
        recognized_text = crypto_chatbot.recognize_from_file(file_path)
        response = crypto_chatbot.chat(recognized_text)
        
        output_file = f"synthesized/{uuid.uuid4()}.wav"
        success = crypto_chatbot.synthesize_to_file(response, output_file)
        
        os.remove(file_path)  # Remove the uploaded file after processing
        
        if success:
            return {"true": response, "audio_file": output_file}
        else:
            return {"true": response, "audio_file": None}
    except Exception as e:
        return {"false": str(e)}

@app.get("/download_audio/{filename}")
async def download_audio(filename: str):
    file_path = f"synthesized/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/set_language")
async def set_language(request: LanguageRequest):
    try:
        if request.language in crypto_chatbot.language_locale_map:
            crypto_chatbot.selected_language = request.language
            crypto_chatbot.speech_config.speech_recognition_language = crypto_chatbot.language_locale_map[request.language]
            crypto_chatbot.speech_config.speech_synthesis_language = crypto_chatbot.language_locale_map[request.language]
            return {"true": f"Language set to {request.language}"}
        else:
            return {"false": "Invalid language selection"}
    except Exception as e:
        return {"false": str(e)}

@app.get("/available_languages")
async def get_available_languages():
    try:
        return {"true": list(crypto_chatbot.language_locale_map.keys())}
    except Exception as e:
        return {"false": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



'''
SAMPLE REQUEST BODY 


Where, 

"message" is for actual user query,
"histroy" will be used to attach chat_history as requested (optional argument)
"debug" is for debugging purpose to see what histroy is maintained for, history will be provided in resposne too (optional argument)


e.g:

{
    "message": "what is bitcoin?",
    "history": [
        {
            "role": "human",
            "content": "Tell me about cryptocurrency"
        },
        {
            "role": "ai",
            "content": "Cryptocurrency is a digital currency..."
        }
    ],
    "debug": true
}

'''