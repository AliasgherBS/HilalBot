from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

app = FastAPI()

class CryptoAssistant:
    def __init__(self):
        self.openai_key = ''
        self.bing_api_key = ''
        self.bing_search_url = ''

    def bing_search(self, query, count=2):
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        params = {"q": query, "count": count}
        response = requests.get(self.bing_search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        urls = [result["url"] for result in search_results.get("webPages", {}).get("value", [])]
        return urls

    def gather_information(self, urls):
        official_site_text = ""
        whitepaper_text = ""
        for url in urls:
            text = self.scrape_website(url)
            if "whitepaper" in url.lower():
                whitepaper_text += text
            else:
                official_site_text += text
        return official_site_text, whitepaper_text

    def scrape_website(self, url):
        if not url:
            return ''
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.find('body').get_text().strip()
        cleaned_text = ' '.join(text.split())
        return cleaned_text

    def analyze_with_openai(self, prompt):
        chat = ChatOpenAI(model="gpt-4-0125-preview", temperature=0.1, openai_api_key=self.openai_key)
        messages = [
            SystemMessage(content="You are a virtual assistant specializing in Crypto Currencies."),
            HumanMessage(content=prompt)
        ]
        return chat.invoke(messages).content

    def analyze_documents_and_generate_report(self, documents):
        prompt = f"""
        [Your existing prompt here]

        Analyze this document provided and draft a report of the discussed token: {documents}
        """
        return self.analyze_with_openai(prompt)

assistant = CryptoAssistant()

class TokenRequest(BaseModel):
    token: str

@app.post("/generate_report")
async def generate_report(request: TokenRequest):
    try:
        token = request.token.lower()
        search_queries = [f"{token} CoinMarketCap"]
        
        urls = []
        for query in search_queries:
            urls.extend(assistant.bing_search(query))
        
        if not urls:
            return {"false": "This token does not exist in the database."}

        documents = assistant.gather_information(urls)
        report = assistant.analyze_documents_and_generate_report(documents)
        
        # Save the report to a text file
        with open(f"{token}_report.txt", "w") as file:
            file.write(report)
        
        return {"true": report}
    except Exception as e:
        return {"false": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)