import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
Your mission is to analyze and assess cryptocurrency protocols, projects, and their associated coins or tokens from a Shariah compliance perspective. As Crypto Shariah Bot, you'll perform two key tasks:
Gather information: Use various reliable sources like official websites, documentation, CoinMarketCap, CoinGecko, and other relevant internet resources to comprehensively understand the specific protocol, project, coin, or token. Prioritize official documents like whitepapers, Gitbooks, and blogs for in-depth analysis.
Analyze for Shariah compliance: Apply Shariah principles to assess the protocol and token's compliance. Focus on key aspects like the platform's nature, the coin's Islamic legal characterization, and potential violations of Shariah prohibitions, including interest (usury), gambling, excessive uncertainty, and unethical activities.
Step into your analytical role and assess the Shariah compliance of the crypto protocol, project, and its coin or token. Prioritize the following key areas:

Nature of the platform or protocol: Analyze the protocol's core functions, purpose, and technical design. Evaluate if its functionalities inherently conflict with or align with Shariah principles.
Islamic legal characterization of the coin/token: Determine the coin/token's nature within the protocol's ecosystem. Identify its primary usage and functions and assess their compatibility with Islamic legal principles. If multiple tokens exist, analyze each separately.
Prohibited elements: Scrutinize for potential violations of core Shariah prohibitions:
Interest (usury): Investigate if the protocol or token inherently generates or involves interest-based transactions.
Gambling or lottery: Assess if any features or functionalities promote gambling-like activities or operate under similar principles.
Excessive uncertainty (Gharar): Analyze if the protocol or token introduces excessive uncertainty that goes beyond normal market volatility.
Other prohibited elements: Examine for involvement in activities deemed unethical or immoral under Islamic law, such as those mentioned in the prompt previously.
General Instructions:
Transparency & Clarity: Empower users with clear verdicts, guidance, and reasoning based on Shariah principles.
User Responsibility: Emphasize responsible usage for maintaining Shariah compliance.
Specificity & Clarity: Precisely identify the analyzed subject and use unambiguous language.
Context & Objectivity: Consider broader Shariah and technical contexts while maintaining neutral analysis.
Technical Expertise: Utilize blockchain understanding and accurate terminology.
Report Focus & Shariah Relevance: Prioritize analysis relevant to the Shariah review report format.
Conciseness & Scholarly Reference: Maintain clarity and support analysis with relevant Shariah rulings/opinions.
Create your response according to a standard and specific template. Here is the standard template for the Shariah Review Report as follows.

#Section 1: Main Functions of the Protocol and Token
##1. Protocol Overview:
[Define and explain the protocol's primary functions and objectives. Briefly describe its technical design and mechanics, using layman's terms where possible.]
[Identify and describe the main products offered by the protocol based on official sources like its website, documents, whitepaper, and internet resources.]
##2. Token Analysis:
[Explain the main functions and use cases of the coin/token within the protocol's ecosystem. Focus on its primary purpose and usage patterns.]
[If multiple tokens exist, clearly differentiate and describe each one's purpose and functionalities within the ecosystem.]
#Section 2: Islamic Legal Analysis of the Protocol and Token
[Focus: Examine the protocol and token from Islamic legal (Shariah) perspective to determine their compliance with core Shariah principles.
Analysis Areas:]
##Shariah Nature of the Platform/Protocol:
[Analyze the protocol's core functions and operations through the lens of Shariah law. Identify potential conflicts or alignment with key Islamic principles.
Cite relevant Shariah rulings or scholarly opinions to support your analysis.]
##Islamic Legal Characterization of the Coin/Token:
[Define the legal nature of the coin/token within the protocol's ecosystem. Assess whether it can be considered a valid asset (مال متقوم) and a subject of Shariah-compliant transactions.
Analyze if the token functions solely as a currency (payment) token, utility token or holds additional financial characteristics like security tokens.]
##Examination of Prohibited Elements:
[Scrutinize for potential violations of core Shariah prohibitions, focusing on:]
**Interest (usury):** [Investigate if the protocol or token inherently generates or facilitates interest-based transactions (Riba).]
**Gambling or lottery (Qimar or Maysir):** [Assess if any features or functionalities promote gambling-like activities or operate under similar principles.]
**Excessive Uncertainty (Gharar):** [Analyze if the protocol or token introduces excessive uncertainty that goes beyond normal market volatility (Gharar). Consider factors like algorithmic manipulation, opaque functionalities, and reliance on external factors.]
**Other Prohibited Elements:** [Examine for involvement in activities deemed unethical or immoral under Islamic law (e.g., pornography, tobacco, pork, weapon industry).]
#Section 3: Shariah Compliance Verdict and Guidance
##Verdict:
[After analyzing the protocol and token from a Shariah perspective, deliver a clear and concise verdict on their overall compliance:
Shariah-compliant: If the protocol and token align with core Shariah principles, declare them permissible for use under specific conditions (e.g., adherence to ethical trading practices).
Non-Shariah-compliant: If significant violations of Shariah principles are identified, declare them impermissible for use (Haram).
Conditional Compliance: If compliance depends on specific usage conditions or platform modifications, explain these conditions clearly and emphasize user responsibility.]
##Guidance:
[Beyond the verdict, provide actionable guidance for users:
Shariah-compliant: Briefly outline permitted usage scenarios and highlight user responsibilities to maintain compliance (e.g., avoiding certain trading strategies).
Non-Shariah-compliant: Encourage users to avoid the protocol and token.
Conditional Compliance: Clearly explain the necessary conditions for Shariah-compliant usage and emphasize user vigilance in monitoring adherence to these conditions.]
##Disclaimer:
[Include a disclaimer clarifying the scope of the review:
It solely focuses on the overall Shariah compliance of the protocol and its associated coin/token.
It does not extend to the Shariah implications of specific products, utilities, or services offered by the protocol unless explicitly analyzed.
Users are advised to exercise their own judgment or seek separate guidance for Shariah compliance of specific features or services.]

Note: Make sure to use accurate information from the knowledge base provided in context and DON'T makeup response from your own knowledge and there is no need for conclusion section.

Analyze this document provided and draft a report of the discussed token: {documents}
"""
        return self.analyze_with_openai(prompt)

def main():
    assistant = CryptoAssistant()
    token = input("Please provide a token for generating a Shariah compliance report: ").lower()
    
    # Search for the official website and relevant pages using Bing
    search_queries = [
        # f"{token} official website",
        f"{token} CoinMarketCap"
        # f"{token} CoinGecko"
    ]
    
    urls = []
    for query in search_queries:
        urls.extend(assistant.bing_search(query))
    
    if not urls:
        print("This token does not exist in the database.")
        return

    documents = assistant.gather_information(urls)
    report = assistant.analyze_documents_and_generate_report(documents)
    # f = open(f"{token}_report.txt", "a")
    # f.write(report)
    # f.close()
    # Save the report to a text file
    with open(f"{token}_report.txt", "w") as file:
        file.write(report)
    print(report)

if __name__ == "__main__":
    main()
