import os
import requests
import json
from dotenv import load_dotenv
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Bing Search API Key

# BING_API_KEY1 = "a50791a745ca4ac5b0f4006c31b8841f"

BING_API_KEY = os.getenv("BING_API_KEY1")
if not BING_API_KEY:
    raise ValueError('AZURE_SUBSCRIPTION_KEY is not set.')

# Setup for Bing Web Search API
BING_SEARCH_URL = 'https://api.bing.microsoft.com/v7.0/search?'


# url = 'https://api.bing.microsoft.com/v7.0/search?' #  + 'q=' + searchTerm + '&' + 'customconfig=' + custom_config_id

# # OpenAI Model Configuration
# base_model = "gpt-4-1106-preview"
# max_tokens = 7000 
# temperature = 0.2



# Initialize session for HTTP requests
session = requests.Session()

def perform_web_search(query):
    """Perform a web search using Bing Search API and return results."""
    headers = {'Ocp-Apim-Subscription-Key': BING_API_KEY}
    params = {'q': quote_plus(query)}
    response = session.get(BING_SEARCH_URL, headers=headers, params=params)
    
    
    response.raise_for_status()
    search_results = response.json()
    
    pages = search_results['webPages']
    results = pages['value']
    
    for result in results[:1]:
        response = requests.get(result['url'])
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.find('body').get_text().strip()
        cleaned_text = ' '.join(text.split('\n'))
        cleaned_text = ' '.join(text.split())
    
#     return response.json()
    return cleaned_text

def gather_information(token):
    """Gather information about a cryptocurrency from various sources."""
    # Example implementation; details depend on specific requirements
#     search_queries = [f"{token} cryptocurrency official website", f"coinmarketcap {token}", f"coingecko {token}"] #, f"{token} cryptocurrency whitepaper"
    search_queries = [f"{token} cryptocurrency official website"]
    results = []
    for query in search_queries:
        results.append(perform_web_search(query))
    return results


def analyze_with_openai(prompt):
    """Use OpenAI to analyze text and return the response."""
    
    chat = ChatOpenAI(model="gpt-4-0125-preview",temperature=0.1, openai_api_key=OPENAI_API_KEY)
    
    messages = [
    SystemMessage(
        content="You are a virtual assistant specializing in Crypto Currencies."
    ),
    HumanMessage(
        content=prompt
    ),
    ]
    
    
    return chat.invoke(messages).content
    
    
#     data = {
#         "model": "gpt-4-0125-preview",
#         "prompt": prompt,
#         "max_tokens": 4000,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a virtual assistant specializing in Crypto Currencies. Make sure to use accurate information from the knowledge base provided in context"
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.1,
#         "top_p": 1,
#         "frequency_penalty": 0,
#         "presence_penalty": 0
#     }
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
#     try:
#         response = requests.post("https://api.openai.com/v1/completions", json=data, headers=headers)
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         return {"error": str(e)}



def analyze_documents_and_report_generation(documents):
    """Analyze cryptocurrency documents from a Shariah perspective."""
    # Placeholder for actual implementation
    analysis_results = []
    for doc in documents:
        # Assume `doc` contains necessary information to be sent to OpenAI for analysis
        prompt = f"""Analyze this document provided and draft a report of the discussed token: {doc}.
        
Example of some token reports is given below:
        
1. Name of the Protocol: ROSE
1.1. Main Function of the Protocol and Token: 
The Oasis network platform is a layer one blockchain Proof of Stake smart contract platform that provides scalability, extensibility, and privacy. The main feature of the platform enables efficient verifiable and confidential smart contract execution. It aims to achieve this goal by separating its consensus layer from its layer of contract execution while providing a built-in interface connecting the two for privacy-preserving computation. The consensus layer acts as a hub that uses a Proof-of-Stake (PoS) mechanism to secure the network and reach a consensus on transaction validity. The execution layer consists of multiple, parallel runtimes (called ParaTimes) for specialized computation needs that each plug into the consensus layer. 
To access the Oasis Network functions you need to have the ROSE token in your possession. ROSE is used for transaction fees, staking, and delegation at the consensus layer. By staking or staking ROSE, users can secure the Oasis blockchain and earn rewards.
1.2. Shariah Description of the Protocol and Token: 
The Oasis Network is a neutral platform to develop dApps that can be used for halal and haram purposes. The main objective of the platform is to bring more privacy to the users of blockchain. 
The native token ROSE is a hybrid token that can be used as a means of payment and to unlock some utilities. 
1.3. Shariah Opinion of the Protocol and Token: 
There was no shariah issues found at this stage with using the platform and the token in a halal manner. 



2. Name of the Protocol: RADIX
2.1. Main Function of the Protocol and Token: 
Radix is a publicly accessible and decentralized ledger specifically designed to facilitate the development of decentralized applications, particularly those related to decentralized finance (DeFi). By utilizing Radix's comprehensive layer-1 protocol, developers can build these applications without worrying about potential vulnerabilities such as smart contract hacks, exploits, or network congestion.
Radix is building an open, interconnected platform where the full range of powerful DeFi applications will be built securely and safely.
•	Stable Coins
•	Collateralized Lending
•	Perpetual Futures
•	Decentralized Exchanges
•	Wallets & Dashboards
•	Money Markets
•	Yield Farming
•	Options & Derivatives
•	NFTs
•	Gaming
•	DeFi Insurance
•	Portfolio Management
The native crypto currency of the Radix network is called RADIX (XRD) and is required for securing the network via staking, accessing DeFi, deploying smart contracts and paying for transactions.
2.2. Shariah Description of the Protocol and Token:
Radix is a layer one platform with the aim of facilitating the development of fintech on the blockchain. This includes many aspects of haram activities such as derivatives, futures, gaming etc. 
The token is a payment token within the ecosystem.
2.3. Shariah Opinion of the Protocol and Token: 
Because of the nature of the activities that Radix is facilitating, the protocol will not be considered as shariah compliant including its native token.



3. Name of the Protocol: KOINOS
3.1. Main Function of the Protocol and Token: 
Koinos is a blockchain-based decentralized network that aims to create a fee-less and accessible environment for decentralized applications (dApps). It introduces a unique mechanism called "Mana" to dynamically price network resources based on opportunity cost, allowing for free-to-use applications. The consensus algorithm, known as proof-of-burn, maximizes efficiency, decentralization, and egalitarianism while deterring attacks and spam. Koinos utilizes a modular upgradeability approach for seamless and fork-less upgrades. The goal is to provide a user-friendly and truly decentralized blockchain framework that empowers developers and maximizes accessibility for end-users, developers, and node operators.
KOIN is the native token of the Koinos blockchain network and serves several purposes within the ecosystem:
•	Means of exchange
•	Voting rights
•	Earning MANA power
3.2. Shariah Description of the Protocol and Token:
Koinos is a neutral platform with the aim of bringing more people to use blockchain technology. In that sense, it can accommodate halal and haram activities. However, it seems that its main objective is to serve Web3 gaming, Defi, and social apps. This may raise some concerns from the shariah perspective.
The KOIN token is a hybrid token that can be used for payment as well as access to utilities such as governance right.
3.3. Shariah Opinion of the Protocol and Token: 
If the platform is used in a shariah compliant manner along with the token, then there might be no issues. However, it is better to avoid this protocol and token due its focus on gaming and Defi.


Note: Make sure to use accurate information from the knowledge base provided in context and DON'T makeup response from your own knowledge and there is no need for conclusion section.
"""
        analysis_results.append(analyze_with_openai(prompt))
    return analysis_results[0]


def main():
    # Example usage
    token = input("Please provide a token for generating a Shariah compliance report: ")
    documents = gather_information(token)
#     print(documents)
    report = analyze_documents_and_report_generation(documents)
    print(report)

if __name__ == "__main__":
    main()
