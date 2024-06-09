Workflow and Functionalities of Report Generation

This document outlines the workflow and functionalities of the CryptoAssistant class for generating a Shariah compliance report on a given cryptocurrency token.

1. Initialization
The CryptoAssistant class initializes with API keys for OpenAI and Bing Search, setting up the necessary configuration to interact with these services.

2. Bing Search
The bing_search method is triggered by user input. Specifically, it performs a Bing search using the provided token name combined with specific suffixes such as "official website" and "CoinMarketCap". This strategy ensures more accurate search results by targeting the official and reputable sources of information. The method retrieves only the top two search results for each query, focusing on the most relevant URLs.

3. Gathering Information
The gather_information method takes the list of URLs obtained from the Bing search and scrapes the content from each URL. It differentiates between official site text and whitepaper text, categorizing the information accordingly. This ensures that the analysis is based on both general information and detailed technical documents.

4. Scraping Website Content
The scrape_website method is responsible for scraping the content of a given URL using BeautifulSoup. It cleans and processes the text to remove unnecessary formatting, ensuring that the information is in a usable form for further analysis.

5. Analyzing with OpenAI
The analyze_with_openai method leverages the OpenAI API to analyze the collected information. It constructs a detailed prompt outlining the requirements for Shariah compliance analysis and sends this prompt to the OpenAI model. The model then provides a comprehensive analysis based on the input data.

6. Generating the Shariah Compliance Report
The analyze_documents_and_generate_report method creates a prompt for the OpenAI model using the gathered information. This prompt includes specific instructions for analyzing the protocol and token from a Shariah compliance perspective. The method then calls analyze_with_openai to generate the final report.

7. Main Function
The main function orchestrates the entire process. It prompts the user to provide a token for analysis, performs Bing searches with specific suffixes ("official website" and "CoinMarketCap") to gather relevant URLs, and invokes the necessary methods to gather information and generate the report. The workflow ensures that the assistant systematically collects, analyzes, and reports on the Shariah compliance of the given cryptocurrency token.

Summary
The workflow for generating a Shariah compliance report involves:
1.	Initializing the CryptoAssistant class with the required API keys.
2.	Performing a Bing search triggered by user input, using specific suffixes to gather relevant URLs.
3.	Scraping the content from these URLs.
4.	Using the OpenAI API to analyze the gathered information.
5.	Generating a detailed Shariah compliance report based on the analysis.
This structured approach ensures a thorough evaluation of the Shariah compliance of a cryptocurrency token, utilizing reliable data sources and advanced AI analysis.
