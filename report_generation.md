# Cryptocurrency Shariah Compliance Report Generator

This project is a sophisticated tool designed to automate the process of evaluating cryptocurrencies for Shariah compliance. It utilizes advanced web scraping techniques and natural language processing (NLP) capabilities to source information, analyze it, and generate detailed compliance reports. Below is an overview of its components and the workflow that drives this application.

## Overview

The tool's primary objective is to simplify the task of assessing cryptocurrencies from a Shariah perspective. It achieves this through a sequence of automated tasks:

1. **Information Gathering**: Utilizes the Bing Search API to perform web searches and collect data about specified cryptocurrencies.
2. **Data Analysis**: Leverages OpenAI's powerful GPT models to process and analyze the text information extracted from web searches.
3. **Report Generation**: Compiles the analyzed data into comprehensive reports detailing the Shariah compliance status of the cryptocurrencies in question.

## System Requirements

- Python 3.6 or higher
- External Libraries: Requests, BeautifulSoup4, python-dotenv, langchain-core, langchain-openai

## Setting Up the Project

### Initial Configuration

1. **Repository Cloning**: Begin by cloning this repository to your local environment.
2. **Dependency Installation**: Install all required Python libraries by executing:

    ```bash
    pip install requests beautifulsoup4 python-dotenv langchain-core langchain-openai
    ```

3. **API Keys Configuration**: Secure API keys for both Bing Search and OpenAI. Then, create a `.env` file at the root of your project directory and include your API keys as follows:

    ```
    OPENAI_API_KEY=<your_openai_api_key>
    BING_API_KEY1=<your_bing_search_api_key>
    ```

### Workflow Explanation

1. **Environment Setup**: The script begins by loading necessary environment variables, including API keys for Bing Search and OpenAI, ensuring that these services can be accessed securely.

2. **Session Initialization**: Initializes a session for HTTP requests, specifically for communicating with the Bing Search API.

3. **Web Scraping**: When a cryptocurrency token is specified, the script performs a web search to gather relevant information. It processes the search results to extract and clean text data from the first web page returned.

4. **Analysis with OpenAI**: The gathered text is then sent to OpenAI's GPT model. The script is pre-configured to instruct the model to focus on generating information pertinent to Shariah compliance in the context of cryptocurrencies.

5. **Report Generation**: The model's output is structured into a detailed report, covering various aspects like the main function of the protocol, its Shariah description, and an overall Shariah opinion.

## Usage Guide

To generate a report, simply run the script from your command line:

```bash
python src\report_generation.py
