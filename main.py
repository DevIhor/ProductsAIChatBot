import os
import warnings

from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import chain

from utils import get_product_documents, download_source_data, format_products_data

load_dotenv()
warnings.filterwarnings("ignore")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LARGE_DOCS_MODE = True

if __name__ == "__main__":
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0),
    )

    download_source_data(
        'https://docs.google.com/spreadsheets/d/17vDU2Ud9VhFeyG2yKwrFBaR56lLwya7xntI7oQSU-Hg/export?format=csv'
    )
    format_products_data()
    documents = get_product_documents(LARGE_DOCS_MODE)
    if LARGE_DOCS_MODE:
        search_index = FAISS.from_documents(documents, OpenAIEmbeddings())

    def print_answer(question: str) -> None:
        if LARGE_DOCS_MODE:
            input_documents = search_index.similarity_search(question, k=4)
        else:
            input_documents = documents

        result = chain(
            {
                "input_documents": input_documents,
                "question": question,
            }, return_only_outputs=True,
        )["output_text"].split("\nSOURCES: ")[0]
        print(result)

    while content := input("\nPlease enter question or \"exit\" to exit the script:\n\n"):
        if content == "exit":
            break
        print_answer(f"\n{content}")
