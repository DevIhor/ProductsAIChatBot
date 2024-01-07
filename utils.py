import csv
import os
import re

import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_FILEPATH = os.path.join(PROJECT_DIRECTORY, ".data", "data.csv")
PRODUCTS_DIRECTORY_FILEPATH = os.path.join(PROJECT_DIRECTORY, ".data", "products")


def download_source_data(data_url: str) -> bool:
    if os.path.isfile(SOURCE_DATA_FILEPATH):
        return False

    response = requests.get(data_url)
    if response.status_code == 200:
        with open(SOURCE_DATA_FILEPATH, 'wb') as f:
            f.write(response.content)
        return True

    raise ValueError(f"There was error during getting source data!\n"
                     f"Code: {response.status_code};\n"
                     f"Content: {response.content}")


def format_products_data() -> bool:
    if not os.path.exists(PRODUCTS_DIRECTORY_FILEPATH):
        os.makedirs(PRODUCTS_DIRECTORY_FILEPATH)

    with open(SOURCE_DATA_FILEPATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for index, row in enumerate(reader):
            if index == 0:
                continue
            category, product, price, overview, specifications, sheets, warranty, availability, link = row
            content = f"The category of the \"{product}\" product is \"{category}\". "
            content += f"The price of the \"{product}\" product is {price}. "
            content += f"Here is the \"{product}\" product overview: \"{overview}\". "
            content += f"Here is the \"{product}\" product specifications: \"{overview}\". "
            content += f"You can find safety data sheets of the \"{product}\" product here: {sheets}. "
            content += f"More about availability of the \"{product}\" product in store - \"{warranty}\". "
            content += f"More about warranty and returns of the \"{product}\" product: \"{availability}\". "
            content += f"You can find more information about the \"{product}\" product here - \"{link}\". "
            with open(os.path.join(PRODUCTS_DIRECTORY_FILEPATH, f"{index}.txt"), 'w') as f:
                f.write(content)
            # break
    return True


def get_product_documents(is_split_on_chunks: bool = False) -> list[Document]:
    files = [os.path.join(PRODUCTS_DIRECTORY_FILEPATH, f)
             for f in os.listdir(PRODUCTS_DIRECTORY_FILEPATH)
             if f.endswith(".txt") and os.path.isfile(os.path.join(PRODUCTS_DIRECTORY_FILEPATH, f))]
    documents = []
    for filepath in files:
        with open(filepath, 'r') as f:
            content = f.read()
            source = re.findall(r'(https?://\S+)', content)
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": source[0] if source else ""},
                )
            )
    if not is_split_on_chunks:
        return documents

    doc_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for doc in documents:
        for chunk in splitter.split_text(doc.page_content):
            doc_chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return doc_chunks
