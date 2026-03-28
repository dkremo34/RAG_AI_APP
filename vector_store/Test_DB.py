from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_core.documents import Document 

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")
#embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents= docs, 
    embedding = embeddings,
    persist_directory="chroma_db"
    ) 

vector_store.persist()

response = vector_store.similarity_search("What is Python used for?", k=2)
#for doc in response:
    #print(doc)

retriver = vector_store.as_retriever()
retriver_response = retriver.invoke("What is Python used for?")

for doc in retriver_response:
    print(doc.page_content)