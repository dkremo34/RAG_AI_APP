from langchain_community.document_loaders import TextLoader

data = TextLoader("documents_loader/notes.txt")
docs =data.load()
print(docs[0].page_content)