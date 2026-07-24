from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("documents loaders/ML.pdf")
docs = data.load()

print(docs[59])