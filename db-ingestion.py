from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DATA_PATH=r"data"
CHROMA_PATH=r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="recruiter_profile",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
pdf_documents = pdf_loader.load()

txt_documents = []
data_path = Path(DATA_PATH)

for txt_file in data_path.glob("*.txt"):
    try:
        loader = TextLoader(str(txt_file))
        docs = loader.load()
        txt_documents.extend(docs)
    except Exception as e:
        print(f"Error loading {txt_file.name}.")\

raw_documents = pdf_documents + txt_documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)