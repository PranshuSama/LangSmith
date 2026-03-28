

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_PROJECT"] = 'RAG Chatbot'

load_dotenv()  # expects GROQ_API_KEY (chat + optionally embeddings)

PDF_PATH = "/Users/pranshusama/Downloads/islr.pdf"


def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()

    if provider == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required when EMBEDDING_PROVIDER=groq")

        groq_embedding_model = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
        print(f"Using Groq embeddings: {groq_embedding_model}")

        return OpenAIEmbeddings(
            model=groq_embedding_model,
            openai_api_key=groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            # Groq embedding API expects strings, not token-id arrays.
            check_embedding_ctx_length=False,
        )

    if provider == "huggingface":
        hf_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"Using HuggingFace embeddings: {hf_model}")
        try:
            return HuggingFaceEmbeddings(
                model_name=hf_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                "HuggingFace embeddings require sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    raise ValueError("EMBEDDING_PROVIDER must be 'groq' or 'huggingface'")

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = get_embeddings()
try:
    vs = FAISS.from_documents(splits, emb)
except Exception as exc:
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
    if provider == "groq":
        print("Groq embedding call failed. Falling back to HuggingFace embeddings.")
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        emb = get_embeddings()
        vs = FAISS.from_documents(splits, emb)
    else:
        raise exc
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)