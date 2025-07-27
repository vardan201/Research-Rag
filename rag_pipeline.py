# %%
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Step 1: Extract text from a single PDF file
def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✅ Step 2: Chunk the text for RAG-style retrieval
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# ✅ Step 3: Manually specify PDF paths
pdf_paths = [
    "1706.03762v7.pdf",
    "1810.04805v2.pdf",
    "2005.14165v4.pdf",
    "2010.11929v2.pdf",
    "2302.13971v1.pdf",
    "GenAI_in_Academic_Writing.pdf"
]

# ✅ Step 4: Process PDFs
all_chunks = []

for pdf_file in pdf_paths:
    print(f"📄 Reading: {pdf_file}")
    full_text = parse_pdf(pdf_file)
    chunks = chunk_text(full_text)
    all_chunks.extend(chunks)

print(f"\n✅ Total chunks created from all PDFs: {len(all_chunks)}")

# %%
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Step 1: Wrap your chunks as LangChain Documents
documents = [Document(page_content=chunk) for chunk in all_chunks]

# Step 2: Use BAAI/bge-base-en-v1.5 embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},  # or "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True}  # recommended for bge models
)

# Step 3: Build FAISS index from documents
vector_store = FAISS.from_documents(documents, embedding_model)

# Step 4: Save FAISS index
vector_store.save_local("faiss_store_bge")


# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15,
        "lambda_mult": 0.7
    }
)


# %%
from langchain_groq import ChatGroq
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ✅ Instantiate Groq LLaMA3
qlm = ChatGroq(
    model_name="llama3-70b-8192"
)

# ✅ Simplified Prompt (good for most LLMs, including LLaMA3)
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Rephrase the following question into 3 different, yet related, search queries. List them without any explanation or numbering.

Question: {question}
"""
)



# ✅ Create chain using qlm
multi_query_chain = LLMChain(llm=qlm, prompt=prompt)

# ✅ Function to generate multiple queries
def generate_multi_queries_fn(user_query: str):
    output = multi_query_chain.run(user_query)
    queries = [line.strip() for line in output.strip().split("\n") if line.strip()]
    return queries[:3]

# ✅ Test run
queries = generate_multi_queries_fn("What causes rainbows to form?")
print("Generated Queries:", queries)


# %%
from langchain.schema import Document

def multi_query_retrieve_fn(user_query: str, top_n: int = 5):
    # Step 1: Generate reformulated search queries
    queries = generate_multi_queries_fn(user_query)

    # Step 2: Initialize list of all retrieved documents
    all_docs = []

    # Step 3: Retrieve from original query
    original_docs = retriever.get_relevant_documents(user_query)
    all_docs.extend(original_docs)

    # Step 4: Retrieve for each reformulated query
    for query in queries:
        docs = retriever.get_relevant_documents(query)
        all_docs.extend(docs)

    # Step 5: Deduplicate based on document content
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # Step 6: Return top-N results
    return unique_docs[:top_n]


# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough,RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
parallel_chain = RunnableParallel({
    "question": RunnablePassthrough(),
    "context": RunnablePassthrough() | multi_query_retrieve_fn | RunnableLambda(format_docs)
})


# %%
parallel_chain.invoke('What Is transformer')

# %%
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable AI assistant tasked with answering questions strictly based on the provided context.

Instructions:
- Use only the context to construct your answer.
- If the context is insufficient, say "The provided context does not contain enough information."
- Be concise, factual, and avoid speculation.
- Do not mention the existence of context in your answer.

Context:
{context}

Question:
{question}

Answer:"""
)


# %%
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables (make sure GROQ_API_KEY is set in your .env file)
load_dotenv()

# Define the LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",  # ✅ Best model from Groq
    temperature=0.2  # Optional: keeps responses factual and deterministic
)

# Optional test prompt (for sanity check)
test_prompt = "Explain the transformer architecture in 200 words."
response = llm.invoke(test_prompt)
print("Response:\n", response.content)


# %%
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Define the final RAG chain
rag_chain = prompt | llm | StrOutputParser()


# %%
final_chain = parallel_chain | rag_chain


# %%




