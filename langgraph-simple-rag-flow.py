# Load environment variables from .env file
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

qdrant_instance_url = os.getenv('QDRANT_INSTANCE_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# Prepare LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)

# Prepare Embeddings - use the same embedding model as for ingestion
from langchain_mistralai import MistralAIEmbeddings
embed_model = MistralAIEmbeddings()

# Attach our Qdrant Vector store
from langchain_qdrant import QdrantVectorStore
store_wiki = QdrantVectorStore.from_existing_collection(
    collection_name="wiki",
    embedding=embed_model,
    url=qdrant_instance_url,
    api_key=qdrant_api_key,
)

# Create retriever
wiki_retriever = store_wiki.as_retriever(search_kwargs={"k": 1})

# Define prompt for RAG
prompt_template = """You are an assistant for question-answering tasks.
Think carefully about the context.
Just say 'Diese Frage kann ich nicht beantworten' if there is not enough or no context given.
Provide an answer to the user question using only the given context.
Use three sentences maximum and keep the answer concise.

Here is the context to use to answer the question:

{context}

Now, review the user question:

{question}

Write the answer in German. Don't output an English translation.

Answer:"""

# Setup graph
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    generation: str  # LLM generation
    documents: List[str]  # List of retrieved documents

# Helper function
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = wiki_retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = prompt_template.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation}

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generate

# Define the edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()