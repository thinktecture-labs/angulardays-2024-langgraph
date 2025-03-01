# Python
import os
#import argparse
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
# LangChain
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
# LangGraph
from langgraph.graph import START, END, StateGraph
# LangFuse
#from langfuse.callback import CallbackHandler

# set runmode
#parser = argparse.ArgumentParser(description='Script with debug flag') # Create the parser
#parser.add_argument('--testrun', action='store_true', help='Enable real run with predefined question and LangFuse tracing') # Add the testrun argument as a flag (store_true means it will be False by default)
#args = parser.parse_args() # Parse the command-line arguments
# Access the testrun value
#TESTRUN = args.testrun

# initialize Testrun mode with needed settings
#if TESTRUN:
#    print("Testrun mode is enabled")
#    # disable tokenizers parallelism
#    os.environ["TOKENIZERS_PARALLELISM"] = "false"
#    # prepare Langfuse as debugging and tracing framework for our Generative AI application - never develop GenAI apps without that!
#    handler = CallbackHandler()

# Load environment variables
load_dotenv()

# helper function for env vars
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing environment variable: {var_name}")
    return value


qdrant_instance_url = get_env_variable('QDRANT_INSTANCE_URL')
qdrant_api_key = get_env_variable('QDRANT_API_KEY')

# Prepare LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=1000)

# Prepare Embeddings - use the same embedding model as for ingestion
embed_model = MistralAIEmbeddings()

# Attach our Qdrant Vector store
store_wiki = QdrantVectorStore.from_existing_collection(
    collection_name="wiki",
    embedding=embed_model,
    url=qdrant_instance_url,
    api_key=qdrant_api_key,
)

# Create retriever
wiki_retriever = store_wiki.as_retriever(search_kwargs={"k": 1})

# Setup graph
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    generation: str  # LLM generation
    documents: List[str]  # List of retrieved documents
    trust_level: int # 1= Internal source - highly trustworthy, 2= approved externel source - trusted, 3= untrusted external source

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

    # Search for documents
    documents = wiki_retriever.invoke(question)
    
    # Write retrieved documents or 'No content found' to documents key in state
    if not documents:
        documents = [Document(page_content="No content found")]

    return {"documents": documents, "trust_level": 1}


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

    # define answer prompt
    prompt_template = """You are an assistant for question-answering tasks at ACME GmbH.

    # Instructions:
    1. Before answering, thoroughly analyze the given context and ensure your response is directly based on the information provided therein.
    2. Just say 'Diese Frage kann ich nicht beantworten' if there is not enough or no context given.
    3. Provide a detailed answer to the user question using only the given context.
    4. Use up to five sentences and provide explanations or examples to support your answer.
    5. If the context mentions ACME guidelines, make sure to include and explain them in the answer.

    # Context:
    {context}

    # User Question:
    {question}

    # Answer Format:
    - Write the answer in German.
    - Do not output an English translation.
    - Ensure the answer is concise and within five sentences.
    - Include ACME guidelines if mentioned in the context.

    # Answer:
    """

    # RAG generation
    if not documents:
        docs_txt = "No content found"
    else:
        docs_txt = "\n\n".join(
            doc.page_content for doc in documents if hasattr(doc, 'page_content') and doc.page_content)

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

# Test workflow
#if TESTRUN:
#    result = graph.invoke({"question": "Was ist besser signal inputs oder inputs()?"}, config={"callbacks": [handler]})
#    print(result)
#    print("-" * 10)
#    print(result["generation"].content)
