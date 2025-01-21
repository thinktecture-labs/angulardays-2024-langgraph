# Load environment variables
import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient

class RouterResponse(BaseModel):
    datasource: Literal["websearch", "vectorstore"] = Field(description="The source to route the question to")

class GraderResponse(BaseModel):
    binary_score: Literal["relevant", "not_relevant"] = Field(description="If the retrieved document contains keywords or semantic meaning related to the user question, grade it as relevant; but in any other case, grade it as not relevant")

# Load environment variables
load_dotenv()

# Helper function for environment variables
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing environment variable: {var_name}")
    return value

qdrant_instance_url = get_env_variable('QDRANT_INSTANCE_URL')
qdrant_api_key = get_env_variable('QDRANT_API_KEY')
tavily_api_key = get_env_variable('TAVILY_API_KEY')

# Prepare LLM
llm = ChatMistralAI(model="ministral-8b-latest", temperature=0.1, max_tokens=1500)
llm_question_router = llm.with_structured_output(RouterResponse)
llm_content_grader = llm.with_structured_output(GraderResponse)

# Prepare Embeddings - use the same embedding model as for ingestion
embed_model = MistralAIEmbeddings()

# let's attach our Qdrant Vector store
store_wiki = QdrantVectorStore.from_existing_collection(
    collection_name = "wiki",
    embedding = embed_model,
    url=qdrant_instance_url,
    api_key = qdrant_api_key,
)

# create retriever
wiki_retriever = store_wiki.as_retriever(search_kwargs={"k":1,})

# setup graph
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    answer_grade : str # Retrieved docs good for generation relevant/not_relevant
    documents : List[str] # List of retrieved documents

### Nodes
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
    documents = wiki_retriever.invoke(question) or [Document(page_content="No content found")]
    return {"documents": documents}

def grade(state):
    """
    Grade retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer_grade, that contains grade as relevant or not_relevant
    """
    question = state["question"]
    documents = state["documents"]

    # Doc grader instructions
    doc_grader_instructions = """You are a grader tasked with meticulously and impartially evaluating the relevance of a retrieved document in relation to a user's question. To ensure the highest level of accuracy and usefulness, you should grade a document as relevant only if it provides sufficient and pertinent context that would enable the generation of a comprehensive and highly satisfactory answer to the question. This means the document should contain enough detailed and applicable information that directly addresses the query, allowing for a thorough and well-informed response. If the document lacks adequate context or the necessary details to formulate a very good answer, it should not be considered relevant."""

    # Grader prompt
    doc_grader_prompt = """If the retrieved document contains keywords or semantic meaning related to the user question, grade it as relevant; but in any other case, grade it as not relevant. Here is the user question: \n\n {question}. \n\n Here is the retrieved document: \n\n {document}"""

    # Prepare prompt and run grader
    doc_grader_prompt_formatted = doc_grader_prompt.format(document=documents[0].page_content, question=question)
    result = llm_content_grader.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])

    return {"answer_grade": result.binary_score}

def web_search_angular(state):
    """
    Run web search for Angular content on angular.dev

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    # Instantiating your TavilyClient
    search_client = TavilyClient(api_key=tavily_api_key)

    # Run open web search
    results = search_client.search(question, search_depth="advanced", max_results=3, include_domains=["angular.dev"], include_raw_content=True)

    # List to store the generated Document objects
    documents = []

    # Iterate over each entry in the feed
    for entry in results["results"]:
        # Extract the page content: prefer raw_content, fall back to content, and use default if both are empty
        page_content = entry.get('raw_content') or entry.get('content') or "No content found"

        # Extract metadata
        metadata = {
            "title": entry.get('title', 'No Title'),
            "link": entry.get('url', 'No Link'),
            "score": entry.get('score', '0'),
        }

        # Create a Document object for this entry
        document = Document(page_content=page_content, metadata=metadata)

        # Append the document to the list
        documents.append(document)


    # Write retrieved documents to documents key in state

    return {"documents": documents}

def web_search_full(state):
    """
    Run web search for any content for given question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    # Instantiating your TavilyClient
    search_client = TavilyClient(api_key=tavily_api_key)

    # Run open web search
    results = search_client.search(question, search_depth="advanced", max_results=2)

    # List to store the generated Document objects
    documents = []

    # Iterate over each entry in the feed
    for entry in results["results"]:
        # Extract the page content
        page_content = entry.get('content', 'No content')

        # Extract metadata
        metadata = {
            "title": entry.get('title', 'No Title'),
            "link": entry.get('url', 'No Link'),
            "score": entry.get('score', '0'),
        }

        # Create a Document object for this entry
        document = Document(page_content=page_content, metadata=metadata)

        # Append the document to the list
        documents.append(document)


    # Write retrieved documents to documents key in state

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
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "No content found"

    # define answer prompt
    prompt_template="""You are an assistant for question-answering tasks at ACME GmbH.

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
    rag_prompt_formatted = prompt_template.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"generation": generation}

### Conditional nodes
def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    router_instructions = """You are an expert at routing a user question to a vectorstore or websearch.

    The vectorstore contains documents related to coding, programming, development practices, single page applications, the Angular framework, and coding guidelines for the company ACME.

    Use the vectorstore for any questions containing coding terms, code snippets, programming languages, or technologies relevant to development practices (even in other languages like German).

    If the question is related to coding or development but not specifically covered by the vectorstore, still return 'vectorstore'. Use 'websearch' for non-coding questions."""
    
    route_question = llm_question_router.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    if route_question.datasource == "websearch":
        return "websearch"
    elif route_question.datasource == "vectorstore":
        return "vectorstore"

def decide_retriever_ok(state):
    """
    Determines whether retrieved content is good to generate an answer, or run web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    answer_grade = state["answer_grade"]

    if answer_grade == "not_relevant":
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("generate", generate) # generate
workflow.add_node("grade", grade) # grade
workflow.add_node("web_search_angular", web_search_angular) # websearch angular.dev
workflow.add_node("web_search_full", web_search_full) # full websearch

# Define the edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "web_search_full",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search_full", "generate")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_retriever_ok,
    {
        "websearch": "web_search_angular",
        "generate": "generate",
    },
)
workflow.add_edge("web_search_angular", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
