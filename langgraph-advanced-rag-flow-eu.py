# Python
import os
import argparse
import requests
from dotenv import load_dotenv
from typing import Dict, Literal, List, Annotated
from typing_extensions import TypedDict
# Pydantic
from pydantic import BaseModel, Field
# LangChain
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
# LangGraph
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
# LangFuse
from langfuse.callback import CallbackHandler
# Tavily
from tavily import TavilyClient

# set runmode
parser = argparse.ArgumentParser(description='Script with debug flag') # Create the parser
parser.add_argument('--testrun', action='store_true', help='Enable real run with predefined question and LangFuse tracing') # Add the testrun argument as a flag (store_true means it will be False by default)
args = parser.parse_args() # Parse the command-line arguments
# Access the testrun value
TESTRUN = args.testrun

# initialize Testrun mode with needed settings
if TESTRUN:
    print("Testrun mode is enabled")
    # disable tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # prepare Langfuse as debugging and tracing framework for our Generative AI application - never develop GenAI apps without that!
    handler = CallbackHandler()

# models
class RouterResponse(BaseModel):
    datasource: Literal["external", "internal_db", "internal_tools"] = Field(description="The source to route the question to")

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
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0, max_tokens=1000)
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

# LangGraph elements
## Graph State
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    answer_grade : str # Retrieved docs good for generation relevant/not_relevant
    documents : List[str] # List of retrieved documents
    trust_level: int # 1= Internal source - highly trustworthy, 2= approved externel source - trusted, 3= untrusted external source
    project_symbol : str
    project_id : str
    project_name : str
    project_manager_details : Dict[str, str]
    messages: Annotated[List[AnyMessage], add_messages]

## tools for workflow
def get_project_details_by_project_symbol(project_symbol: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Lookup the project details for a given project symbol
    Returns a dictionary with project_symbol, project_id and project_name
    For unknown project symbols returns {"project_symbol": None, "project_id": None, "project_name": None}
    """
    # call API for project details
    api_url = f"https://tt-project-api.azurewebsites.net/projects/{project_symbol.lower()}"
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            p_symbol = data.get('project_symbol')
            p_id = data.get('project_id')
            p_name = data.get('project_name')
        else:
            # API returned error or no data found
            p_symbol, p_id, p_name = None, None, None
    except Exception as e:
        # Handle any exceptions (connection errors, etc.)
        print(f"Error fetching project details: {e}")
        p_symbol, p_id, p_name = None, None, None

    # return results to state and update message stack
    return Command(
        update={
            # update state for project details
            "project_symbol": p_symbol,
            "project_id": p_id,
            "project_name": p_name,
            "trust_level": 1,
            # update the message history
            "messages": [ToolMessage(f"Found project details for project_symbol: {p_symbol} project_id: {p_id} project_name: {p_name}", tool_call_id=tool_call_id)]
        }
    )

def get_manager_details_by_project_id(project_id: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Lookup manager details based on project_id.
    Returns manager details for ids not None, empty dict otherwise.
    """
    # call API for manager details
    api_url = f"https://tt-project-api.azurewebsites.net/project-chairs/{project_id.lower()}"
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            manager_details = response.json()
        else:
            # API returned error or no data found
            manager_details = {}
    except Exception as e:
        # Handle any exceptions (connection errors, etc.)
        print(f"Error fetching project details: {e}")
        manager_details = {}
        
    # return results to state and update message stack
    return Command(
        update={
            # update state for manager details
            "project_manager_details": manager_details,
            "trust_level": 1,
            # update the message history
            "messages": [ToolMessage(f"Found manager details for project_id {project_id}: {manager_details}", tool_call_id=tool_call_id)]
        }
    )

## tool nodes
def prepare_tool_use(state: GraphState):
    """Create a user message for tool use."""
    question = state["question"]
    return {"messages": [HumanMessage(content=f"Look up information for question: {question}")]}

def tool_time(state: MessagesState):
  messages = state['messages']
  response = llm_tools.invoke(messages)
  return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tools", "generate"]:
  messages = state['messages']
  last_message = messages[-1]
  if last_message.tool_calls:
    return "tools"
  return "generate"

## classic nodes
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
    return {"documents": documents, "trust_level": 1}

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
    return {"documents": documents, "trust_level": 2}

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
    return {"documents": documents, "trust_level": 3}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    # Check if 'documents' key exists in state, otherwise assign an empty list
    documents = state.get("documents", [])
    project_symbol = state.get("project_symbol", None)
    project_id = state.get("project_id", None)
    project_name = state.get("project_name", None)
    project_manager_details = state.get("project_manager_details", {})

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
    # Prepare context string: either project details or retrieved documents depending on state
    if documents == []:
        docs_txt = f"Project details: Project Symbol:{project_symbol} Project ID: {project_id} Project Name: {project_name} Project Manager Details: {project_manager_details}"
    else:
        docs_txt = "\n\n".join(doc.page_content for doc in documents) if documents else "No content found"
    
    # RAG generation
    rag_prompt_formatted = prompt_template.format(context=docs_txt, question=question)
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
    router_instructions = """You are an expert at routing a user question depending on the intent. Choose the most appropriate datasource:

    'internal_db': documents related to coding, code snippets, programming, programming languages, development practices, single page applications, the Angular framework, and coding guidelines for the company ACME.
    'internal_tools': information about projects (project names, project ids, project manager and their contact details) by ACME GmbH
    'external': general information not covered by the topics covered by 'internal_db' or 'interal_tools'.
    
    If the question is related to coding or development but not specifically covered by the 'internal_db', still return 'internal_db'."""
    
    route_question = llm_question_router.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    if route_question.datasource == "internal_db":
        return "internal_db"
    elif route_question.datasource == "internal_tools":
        return "internal_tools"
    elif route_question.datasource == "external":
        return "external"

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

# create tool array, tool llm and tool node with all tools
tools = [get_project_details_by_project_symbol, get_manager_details_by_project_id]
llm_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# add nodes to workflow
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("generate", generate) # generate
workflow.add_node("grade", grade) # grade
workflow.add_node("web_search_angular", web_search_angular) # websearch angular.dev
workflow.add_node("web_search_full", web_search_full) # full websearch
workflow.add_node("prepare_tool_use", prepare_tool_use)
workflow.add_node("tool_time", tool_time)
workflow.add_node("tools", tool_node)

# Define the edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "external": "web_search_full",
        "internal_db": "retrieve",
        "internal_tools": "prepare_tool_use",
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
workflow.add_edge("prepare_tool_use", "tool_time")
workflow.add_conditional_edges("tool_time", should_continue)
workflow.add_edge("tools", "tool_time")
workflow.add_edge("web_search_angular", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()

# Test workflow
if TESTRUN:
    result = graph.invoke({"question": "Wie lautet Vorname und Telefonnummer des Ansprechpartners für das Post-Projekt?"}, config={"callbacks": [handler]})
    #result = graph.invoke({"question": "test"}, config={"callbacks": [handler]})
    # output everything
    print(result)
    print("-" * 10)
    print(result["generation"].content)
