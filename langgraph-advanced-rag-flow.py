# Load environment variables from .env file
from dotenv import load_dotenv
import os, json
load_dotenv()

# Retrieve environment variables
qdrant_instance_url = os.getenv('QDRANT_INSTANCE_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# prepare LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=2500)
llm_json_mode = llm.bind(response_format={"type": "json_object"})

# Prepare Embeddings - use the same embedding model as for ingestion
from langchain_mistralai import MistralAIEmbeddings
embed_model = MistralAIEmbeddings()

# let's attach our Qdrant Vector store
from langchain_qdrant import QdrantVectorStore
store_wiki = QdrantVectorStore.from_existing_collection(
    collection_name = "wiki",
    embedding = embed_model,
    url=qdrant_instance_url,
    api_key = qdrant_api_key,
)

# create retriever
wiki_retriever = store_wiki.as_retriever(search_kwargs={"k":1,})

# setup graph
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    answer_grade : str # Retrieved docs good for generation relevant/not_relevant
    documents : List[str] # List of retrieved documents

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

### Helper function
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

### Nodes
### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = wiki_retriever.invoke(question)
    return {"documents": documents}

def grade(state):
    """
    Grade retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer_grade, that contains grade as relevant or not_relevant
    """
    print("---GRADE---")
    question = state["question"]
    documents = state["documents"]

    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing carefully and objectively the relevance of a retrieved document to a user question.

    If the document contains keywords or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    Return JSON with single key, binary_score, that is 'relevant' or 'not_relevant' score to indicate whether the document contains at least some information that is relevant to the question."""

    # Prepare prompt and run grader
    doc_grader_prompt_formatted = doc_grader_prompt.format(document=documents[0].page_content, question=question)
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    return {"answer_grade": json.loads(result.content)['binary_score']}

def web_search_angular(state):
    """
    Run web search for Angular content on angular.dev

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---WEB SEARCH ANGULAR---")
    question = state["question"]
    # Instantiating your TavilyClient
    from tavily import TavilyClient
    search_client = TavilyClient(api_key=tavily_api_key)

    # Run open web search
    from langchain.schema import Document
    results = search_client.search(question, search_depth="advanced", max_results=2, include_domains=["angular.dev"], include_raw_content=True)

    # List to store the generated Document objects
    documents = []

    # Iterate over each entry in the feed
    for entry in results["results"]:
        # Extract the page content
        page_content = entry.get('raw_content', 'No content')

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

    return{"documents": documents}

def web_search_full(state):
    """
    Run web search for any content for given question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---WEB SEARCH FULL---")
    question = state["question"]
    # Instantiating your TavilyClient
    from tavily import TavilyClient
    search_client = TavilyClient(api_key=tavily_api_key)

    # Run open web search
    from langchain.schema import Document
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

    return{"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # define answer prompt
    prompt_template = """You are an assistant for question-answering tasks at ACME GmbH.
      Think carefully about the context.
      Just say 'Diese Frage kann ich nicht beantworten' if there is not enough or no context given.
      Provide an answer to the user question using only the given context.
      Use three sentences maximum and keep the answer concise.
      If the context mentions ACME guidelines, try to include it in the answer.
      Here is the context to use to answer the question:

      {context}

      Now, review the user question:

      {question}

      Write the answer in German. Don't output an English translation.

      Answer:"""

    # RAG generation
    docs_txt = format_docs(documents)
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

    print("---ROUTE QUESTION---")
    router_instructions = """You are an expert at routing a user question to a vectorstore or websearch.

    The vectorstore contains documents related to coding, programming, development practices, single page applications, the Angular framework, and coding guidelines for the company ACME.

    Use the vectorstore for any questions containing coding terms, code snippets, programming languages, or technologies relevant to development practices (even in other languages like German).

    If the question is related to coding or development but not specifically covered by the vectorstore, still return 'vectorstore'. Use 'websearch' for non-coding questions.

    Return JSON with a single key, "datasource," that is 'websearch' or 'vectorstore' depending on the question."""

    route_question_result = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question_result.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB_SEARCH_FULL---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RETRIEVER---")
        return "vectorstore"

def decide_retriever_ok(state):
    """
    Determines whether retrieved content is good to generate an answer, or run web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---DECIDE RETRIEVER OK---")
    answer_grade = state["answer_grade"]

    if answer_grade.lower() == "not_relevant":
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

from langgraph.graph import StateGraph
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
