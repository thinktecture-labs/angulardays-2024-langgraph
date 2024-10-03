import os, feedparser

from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
qdrant_instance_url = os.getenv('QDRANT_INSTANCE_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# Prepare Embeddings - use the same embedding model as for ingestion
from langchain_mistralai import MistralAIEmbeddings
embed_model = MistralAIEmbeddings()

# setup graph

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    generation : str # LLM generation
    documents : List[str] # List of retrieved documents

from langgraph.graph import START, END

### Nodes
def rssloader(state):
    """
    loads the rss feed with wiki content and creates langchain documents from it

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains loaded documents from rss feed
    """
    # Parse the RSS feed
    feed_url = "https://demowiki.webstage.work/category/angular/feed/"
    feed = feedparser.parse(feed_url)
    results = feed.entries

    # List to store the generated Document objects
    documents = []

    # Iterate over each entry in the feed
    for entry in results:
        # Extract the page content
        if 'content' in entry and entry.content:
            page_content = entry.content[0]['value']
        else:
            page_content = entry.get('summary', '')  # Fallback to summary if no content is present

        # Extract metadata
        metadata = {
            "title": entry.get('title', 'No Title'),
            "link": entry.get('link', 'No Link'),
            "author": entry.get('author', 'Unknown Author'),
            "publish_date": entry.get('published', 'No Date'),
            "feed": feed_url
        }

        # Create a Document object for this entry
        document = Document(page_content=page_content, metadata=metadata)

        # Append the document to the list
        documents.append(document)
    return {"documents": documents}

def reindex(state):
    """
    re-builds Qdrant vectorstore for wiki content

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, containing a finish message
    """

    # get documents from state
    documents = state["documents"]

    # Re-build Qdrant vectore store
    from langchain_qdrant import QdrantVectorStore

    store_wiki = QdrantVectorStore.from_documents(
        documents,
        embed_model,
        url=qdrant_instance_url,
        api_key=qdrant_api_key,
        collection_name="wiki",
        force_recreate=True,
    )
    return {"generation": documents}

from langgraph.graph import StateGraph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("rssloader", rssloader) # rssloader
workflow.add_node("reindex", reindex) # reindex

# Define the edges
workflow.add_edge(START, "rssloader")
workflow.add_edge("rssloader", "reindex")
workflow.add_edge("reindex", END)

# Compile
graph = workflow.compile()
