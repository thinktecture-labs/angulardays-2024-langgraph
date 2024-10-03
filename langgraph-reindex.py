# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()

# Retrieve environment variables
qdrant_instance_url = os.getenv('QDRANT_INSTANCE_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# Prepare Embeddings - use the same embedding model as for ingestion
from langchain_mistralai import MistralAIEmbeddings
embed_model = MistralAIEmbeddings()

# setup graph
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    generation : str # LLM generation

from langgraph.graph import START, END

### Nodes
def reindex(state):
    """
    re-builds Qdrant vectorstore for wiki content

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, containing a finish message
    """

    # Load RSS feed
    from langchain_community.document_loaders import RSSFeedLoader
    urls = ["https://demowiki.webstage.work/category/angular/feed/"]
    loader = RSSFeedLoader(urls=urls, show_progress_bar=False)
    data = loader.load()

    # Re-build Qdrant vectore store
    from langchain_qdrant import QdrantVectorStore

    store_wiki = QdrantVectorStore.from_documents(
        data,
        embed_model,
        url=qdrant_instance_url,
        api_key=qdrant_api_key,
        collection_name="wiki",
        force_recreate=True,
    )
    return {"generation": data}

from langgraph.graph import StateGraph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("reindex", reindex) # reindex

# Define the edges
workflow.add_edge(START, "reindex")
workflow.add_edge("reindex", END)

# Compile
graph = workflow.compile()
