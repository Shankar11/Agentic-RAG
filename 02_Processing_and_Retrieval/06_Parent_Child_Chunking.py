# Small chunks for retrieval, large chunks for generation
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext

import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure LlamaIndex to use Ollama LLM + Ollama embeddings
llm = Ollama(
    model=os.getenv("OLLAMA_LLM", "qwen3.5:4b"),
    base_url="http://localhost:11434",
    request_timeout=120.0,
)
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

Settings.llm = llm
Settings.embed_model = embed_model

ACME_MEMO = """Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and
Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M.
Key risk: Dataview Inc. launched a competing product at 40% lower price point.
Three pipeline deals ($2.1M combined) are now in competitive review.
Product Beta (SMB Reporting Tool)
Q2 revenue: $3.8M (6% below forecast). Churn increased to 4.2% monthly.
Root cause: poor onboarding — 62% of churned customers never completed setup.
Launching Beta Quick Start program in Q3. Target: reduce churn to 2.5%.
Budget: $1.8M for onboarding improvement, $600K for UX redesign.
Product Gamma (AI Data Assistant)
Private beta with 12 design partners. Launch: September 15, 2025.
9 of 12 partners rated natural language query as transformative.
Pricing: $45/user/month (teams), $120/user/month (enterprise).
Q3 investment: $3.2M across product marketing, developer relations, launch events."""

def build_parent_child_index():
    """
    Build a hierarchical index with parent and child chunks.
    Child chunks (small) are indexed. Parent chunks (large) are retrieved.
    """
    # Define chunk sizes: [parent_size, child_size]
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 256]  # parent=1024, child=256
    )

    docs = [Document(text=ACME_MEMO)]
    nodes = node_parser.get_nodes_from_documents(docs)
    leaf_nodes = get_leaf_nodes(nodes)  # Child chunks
    print(f"Total nodes: {len(nodes)}")
    print(f"Leaf (child) nodes: {len(leaf_nodes)}")
    
    # Store ALL nodes (parents + children) in docstore
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    
    # Index ONLY child nodes for retrieval
    index = VectorStoreIndex(
        leaf_nodes,
        embed_model=embed_model,
    )
    
    # AutoMergingRetriever: retrieves children, returns parents
    base_retriever = index.as_retriever(similarity_top_k=6)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        simple_ratio_thresh=0.5,  # Merge if >50% of children retrieved
        verbose=True,
    )
    return retriever

def query_parent_child(retriever, query: str):
    """Retrieve child chunks, get parent context, generate answer."""
    nodes = retriever.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(nodes)} node(s) (after merging to parents):")
    for node in nodes:
        print(f" [{len(node.text)} chars] {node.text[:80]}...")
    return nodes

if __name__ == "__main__":
    retriever = build_parent_child_index()
    query_parent_child(retriever, "What are the Q3 risks for Product Alpha?")
    query_parent_child(retriever, "What is the revenue performance across all products?")