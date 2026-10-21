# Index sentences, retrieve with surrounding context window
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

llm = Ollama(model=os.getenv("OLLAMA_LLM", "qwen3.5:4b"), base_url="http://localhost:11434", request_timeout=500.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest", base_url="http://localhost:11434")

BASE_DIR = Path(__file__).resolve().parent.parent
MEMO_PATH = BASE_DIR / "datasets" / "acme_q3_strategy_memo.txt"
def load_memo(file_path: Path) -> str:
    """Read and return document contents from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find memo file at: {file_path}\n"
            f"Please ensure 'acme_q3_strategy_memo.txt' exists in the 'datasets/' folder."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
ACME_MEMO = load_memo(MEMO_PATH)

def build_sentence_window_index(window_size: int =3):
    """
    Build an index where each node is a sentence,
    but retrieval retruns the sentence + surrounding window
    """
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size = window_size, # Sentence on each side
        window_metadata_key = "window", # Store window in metadata
        original_text_metadata_key = "original_sentence",
    )
    docs = [Document(text=ACME_MEMO)]
    nodes = node_parser.get_nodes_from_documents(docs)
    print(f"Sentences indexed: {len(nodes)}")
    print(f"Window size: {window_size} sentences each side")

    # Build index on individual sentences
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    #Post processor swaps the sentence for its window at retrieval time
    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key = "window"
    )
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k =3,
        node_postprocessors=[postprocessor],
    )
    return query_engine

if __name__ == "__main__":
    engine = build_sentence_window_index(window_size=2)
    response = engine.query("What is the competitive situation for Product Alpha?")
    print(f"\nAnswer: {response}")