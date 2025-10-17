"""
intfloat/multilingual-e5-large ranker
"""

import json, os
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Optional
import psutil

# llmaindex
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.schema import Document, MetadataMode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import (
    ServiceContext,
    PromptHelper,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.text_splitter import SentenceSplitter
from config import is_STAGING

STAGING = is_STAGING


def save_list_to_json(lst, filename):
    """Save Files"""
    with open(filename, "w") as file:
        json.dump(lst, file)


def wr_dict(filename, dic):
    """Write Files"""
    try:
        if not os.path.isfile(filename):
            data = []
            data.append(dic)
            with open(filename, "w") as f:
                json.dump(data, f)
        else:
            with open(filename, "r") as f:
                data = json.load(f)
                data.append(dic)
            with open(filename, "w") as f:
                json.dump(data, f)
    except Exception as e:
        print("Save Error:", str(e))
    return


def rm_file(file_path):
    """Delete Files"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")


class JSONReader:
    """JSON reader.
    Reads JSON documents with options to help suss out relationships between nodes.
    """

    def __init__(
        self,
        is_jsonl: Optional[bool] = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.is_jsonl = is_jsonl

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        documents = []
        with open(input_file, "r") as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {
                "title": data["title"],
                "published_at": data["published_at"],
                "source": data["source"],
            }
            documents.append(Document(text=data["body"], metadata=metadata))
        return documents


class CustomExtractor(BaseExtractor):
    async def aextract(self, nodes) -> List[Dict]:
        metadata_list = [
            {
                "title": (node.metadata["title"]),
                "source": (node.metadata["source"]),
                "published_at": (node.metadata["published_at"]),
            }
            for node in nodes
        ]
        return metadata_list


def start_retrival(corpus, queries, rank_model_name, output_name):
    model_name = rank_model_name
    topk = 10
    chunk_size = 256
    context_window = 2048
    num_output = 256
    save_file = output_name

    print("Remove save file if exists.")
    rm_file(save_file)

    embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

    # Create a service context
    text_splitter = SentenceSplitter(chunk_size=chunk_size)

    prompt_helper = PromptHelper(
        context_window=context_window,
        num_output=num_output,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )

    service_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
    )

    set_global_service_context(service_context)

    # Loading corpus
    reader = JSONReader()
    data = reader.load_data(corpus)
    print("Corpus Data")
    print("--------------------------")
    print(data[0])
    print("--------------------------")

    # Now initialise the model.
    # Finetune the embeddings for our corpus.
    print("Initialising pipeline")
    transformations = [text_splitter, CustomExtractor()]
    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(documents=data)
    nodes_see = deepcopy(nodes)
    print("LLM sees:\n", (nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM))
    print("Finished Loading...")

    index = VectorStoreIndex(nodes, show_progress=True)
    print("Vector Store Created ...")

    # Now we are finally ready to parse the queries.
    with open(queries, "r") as file:
        query_data = json.load(file)

    print("Query Data")
    print("--------------------------")
    print(query_data[0])
    print("--------------------------")

    # Run the retrieval.
    retrieval_save_list = []
    print("Running Retrieval ...")
    for data in tqdm(query_data):
        query = data["query"]
        nodes_score = index.as_retriever(similarity_top_k=topk).retrieve(query)

        retrieval_list = []
        for ns in nodes_score:
            dic = {}
            dic["text"] = ns.get_content(metadata_mode=MetadataMode.LLM)
            dic["score"] = ns.get_score()
            retrieval_list.append(dic)

            save = {}
            save["query"] = data["query"]
            save["answer"] = data["answer"]
            save["question_type"] = data["question_type"]
            save["retrieval_list"] = retrieval_list
            save["gold_list"] = data["evidence_list"]
            retrieval_save_list.append(save)

    print("Retieval complete. Saving Results")
    with open(save_file, "w") as json_file:
        json.dump(retrieval_save_list, json_file)


if __name__ == "__main__":
    if STAGING:
        corpus = "data/sample-corpus.json"
        queries = "data/sample-rag.json"
    else:
        corpus = "data/corpus.json"
        queries = "data/rag.json"

    rank_model_name = "intfloat/multilingual-e5-large"

    if STAGING:
        output_name = "output/staging/rankerD.json"
    else:
        output_name = "output/production/rankerD.json"

    start_retrival(corpus, queries, rank_model_name, output_name)

    if STAGING == True:
        process = psutil.Process(os.getpid())
        peak_wset_bytes = process.memory_info().rss
        # peak_wset represents the peak working set size in bytes.
        peak_wset_gb = peak_wset_bytes / (1024 * 1024 * 1024)
        print(f"Peak working set size: {peak_wset_gb:.2f} GB")
