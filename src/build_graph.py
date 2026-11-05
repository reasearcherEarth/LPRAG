import json
import logging
import warnings
from typing import Dict, List, Tuple

import torch
import yaml
from torch_geometric.data.data import Data
from torch_geometric.utils import remove_self_loops
from transformers import AutoModel, AutoTokenizer

from src.utils import embedding_based_edge_connection, get_node_embedding

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONTRIEVER_MODEL_NAME = "facebook/contriever"
QUERIES_PER_DOCUMENT = 2  


# ---------------------------------------------------------------------------
# Config / IO
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_encoder() -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_MODEL_NAME)
    encoder = AutoModel.from_pretrained(CONTRIEVER_MODEL_NAME)
    return tokenizer, encoder


def build_nodes_for_document(doc_key: str, chunk_groups: List[list], tokenizer, encoder) -> List[dict]:
    chunk_list = [chunk for group in chunk_groups for chunk in group]
    embeddings = get_node_embedding(chunk_list, tokenizer, encoder)

    node_list = []
    for idx, (viewpoint, embedding) in enumerate(zip(chunk_list, embeddings)):
        node_list.append(
            {
                "viewpoint": viewpoint,
                "embedding": embedding.tolist(),
                "node_id": f"{doc_key}[{idx}]",
                "source": "extracted",
            }
        )
    return node_list


def build_nodes_and_edges(
    chunks: Dict[str, list], tokenizer, encoder
) -> Tuple[List[List[dict]], List[list]]:

    logger.info("Generating connection chunks...")
    nodes_by_document = []
    edges_by_document = []

    for doc_key, chunk_groups in chunks.items():
        node_list = build_nodes_for_document(doc_key, chunk_groups, tokenizer, encoder)
        nodes_by_document.append(node_list)
        edges_by_document.append(embedding_based_edge_connection(node_list))

    logger.info("connection done!")
    return nodes_by_document, edges_by_document

def build_global_node_index(chunks: Dict[str, list], edges_by_document: List[list]) -> Dict[str, int]:

    used_node_ids = set()
    for edges in edges_by_document:
        for connection in edges:
            used_node_ids.add(connection["node_id1"])
            used_node_ids.add(connection["node_id2"])

    global_index: Dict[str, int] = {}
    next_idx = 0
    for doc_key in chunks.keys():
        local_idx = 0
        while f"{doc_key}[{local_idx}]" in used_node_ids:
            global_index[f"{doc_key}[{local_idx}]"] = next_idx
            next_idx += 1
            local_idx += 1

    return global_index


def build_edge_tensors(
    edges_by_document: List[list], global_index: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_pairs: List[List[int]] = []
    edge_weight_by_pair: Dict[str, float] = {}

    for edges in edges_by_document:
        for connection in edges:
            edge = sorted([global_index[connection["node_id1"]], global_index[connection["node_id2"]]])
            edge_pairs.append(edge)
            edge_weight_by_pair[str(edge)] = connection["weight"]

    edge_index = torch.tensor(edge_pairs).T
    edge_index = torch.unique(edge_index.T, dim=0).T  # remove duplicate
    edge_index, _ = remove_self_loops(edge_index)

    edge_attr = torch.tensor(
        [edge_weight_by_pair[str(sorted(edge.tolist()))] for edge in edge_index.T]
    )
    return edge_index, edge_attr


def build_node_features(nodes_by_document: List[List[dict]]) -> torch.Tensor:
    features = [node["embedding"][0] for node_list in nodes_by_document for node in node_list]
    return torch.tensor(features).float()


def build_chunk_graph(chunks: Dict[str, list], tokenizer, encoder) -> Data:
    nodes_by_document, edges_by_document = build_nodes_and_edges(chunks, tokenizer, encoder)

    logger.info("Generating edges...")
    global_index = build_global_node_index(chunks, edges_by_document)
    edge_index, edge_attr = build_edge_tensors(edges_by_document, global_index)
    x = build_node_features(nodes_by_document)
    logger.info("Edges done!!")

    logger.info("Generating graph with chunks and queries...")
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    valid_edges_mask = graph.edge_attr > 0.0
    graph.edge_index = graph.edge_index.T[valid_edges_mask, :].T
    graph.edge_attr = graph.edge_attr[valid_edges_mask]

    logger.info("Graph done!!")
    logger.info(graph)
    return graph


def embed_queries(questions: Dict[str, list], tokenizer, encoder) -> torch.Tensor:
    all_queries = [query for doc_queries in questions.values() for query in doc_queries]

    query_embeddings = [
        list(get_node_embedding(query, tokenizer, encoder)[0][0].detach().numpy())
        for query in all_queries
    ]
    return torch.tensor(query_embeddings).float()


def add_query_nodes(
    graph: Data,
    questions: Dict[str, list],
    tokenizer,
    encoder,
    queries_per_document: int = QUERIES_PER_DOCUMENT,
) -> Data:
    
    logger.info("Generating graph with queries syntethic...")

    num_chunk_nodes = graph.x.shape[0]
    query_embeddings = embed_queries(questions, tokenizer, encoder)
    graph.x = torch.cat((graph.x, query_embeddings), dim=0)

    edges = []
    next_query_node_id = num_chunk_nodes
    for doc_key in questions.keys():
        for i in range(queries_per_document):
            edges.append([int(doc_key), next_query_node_id + i])
        next_query_node_id += queries_per_document

    new_edges = torch.tensor(edges).T
    graph.edge_index = torch.cat((graph.edge_index, new_edges), dim=1)

    logger.info("Graph done!!")
    logger.info(graph)
    return graph

def main():
    config = load_config()
    tokenizer, encoder = build_encoder()

    chunks = load_json(config["path_chunks"])
    graph = build_chunk_graph(chunks, tokenizer, encoder)
    torch.save(graph, config["graph_chunks_destiny"])
    logger.info("Graph chunks save in %s", config["graph_chunks_destiny"])

    questions = load_json(config["destiny_path_syn_query"])
    graph = add_query_nodes(graph, questions, tokenizer, encoder)
    torch.save(graph, config["graph_chunks_plus_queries_destiny"])
    logger.info("Graph with queries save in %s", config["graph_chunks_plus_queries_destiny"])


if __name__ == "__main__":
    main()