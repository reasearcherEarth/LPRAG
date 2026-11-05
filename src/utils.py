import statistics
import ast
import json
import torch
from typing import List
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModel

import warnings 
warnings.filterwarnings("ignore")

# used for obtaining viewpoint node embeddings
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_node_embedding(viewpoint_list: List[str], tokenizer: AutoTokenizer, encoder: AutoModel) -> List[torch.Tensor]:
    inputs = tokenizer(viewpoint_list, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    outputs = encoder(**inputs)
    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    embedding_list = [embedding.unsqueeze(0) for embedding in embeddings]

    return embedding_list

# Constructing a viewpoint-graph using node embeddings.
def embedding_based_edge_connection(node_list: List[dict]) -> List[dict]:
    intra_gragh_deg = 5#config["intra_graph_deg"]
    inter_gragh_deg = 10#config["inter_graph_deg"]
    batch_size = -1#config["batch_size"]  # Set batch size for processing nodes in smaller batches

    # Create string_to_index mapping for node_ids
    raw_node_ids = [node["node_id"] for node in node_list]
    node_ids = [re.sub(r'\[\d+\]', '', node_id) for node_id in raw_node_ids]
    paper_id_set = list(set(node_ids))
    string_to_index = {s: i for i, s in enumerate(paper_id_set)}
    node_indices = torch.tensor([string_to_index[node_id] for node_id in node_ids])

    node_embs = torch.stack([torch.tensor(node["embedding"]) for node in node_list]).squeeze(1)
    n = len(node_embs)

    edge_set = set()
    edge_list = []

    def store_edges(topk_indices, start_idx, scores):
        for row_idx, row in enumerate(topk_indices):
            for elem in row:
                tuple_pair = (min(start_idx + row_idx, elem), max(start_idx + row_idx, elem))
                if tuple_pair not in edge_set:
                    edge_set.add(tuple_pair)
                    data = {
                        "node_id1": raw_node_ids[start_idx + row_idx],
                        "node_id2": raw_node_ids[elem],
                        "relation": "unknown",
                        "logical_connector": "unknown",
                        "weight": float(scores[row_idx, elem]),
                    }
                    edge_list.append(data)

    if batch_size == -1:
        batch_size = n
    for start_idx in tqdm(range(0, n, batch_size), desc="batches"):
        end_idx = min(start_idx + batch_size, n)

        # Extract the embeddings for the current batch
        batch_embs = node_embs[start_idx:end_idx]

        # Calculate pairwise similarities with the entire node set
        scores = torch.matmul(batch_embs, node_embs.T)  # Shape: [batch_size, n]

        # Determine intra- and inter-graph connections
        intra_gragh_mask = node_indices[start_idx:end_idx].unsqueeze(1) == node_indices.unsqueeze(0) # Shape: [batch_size, n]
        inter_gragh_mask = ~intra_gragh_mask
        # Fill diagonal for intra-graph (self connections)
        intra_gragh_mask.fill_diagonal_(0)

        # Apply masks to calculate the actual scores
        intra_gragh_scores = scores * intra_gragh_mask.float()
        inter_gragh_scores = scores * inter_gragh_mask.float()

        # Extract top-k connections for intra-graph and inter-graph
        intra_topk_values, intra_topk_indices = torch.topk(intra_gragh_scores, intra_gragh_deg, dim=1)
        inter_topk_values, inter_topk_indices = torch.topk(inter_gragh_scores, inter_gragh_deg, dim=1)

        # Store intra-graph edges
        store_edges(intra_topk_indices, start_idx, intra_gragh_scores)

        # Store inter-graph edges
        store_edges(inter_topk_indices, start_idx, inter_gragh_scores)

    return edge_list


def open_jsonl(path):
    
    decision_mapping = {
        "Reject": 0,
        "Accept (Poster)": 1,
        "Accept (Spotlight)": 2,
        "Accept (Oral)": 3
    }
    processed_dataset = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            ratings = ast.literal_eval(data["ratings"]) if isinstance(data["ratings"], str) else data["ratings"]
            processed_dataset.append({
                "paper_id": data["paper_id"],
                "title": data["title"],
                "abstract": data["abstract"],
                "rating": statistics.mean(ratings),
                "decision": decision_mapping.get(data['decision'], -1)
            })
    
    return processed_dataset