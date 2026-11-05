import torch
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import train_test_split_edges
import argparse
import json
from tqdm import tqdm

def common_neighbors(graph, u, v):
    """
    Compute Common Neighbors score for nodes u and v.
    Based on: Liben-Nowell and Kleinberg, "The Link Prediction Problem for Social Networks" (2003)
    """
    return len(list(nx.common_neighbors(graph, u, v)))

def jaccard_coefficient(graph, u, v):
    """
    Compute Jaccard Coefficient score for nodes u and v.
    Based on: Liben-Nowell and Kleinberg, "The Link Prediction Problem for Social Networks" (2003)
    """
    u_neighbors = set(graph.neighbors(u))
    v_neighbors = set(graph.neighbors(v))
    intersection = len(u_neighbors.intersection(v_neighbors))
    union = len(u_neighbors.union(v_neighbors))
    return intersection / union if union != 0 else 0

def adamic_adar_index(graph, u, v):
    """
    Compute Adamic-Adar score for nodes u and v.
    Based on: Adamic and Adar, "Friends and Neighbors on the Web" (2003)
    """
    try:
        return sum(1 / np.log(graph.degree(neighbor)) for neighbor in nx.common_neighbors(graph, u, v) if graph.degree(neighbor) > 1)
    except:
        return 0

def preferential_attachment(graph, u, v):
    """
    Compute Preferential Attachment score for nodes u and v.
    Based on: Barabási and Albert, "Emergence of Scaling in Random Networks" (1999)
    """
    return graph.degree(u) * graph.degree(v)

def resource_allocation_index(graph, u, v):
    """
    Compute Resource Allocation Index score for nodes u and v.
    Based on: Zhou et al., "Predicting missing links via local information" (2009)
    """
    try:
        return sum(1 / graph.degree(neighbor) for neighbor in nx.common_neighbors(graph, u, v) if graph.degree(neighbor) > 0)
    except:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_query', type=str, default='./artifacts/ICLR/query_training.json')
    parser.add_argument('--path', type=str, default="./artifacts/ICLR/dataset_lpretriever_ICLR_artificial_query.pt")
    parser.add_argument('--heuristic', type=str, default='common_neighbors', 
                        choices=['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment', 'resource_allocation'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to decide if an edge should be added')
    args = parser.parse_args()
    
    with open(args.path_query, 'r', encoding='utf-8') as arquivo:
        questions = json.load(arquivo)

    all_q = []
    for q in questions.values():
        for que in q:
            all_q.append(que[0][0])
                
    data = torch.load(args.path, weights_only=False)
    data.edge_index = data.edge_index[:, -(2*len(all_q)):]

    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    # Create a NetworkX graph from the training edges
    train_edges = data.train_pos_edge_index.t().tolist()
    graph = nx.Graph()
    graph.add_edges_from(train_edges)
    num_nodes = data.num_nodes  # Total nodes in the graph
    graph.add_nodes_from(range(num_nodes))  # Ensure all nodes are included, even isolated ones

    # Prepare test edges: positive and negative
    test_pos_edges = data.test_pos_edge_index.t().tolist()
    test_neg_edges = data.test_neg_edge_index.t().tolist()
    
    # Map heuristic name to function
    heuristics = {
        'common_neighbors': common_neighbors,
        'jaccard': jaccard_coefficient,
        'adamic_adar': adamic_adar_index,
        'preferential_attachment': preferential_attachment,
        'resource_allocation': resource_allocation_index
    }
    
    heuristic_func = heuristics[args.heuristic]
    
    # Compute scores for positive test edges
    pos_scores = []
    for u, v in tqdm(test_pos_edges, desc=f'Computing {args.heuristic} for positive edges'):
        try:
            score = heuristic_func(graph, u, v)
        except:
            score = 0
        pos_scores.append(score)
    
    # Compute scores for negative test edges
    neg_scores = []
    for u, v in tqdm(test_neg_edges, desc=f'Computing {args.heuristic} for negative edges'):
        try:
            score = heuristic_func(graph, u, v)
        except:
            score = 0
        neg_scores.append(score)
    
    # Combine scores and labels
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    # Compute ROC AUC and AP
    roc_auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    print(f"{args.heuristic}: ROC_AUC = {roc_auc:.4f}, AP = {ap:.4f}")
    
    # Generate predicted edges for the entire graph (excluding existing edges)
    all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes) if not graph.has_edge(i, j)]
    predicted_edge_scores = []
    for u, v in tqdm(all_possible_edges, desc='Computing scores for all non-existing edges'):
        try:
            score = heuristic_func(graph, u, v)
        except:
            score = 0
        predicted_edge_scores.append((u, v, score))
    
    # Sort edges by score in descending order and apply threshold
    predicted_edge_scores.sort(key=lambda x: x[2], reverse=True)
    predicted_edges = [(u, v) for u, v, score in predicted_edge_scores if score >= args.threshold]
    
    # Add predicted edges to the graph
    graph_with_predicted_edges = graph.copy()
    graph_with_predicted_edges.add_edges_from(predicted_edges)
    
    # Return the graph with predicted edges (here, we print the number of added edges for demonstration)
    print(f"Added {len(predicted_edges)} predicted edges to the graph.")
    
    # For further use, you can return graph_with_predicted_edges
    # return graph_with_predicted_edges

if __name__ == '__main__':
    main()