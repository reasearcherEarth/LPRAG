import torch
from torch_geometric.utils import remove_self_loops
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graph_dataset_with_synthetic_query', type=str, default='./artifacts/ICLR/graph_dataset_iclr_with_synthetic_query.pt')
parser.add_argument('--dataset_link_prediction', type=str, default="./artifacts/ICLR/ICLR_link_prediction.pt")

args = parser.parse_args()

def main():
    data = torch.load(args.graph_dataset_with_synthetic_query, weights_only=False)
    print("Building graph to link prediction...")
    edge1 = data.edge_index[0]
    edge2 = data.edge_index[1]

    neg_edges = []
    size = len(edge1) // 4
    for edge in edge2:
        idxs = torch.randint(0, len(edge1), (size,))
        sub_edge1 = edge1[idxs].view(-1,1).long()
        sub_edge2 = (torch.zeros(len(sub_edge1)) + edge).view(-1,1).long()
        new_neg_edge_index = torch.cat((sub_edge1, sub_edge2), dim=1)
        neg_edges.append(new_neg_edge_index)
        
    neg_edges = torch.cat(neg_edges, dim=0).T    
        
    data.train_neg_edge_index = neg_edges
    data.train_pos_edge_index = data.edge_index

    data.edge_index = data.edge_attr = None

    data.train_neg_edge_index, _ = remove_self_loops(data.train_neg_edge_index)

    filtrer_train_neg = torch.unique(data.train_neg_edge_index.T, dim=0)
    data.train_neg_edge_index = filtrer_train_neg.T

    torch.save(data, args.dataset_link_prediction)
    print("Graph Created!!")
    print(data)

if __name__ == '__main__':
    main()