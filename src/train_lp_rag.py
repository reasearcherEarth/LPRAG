import torch
from torch.optim import Adam
from tqdm import tqdm
import argparse
from src.gnn import DeepVGAE
from torch_geometric.utils import train_test_split_edges

parser = argparse.ArgumentParser()
parser.add_argument('--path_graph', type=str, default='./artifacts/ICLR/graph_dataset_iclr_with_synthetic_query.pt')
parser.add_argument('--destiny_model', type=str, default="./artifacts/ICLR/lpretriever_ICLR_model.pt")
args = parser.parse_args()

data = torch.load(args.path_graph, weights_only=False)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

device = 'cpu'
enc_in_channels = data.x.shape[1]
enc_hidden_channels = 32
enc_out_channels = 16

lr = 0.01
epochs = 400
model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data.x, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        model.eval()
        roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch+1, loss.cpu().item(), roc_auc, ap))


        import torch

torch.save(model.state_dict(), args.destiny_model)