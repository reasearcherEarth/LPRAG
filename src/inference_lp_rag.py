import json
from src.utils import get_node_embedding
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
encoder = AutoModel.from_pretrained('facebook/contriever')
import torch
from torch_geometric.utils import train_test_split_edges
from src.gnn import DeepVGAE
from src.llms import GroqLLM
from tqdm import tqdm 
import time 
from src.evaluate import Evaluator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--extracted_chunks', type=str, default='./artifacts/ICLR/extracted_chunks_test.json')
parser.add_argument('--query_and_answer', type=str, default="./artifacts/ICLR/query_+_answers.json")
parser.add_argument('--ICLR_chunks', type=str, default="./artifacts/ICLR/ICLR_chunks.pt")
parser.add_argument('--gcn_model', type=str, default="./artifacts/ICLR/lpretriever_ICLR_model.pt")

args = parser.parse_args()

with open(args.extracted_chunks, 'r', encoding='utf-8') as arquivo:
    chunks = json.load(arquivo)

all_chunks = []
for chun in chunks:
    all_chunks.extend(chunks[chun])
import numpy as np
all_chunks = np.array(all_chunks)

with open(args.query_and_answer, 'r', encoding='utf-8') as arquivo:
    queries = json.load(arquivo)

data = torch.load(args.ICLR_chunks, weights_only=False)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
print(data)

device = 'cpu'
enc_in_channels = data.x.shape[1]
enc_hidden_channels = 32
enc_out_channels = 16
lr = 0.01
epochs = 400
model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
model.load_state_dict(torch.load(args.gcn_model, map_location="cpu"))
model.eval()

llm = GroqLLM()

results = []

for query in tqdm(queries.values()):
    for qs in query:
        ini = time.time()
        Q = qs[0][0].replace('[','').replace(']','')
        A = qs[-1][0].replace('[','').replace(']','')

        emb_Q = get_node_embedding(Q, tokenizer, encoder)[0][0]
        new_x = torch.cat((data.x, emb_Q.unsqueeze(0)), dim=0)

        edge_1 = data.train_pos_edge_index.flatten().unique().view(-1,1)
        idx_do_no_query_real = edge_1.max()+1 # index of the new node in the actual query we are going to evaluate

        edge_2 = (torch.zeros(len(edge_1)).long() + idx_do_no_query_real).view(-1,1)
        new_edge_index = torch.cat((edge_1, edge_2), dim=1).T # Each chunk node is connected to the new actual query node.

        print("emb+join ", time.time()-ini)

        new_train_pos_edge_index = torch.cat((data.train_pos_edge_index, new_edge_index), dim=1)
        ini = time.time()
        
        model.eval()
        pred = model.predict(new_x, new_train_pos_edge_index, new_edge_index)
        pred = pred.sigmoid()

        id_selected_chunks = pred > 0.8 # For the threshold to decide which chunks are chosen, I selected 0.5 (though it could be higher).
        selected_chunks = new_edge_index[:, id_selected_chunks][0] # index of selected chunks
        text_selected_chunks = list(all_chunks[selected_chunks.numpy()]) # text of the selected chunks
        print("GNN ", time.time()-ini)

        ini = time.time()
        text_selected_chunks = '\n'.join(text_selected_chunks)
        prompt = f"""
            [Query]
            {Q}

            [Bag of chunks]
            {text_selected_chunks}"""

        
        response, tot_tok = llm.generate(system_prompt="Answer the provided Query using only the provided Bag of chunks", user_prompt=prompt)

        print("LLm ", time.time()-ini)

        result = {
            "question": Q,
            "gold_answer": A,
            "sorted_passage": list(all_chunks[selected_chunks.numpy()]),
            "pred_answer": response,
            "total_tokens": tot_tok
        }

        results.append(result)

        with open("./artifacts/ICLR/result_lp_rag.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
   
        break
    break

evaluator = Evaluator("./artifacts/ICLR/result_lp_rag.json")
evaluator.evaluate(max_workers=10)