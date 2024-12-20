from fastapi import FastAPI, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import torch
import pandas as pd
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from sklearn import preprocessing as pp
import uvicorn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df_all=pd.read_csv('customer_transactions_dataset (3).csv')
df=df_all.copy()
# n_users = df['userId'].nunique()
# n_items = df['Item_ID'].nunique()
# df=df.drop(['Transaction_ID','Payment_Method', 'Amount', 'Total_Purchases', 'Total_Amount_EGP',
#        'Customer_Name', 'Customer_Email', 'Country', 'Age', 'Gender',
#        'Product_Category', 'Product_Type', 'Product_Brand'],axis=1)
# df = df[df['Rating']>=3]
# # Perform a 80/20 train-test split on the interactions in the dataset
# train, test = train_test_split(df.values, test_size=0.2, random_state=16)
# train_df = pd.DataFrame(train, columns=df.columns)
# test_df = pd.DataFrame(test, columns=df.columns)

# le_user = pp.LabelEncoder()
# le_item = pp.LabelEncoder()
# train_df['user_id_idx'] = le_user.fit_transform(train_df['Customer_ID'].values)
# train_df['item_id_idx'] = le_item.fit_transform(train_df['Item_ID'].values)


# train_user_ids = train_df['Customer_ID'].unique()
# train_item_ids = train_df['Item_ID'].unique()

# #print(len(train_user_ids), len(train_item_ids))

# test_df = test_df[
#   (test_df['Customer_ID'].isin(train_user_ids)) & \
#   (test_df['Item_ID'].isin(train_item_ids))
# ]
# #print(len(test))
# n_users = train_df['user_id_idx'].nunique()
# n_items = train_df['item_id_idx'].nunique()

# test_df['user_id_idx'] = le_user.transform(test_df['Customer_ID'].values)
# test_df['item_id_idx'] = le_item.transform(test_df['Item_ID'].values)



# scaler = MinMaxScaler()

# # Apply Standard Scaling to the 'timestamp' column
# train_df['Date'] = scaler.fit_transform(train_df[['Date']])
# test_df['Date'] = scaler.transform(test_df[['Date']])



# def data_loader(data, batch_size, n_usr, n_itm):

#     def sample_neg(x):
#         while True:
#             neg_id = random.randint(0, n_itm - 1)
#             if neg_id not in x:
#                 return neg_id

#     interected_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
#     indices = [x for x in range(n_usr)]

#     if n_usr < batch_size:
#         users = [random.choice(indices) for _ in range(batch_size)]
#     else:
#         users = random.sample(indices, batch_size)
#     users.sort()
#     users_df = pd.DataFrame(users,columns = ['users'])

#     interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id_idx', right_on = 'users')
#     pos_items = interected_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values
#     neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

#     return (
#         torch.LongTensor(list(users)).to(device), 
#         torch.LongTensor(list(pos_items)).to(device) + n_usr, 
#         torch.LongTensor(list(neg_items)).to(device) + n_usr
#     )

# data_loader(train_df, 16, n_users, n_items)



# u_t = torch.LongTensor(train_df.user_id_idx)
# i_t = torch.LongTensor(train_df.item_id_idx) + n_users

# train_edge_index = torch.stack((
#   torch.cat([u_t, i_t]),
#   torch.cat([i_t, u_t])
# )).to(device)


latent_dim = 64
n_layers = 3 
EPOCHS = 50
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005 
K = 20




# Load the model

class LightGCNConv(MessagePassing):
  def __init__(self, **kwargs):  
    super().__init__(aggr='add')

  def forward(self, x, edge_index):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages (no update after aggregation)
    return self.propagate(edge_index, x=x, norm=norm)

  def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j


class RecSysGNN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items):
        super(RecSysGNN, self).__init__()
        # Initialize user and item embeddings
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)
        # LightGCN layers
        self.convs = nn.ModuleList(
            LightGCNConv() for _ in range(num_layers)
        )
        self.init_parameters()

    def init_parameters(self):
        # Normal initialization for LightGCN
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight  # Initial embeddings
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        # Aggregate embeddings by averaging
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items],
        )

# Create FastAPI app
templates = Jinja2Templates(directory="templates")

app = FastAPI()
n_users=652
n_items=7747
# Load model and parameters
model_path = "lightgcn_model (2).pth"
model = RecSysGNN(latent_dim=64, num_layers=3, num_users=n_users, num_items=n_items)
model.load_state_dict(torch.load(model_path ,map_location=torch.device('cpu'), weights_only=True))
model.eval()

def preprocess_data(df, le_user=None, le_item=None, scaler=None):
    if le_user is not None:
        df['user_id_idx'] = le_user.fit(df['Customer_ID'].values)
        df['user_id_idx'] = le_user.transform(df['Customer_ID'].values)
    if le_item is not None:
        df['item_id_idx'] = le_item.fit(df['Item_ID'].values)
        df['item_id_idx'] = le_item.transform(df['Item_ID'].values)
    if scaler is not None:
        df['Date'] = scaler.fit(df['Date'].values)
        df['Date'] = scaler.transform(df['Date'].values)

    return df

user_label_encoder = pp.LabelEncoder()  
item_label_encoder = pp.LabelEncoder()  
date_scaler = None  

# Example usage:
preprocessed_df = preprocess_data(df, user_label_encoder, item_label_encoder, date_scaler)

u_t = torch.LongTensor(preprocessed_df.user_id_idx)
i_t = torch.LongTensor(preprocessed_df.item_id_idx) + n_users #differentiate between user and item IDs within the graph structure.

edge_index = torch.stack((
  torch.cat([u_t, i_t]),# represents the source nodes (users) for each edge in the graph.
  torch.cat([i_t, u_t])#represents the target nodes (items) for each edge in the graph.
)).to(device)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "items": []})

item_id_mapping = dict(zip(preprocessed_df['item_id_idx'], preprocessed_df['Item_ID']))

@app.post("/recommendations/", response_class=HTMLResponse)
async def recommend(request: Request, user_id: int = Form(...), top_k: int = Form(10)):
    if user_id < 0 or user_id >= n_users:
        raise HTTPException(status_code=404, detail="User not found")

    with torch.no_grad():
        _, out = model(edge_index)
        user_emb, item_emb = torch.split(out, (n_users, n_items))

    user_embedding = user_emb[user_id]
    scores = torch.matmul(user_embedding, item_emb.T)
    top_k_indices = torch.topk(scores, k=top_k).indices.tolist()

    # Map indices to item IDs and fetch product details
    recommended_items = []
    for idx in top_k_indices:
        item_id = item_id_mapping[idx]
        product_info = preprocessed_df[preprocessed_df['Item_ID'] == item_id][
            ['Product_Category', 'Product_Type', 'Product_Brand']
        ].iloc[0].to_dict()
        product_info['Item_ID'] = item_id
        recommended_items.append(product_info)

    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "user_id": user_id,
        "recommended_items": recommended_items
    })

# Run the application with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)