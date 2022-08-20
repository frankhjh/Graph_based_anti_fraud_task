import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import joblib
import argparse

import sys

# append the path into your env, you may adjust based on your own system
sys.path.append("..")
print(sys.path)
from feature.data_preprocess import graph_data_loader,graph_info_extractor,graph2table
from feature.feature_engineer import feature_building,feature_comb
from utils.dataset import User_info
from utils.evaluator import Evaluator
from utils.normalize import train_test_x_normalization

from model.mlp import MLP


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(model,metric,valid_data,device,evaluator):
    model.eval()

    x,y = valid_data[0],valid_data[1]
    x,y = x.to(device),y.to(device)
    
    with torch.no_grad():
        out = model(x)
        loss = metric(out,y)

        auc = evaluator.eval(y,out.exp())
    return loss,auc


def train(model,train_dataloader,valid_data,epochs,lr,device,evaluator):
    
    model = model.to(device)
    model.train()
    
    optimizer = Adam(model.parameters(),lr = lr)
    metric = nn.NLLLoss()

    min_loss,best_epoch = 1e5,1

    for epoch in range(1,epochs + 1):
        tot_loss = 0.0
        for step,(x,y) in enumerate(train_dataloader):
            x,y = x.to(device),y.to(device)
            out = model(x)

            loss = metric(out,y)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            tot_loss += loss.item()
        avg_loss = tot_loss / (step + 1)

        # val
        val_loss,val_auc = test(model,metric,valid_data,device,evaluator)

        print('at epoch {}, training loss:{} validation loss:{} validation auc:{}'.format(epoch,avg_loss,val_loss,val_auc))

        if val_loss < min_loss:
            min_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),'../model_files/opt_model.ckpt')



def predict(model,param_path,test_x,device):
    model.load_state_dict(torch.load(param_path))

    model.eval()

    test_x = test_x.to(device)

    predicts = model(test_x)

    output = predicts.exp()
    output_arr = output.detach().numpy() if device == 'cpu' else output.detach().cpu().numpy()

    np.save('../submit/output.npy',output_arr)



def main():
    
    parser = argparse.ArgumentParser(description='train_params')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--valid_ratio',type = float,default=0.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr',type = float,default=1e-3)
    parser.add_argument('--seed',type = int,default=100)

    args = parser.parse_args()


    # load graph data
    gdata_path = '../../xydata/gdata.npz'
    gdata = graph_data_loader(gdata_path)

    # get different parts
    x = gdata['x']
    y = gdata['y']
    edge_index = gdata['edge_index']
    edge_type = gdata['edge_type']
    edge_timestamp = gdata['edge_timestamp']
    train_mask = gdata['train_mask']
    test_mask = gdata['test_mask']

    # extract info from graph data
    start_info,end_info = graph_info_extractor(x.shape[0],edge_index,edge_type,edge_timestamp)

    # transfer info into structured data
    df = graph2table(x.shape[0],start_info,end_info)

    # feature engineering
    user_feats = feature_building(df,edge_type)

    # concat new features with raw embeddings
    user_feats_all = feature_comb(user_feats,x)

    # split train/validation/test set
    validation_ratio = args.valid_ratio

    valid_start = int(len(train_mask) * 0.01)
    valid_end = int(len(train_mask) * 0.21)
    train_mask_new = np.concatenate([train_mask[:valid_start],train_mask[valid_end:]],axis=0)
    valid_mask = train_mask[valid_start:valid_end]

    train_x = user_feats_all.loc[train_mask_new,:].reset_index(drop=True)
    train_y = y[train_mask_new]

    valid_x = user_feats_all.loc[valid_mask,:].reset_index(drop=True)
    valid_y = y[valid_mask]

    test_x = user_feats_all.loc[test_mask,:].reset_index(drop=True)
    #test_y = y[test_mask]

    # normalization
    scaler,normalized_train_x,normalized_valid_x,normalized_test_x = train_test_x_normalization(train_x,valid_x,test_x)

    joblib.dump(scaler,'../model_files/scaler.pkl')

    # into torch.tensor
    train_x_tensor = torch.Tensor(normalized_train_x)
    valid_x_tensor = torch.Tensor(normalized_valid_x)
    test_x_tensor = torch.Tensor(normalized_test_x)

    train_y_tensor = torch.LongTensor(train_y)
    valid_y_tensor = torch.LongTensor(valid_y)
    #test_y_tensor = torch.Tensor(test_y)

    train_data = User_info(train_x_tensor,train_y_tensor)
    train_dataloader = DataLoader(train_data,batch_size = 512,shuffle = True)
    
    valid_data = (valid_x_tensor,valid_y_tensor)
    # valid_data = User_info(valid_x_tensor,valid_y_tensor)
    # valid_dataloader = DataLoader(valid_data,batch_size = 512,shuffle = False)

    # model structure parameters
    input_dim = user_feats_all.shape[1]
    hidden_dim = 128
    output_dim = 2
    num_layers = 4
    dropout = 0.3

    fix_seed(args.seed)

    model = MLP(input_dim,hidden_dim,output_dim,num_layers,dropout)

    if args.device != 'cpu' and torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'
    epochs = args.epochs
    lr = args.lr

    # evaluator = Evaluator('auc')
    # train(model,train_dataloader,valid_data,epochs,lr,device,evaluator)

    param_path = '../model_files/opt_model.ckpt'
    predict(model,param_path,test_x_tensor,device)



if __name__ == '__main__':
    main()

























