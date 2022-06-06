import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models import LSTM_RUL, RMSELoss, Score, MyDataset
from torch.optim.lr_scheduler import StepLR
import argparse
import copy
from sgdr import CosineAnnealingLR_with_Restart
import glob
import time
import pynvml

def _init_fn(worker_id):
    np.random.seed(int(1))
def set_random_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# set_random_seed(4)

def train(model,device,train_loader,optimizer):
    model.train()
    Loss = 0.0
    critirion = nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):

        batch_x, batch_y, _ = data
        batch_x,batch_y= batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        # Forward
        pred, _ = model(batch_x)
        # Calculate loss and update weights
        loss = critirion(pred,batch_y)
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    return Loss/len(train_loader)

def test(model,device,x,y,max_RUL):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _  = model(x_cuda)
        pred = pred *max_RUL
        rmse = RMSELoss(pred,y_cuda)
        score = Score(pred,y_cuda)
    return rmse, score

def validate(model,device,x,y):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _ = model(x_cuda)
        loss = RMSELoss(pred,y_cuda)
    return loss

def main(args):
    train_enable = (args.train==0)
    dataset_index = args.dataset
    data_identifiers = ['FD001', 'FD002', 'FD003', 'FD004']
    data_identifier = data_identifiers[dataset_index - 1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_RUL = 130
    epochs = 240
    batch_size = 64
    lrate = 1e-3
    num_of_teachers = 20

    t_max = 10
    t_mult = 1

    lr_min = 1e-5

    model_path = 'teacher_models/'
    snapshot_checkpoint_path= model_path + 'snapshot/'+data_identifier+'/'
    if not os.path.exists(snapshot_checkpoint_path):
        os.makedirs(snapshot_checkpoint_path)

    data_path = 'processed_data/cmapss_train_valid_test_dic.pt'
    my_dataset = torch.load(data_path)

    # Create Training Dataset
    train_x = torch.from_numpy(my_dataset[data_identifier]['train_data']).float()[:128]
    train_y = torch.from_numpy(my_dataset[data_identifier]['train_labels']).float()[:128]
    test_x = torch.from_numpy(my_dataset[data_identifier]['test_data']).float()[:128]
    test_y = torch.FloatTensor(my_dataset[data_identifier]['test_labels'])[:128]
    val_x = torch.from_numpy(my_dataset[data_identifier]['valid_data']).float()[:128]
    val_y = torch.from_numpy(my_dataset[data_identifier]['valid_labels']).float()[:128]
    train_ds = MyDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=0,worker_init_fn=_init_fn)

    print("train.shape = {}, val_shape={}, test_Shape={}".format(train_x.shape, val_x.shape, test_x.shape))
    model = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)

    if train_enable:
        print("Start Snapshot Ensembles training dataset:", data_identifier)

        optimizer =optim.AdamW(model.parameters(), lr=lrate)

        mySGDR = CosineAnnealingLR_with_Restart(optimizer=optimizer, T_max=t_max, T_mult=t_mult, model=model,
                                                out_dir= snapshot_checkpoint_path, data_identifier=data_identifier,
                                                take_snapshot=True, eta_min=lr_min)
        if True:
            for epoch in range(1, epochs+1):
                mySGDR.step()
                epoch_loss = train(model,device,train_loader,optimizer)

                print('Epoch{} | Training Loss= {:.6f}'.format(epoch, epoch_loss))

                if epoch % 10 == 0:
                    with torch.no_grad():
                        val_loss = validate(model, device, val_x, val_y)
                        print('Epoch{} | Training Loss= {:.6f}, Validation Loss={:.6f}'.format(epoch, epoch_loss, val_loss))
        # Evaluation on Test Datasets
        checkpoints = sorted(glob.glob(snapshot_checkpoint_path+'/*.tar'))
        models =[]
        for path in checkpoints:
            model = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)
            ch = torch.load(path)
            model.load_state_dict(ch['state_dict'])
            models.append(model)

        rmse_list = []
        score_list = []
        num_net = 1

        for model in models[(len(models)-num_of_teachers):]:
            rmse, score = test(model, device, test_x, test_y, max_RUL)
            rmse_list.append(rmse.item())
            score_list.append(score)

            model_name = snapshot_checkpoint_path + data_identifier + '_teacher_snapshot_' + str(num_net) + '.pt'
            # save trained model
            torch.save(model.state_dict(), model_name)
            num_net+=1

        test_rmse = [round(elem, 2) for elem in rmse_list]
        test_score = [round(elem, 2) for elem in score_list]
        print("{} | Test RMSE={}".format(data_identifier, test_rmse))
        print("{} | Test Score={}".format(data_identifier, test_score))

    else:
        print("To be Implemented...")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',
                        help='0:Model Train, 1: Model Inference',
                        type=int,
                        choices=[0, 1],
                        default=0)
    parser.add_argument('-d', '--dataset',
                        help='1:FD001, 2: FD002, 3:FD003, 4:FD004',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=1)
    args = parser.parse_args()
    main(args)