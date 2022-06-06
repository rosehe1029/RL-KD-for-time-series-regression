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
import time

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
    epochs = 40
    batch_size = 64
    lrate = 1e-3
    num_of_teachers = 20

    model_path= 'teacher_models/independent/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    data_path = 'processed_data/cmapss_train_valid_test_dic.pt'
    my_dataset = torch.load(data_path)

    # Create Training Dataset
    train_x = torch.from_numpy(my_dataset[data_identifier]['train_data']).float()
    train_y = torch.from_numpy(my_dataset[data_identifier]['train_labels']).float()
    test_x = torch.from_numpy(my_dataset[data_identifier]['test_data']).float()
    test_y = torch.FloatTensor(my_dataset[data_identifier]['test_labels'])
    val_x = torch.from_numpy(my_dataset[data_identifier]['valid_data']).float()
    val_y = torch.from_numpy(my_dataset[data_identifier]['valid_labels']).float()
    train_ds = MyDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=0,worker_init_fn=_init_fn)
    test_rmse = []
    test_score = []
    print("train.shape = {}, val_shape={}, test_Shape={}".format(train_x.shape, val_x.shape, test_x.shape))

    for ind_of_teacher in range(1, num_of_teachers+1):

        model = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)
        model_name = model_path + data_identifier + '_teacher_'+str(ind_of_teacher)+'.pt'

        if train_enable:
            print("Start training dataset:", data_identifier, 'Teacher Index:', ind_of_teacher)
            optimizer =optim.AdamW(model.parameters(), lr=lrate)
            scheduler = StepLR(optimizer,gamma=0.9,step_size=1)

            best_val_loss = 1e+9
            for epoch in range(1, epochs+1):
                epoch_loss = train(model,device,train_loader,optimizer)
                scheduler.step()

                if epoch % 5 == 0:
                    print('Teacher Index {} Epoch{} Training Loss= {:.9f}'.format(ind_of_teacher, epoch, epoch_loss))
                    with torch.no_grad():
                        val_loss = validate(model, device, val_x, val_y)
                        if val_loss < best_val_loss:
                            print("val_loss improved from {:.9f} to {:.9f}, save the model".format(best_val_loss,val_loss))
                            best_val_loss = val_loss
                            model_best = copy.deepcopy(model.state_dict())

            # Evaluate on test data set
            model.load_state_dict(model_best)
            rmse,score = test(model,device,test_x,test_y,max_RUL)
            print("{} | it-{} | Test RMSE = {:.2f}, Test Score ={:.2f}".format(data_identifier, ind_of_teacher, rmse,score))
            test_rmse.append(rmse.item())
            test_score.append(score)
            # save trained model
            torch.save(model.state_dict(),model_name)
            print()
        else:
            model.load_state_dict(torch.load(model_name))
            rmse,score = test(model,device,test_x,test_y,max_RUL)
            print("Test RMSE = {}, Test Score ={}".format(rmse,score))

    test_rmse = [round(elem,2) for elem in test_rmse]
    test_score = [round(elem, 2) for elem in test_score]
    print("{} | Test RMSE={}".format(data_identifier, test_rmse))
    print("{} | Test Score={}".format(data_identifier, test_score))

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