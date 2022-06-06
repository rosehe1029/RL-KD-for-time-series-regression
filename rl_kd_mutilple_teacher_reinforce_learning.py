import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from models import LSTM_RUL, RMSELoss, Score, centered_average,weights_init
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import argparse
import copy
import logging
from models import MyDataset, CNN_RUL_student_stack, Dueling_DQN
from replay_memory import ReplayMemory, Memory
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

def beta_decay(epoch,epochs=80,alpha=1.0,d_rate=0.1):
    return alpha - alpha * np.exp(-d_rate*(epochs - epoch))

def test(model,device,x,y,max_RUL):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _, _ = model(x_cuda.permute(0,2,1))
        pred = pred *max_RUL
        rmse = RMSELoss(pred,y_cuda)
        score = Score(pred,y_cuda)
        return rmse, score
    pass


def validate(model,device,x,y):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _, _= model(x_cuda.permute(0,2,1))
        loss = RMSELoss(pred,y_cuda)
    return loss


def validate_rmse_score(model,x,y):
    with torch.no_grad():
        pred, _, _= model(x.permute(0,2,1))
        rmse = RMSELoss(pred,y)
        score = Score(pred*130, y*130)
    return rmse, score

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    # Target_net have already been normalized in avg_inference function
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss


def ensembl_inference(model_ensembls, input_x):
    pred_list = []
    feat_list = []
    for model in model_ensembls:
        with torch.no_grad():
            pred_t, feat_t = model(input_x)
            pred_t = Variable(pred_t, requires_grad=False)
            feat_t = Variable(feat_t, requires_grad=False)
            pred_list.append(pred_t)
            feat_list.append(feat_t)
    return feat_list, pred_list


def avg_over_selected_teacher(feat_list, pred_list, teacher_mask_off,norm=True):
    feat_list_select = []
    pred_list_select = []

    eps = 0.0000001
    for i in range(len(feat_list)):
        if teacher_mask_off:
            if norm:
                target_net = feat_list[i]
                target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
                target_net = target_net / (target_net_norm + eps)
                target_net[target_net != target_net] = 0
                feat_list_select.append(target_net)
            else:
                feat_list_select.append(feat_list[i])
            pred_list_select.append(pred_list[i])
    # Get Average value of Teachers Prediction
    pred_mean = torch.mean(torch.stack(pred_list_select), dim=0)
    feat_mean = torch.mean(torch.stack(feat_list_select), dim=0)

    return pred_mean, feat_mean


def main(opt):

    dataset_index = opt.dataset
    data_identifiers = ['FD001', 'FD002', 'FD003', 'FD004']
    data_identifier = data_identifiers[dataset_index - 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t_type = opt.teacher

    max_RUL= 130
    epochs = 40
    batch_size = 64
    iterations = 5
    lrate = 1e-3
    episodes = 3 # For Reinforcement Steps
    num_of_teachers = 20

    model_s_path = 'student_models/'

    if not os.path.exists(model_s_path):
        os.makedirs(model_s_path)

    data_path = 'processed_data/cmapss_train_valid_test_dic.pt'
    my_dataset = torch.load(data_path)

    log_dir = 'log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if t_type == "ind":
        logger_name = log_dir + data_identifier + '_multiple_teacher_reinforced_kd_ind.log'
    elif t_type == 'ncl':
        logger_name = log_dir + data_identifier + '_multiple_teacher_reinforced_kd_ncl.log'
    elif t_type == 'snapshot':
        logger_name = log_dir + data_identifier + '_multiple_teacher_reinforced_kd_snapshot.log'
    logging.basicConfig(filename=logger_name, level=logging.INFO)

    print('Start Reinforcement Learning Multi-Teacher - Student KD training: ',data_identifier)
    logging.info('Start Reinforcement Learning Multi-Teacher - Student KD training:{} '.format(data_identifier))

    # Create Training Dataset
    train_x = torch.from_numpy(my_dataset[data_identifier]['train_data']).float()
    train_y = torch.from_numpy(my_dataset[data_identifier]['train_labels']).float()
    test_x = torch.from_numpy(my_dataset[data_identifier]['test_data']).float()
    test_y = torch.FloatTensor(my_dataset[data_identifier]['test_labels'])
    val_x = torch.from_numpy(my_dataset[data_identifier]['valid_data']).float()
    val_y = torch.from_numpy(my_dataset[data_identifier]['valid_labels']).float()
    train_ds = MyDataset(train_x,train_y)
    val_ds = MyDataset(val_x,val_y)
    train_loader = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,
                              num_workers=0, worker_init_fn=_init_fn, drop_last=True)
    val_loader = DataLoader(dataset=val_ds,batch_size=128,shuffle=True,
                              num_workers=0, worker_init_fn=_init_fn, drop_last=True)

    print("train_x.size=", train_x.size())
    print("val_x.size=", val_x.size())

    model_t_ensemble =[]
    for t_count in range(1,num_of_teachers+1):
        model_t = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)
        if t_type == 'ind':
            model_t_name = 'teacher_models/independent/' + data_identifier + '_teacher_'+str(t_count)+'.pt'
        elif t_type == 'ncl':
            model_t_name = 'teacher_models/ncl/' + data_identifier + '_teacher_ncl_' + str(t_count) + '.pt'
        elif t_type == 'snapshot':
            model_t_name = 'teacher_models/snapshot/' + data_identifier + '/' + data_identifier + '_teacher_snapshot_' + str(t_count) + '.pt'
        model_t.load_state_dict(torch.load(model_t_name))
        print("load teacher:", model_t_name)
        model_t_ensemble.append(model_t)

    for model_t in model_t_ensemble:
        model_t.eval()
    test_rmse = []
    test_score = []

    for iteration in range(iterations):
        # ===============================Initialize Student =======================================
        model_s = CNN_RUL_student_stack(input_dim=14, hidden_dim=32, dropout=0.5).to(device)
        model_s_backup = CNN_RUL_student_stack(input_dim=14, hidden_dim=32, dropout=0.5).to(device)

        model_s.apply(weights_init)
        model_s_backup.load_state_dict(model_s.state_dict())
        model_s_backup.eval()

        optimizer_s = optim.AdamW(model_s.parameters(), lr=lrate)
        scheduler = StepLR(optimizer_s, gamma=0.9, step_size=10)

        # ===============================Initialize Dueling Double DQN and Replay Memory ===========
        DQN = Dueling_DQN(in_channels=num_of_teachers, num_actions=num_of_teachers).to(device) # in_channels = num_actions = Teacher numbers
        DQN_target = Dueling_DQN(in_channels=num_of_teachers, num_actions=num_of_teachers).to(device)
        DQN_target.load_state_dict(DQN.state_dict())

        memory = Memory(5000)
        optimizer_dqn =  optim.Adam(DQN.parameters(), lr=0.0001)

        global_step = 0
        best_val_loss = 1e+9

        sum_teacher_mask_off = [0] * num_of_teachers

        for epoch in range(1, epochs+1):

            print(">> {} | {} | Iteration-{} | training epoch - {}".format(t_type, data_identifier, str(iteration),str(epoch)))
            model_s.train()

            for batch_idx, data in enumerate(train_loader):

                batch_x, batch_y, _ = data
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # sample a golden batch for validation
                val_x_batch, val_y_batch, _ = iter(val_loader).next()
                val_x_batch, val_y_batch = val_x_batch.to(device), val_y_batch.to(device)

                # inference with all teacher ensembles
                feat_t_list, soft_labels_list = ensembl_inference(model_t_ensemble, batch_x)

                teacher_mask_off=[0]*num_of_teachers    # 1-seleced, 0-not selected

                t_rmse_list = []
                t_score_list = []

                for ii in range(len(feat_t_list)):

                    rmse = RMSELoss(soft_labels_list[ii].clone().detach(), batch_y)
                    t_rmse_list.append(rmse.item())

                    score = Score(soft_labels_list[ii].clone().detach() * max_RUL, batch_y * max_RUL)
                    t_score_list.append(score)

                for episode in range(episodes):

                    optimizer_s.zero_grad()
                    # Feed forward the network and update
                    pred_s, _, feat_s = model_s(batch_x.permute(0, 2, 1))

                    # Get the previous rmse and score before optimizaiton on validation set
                    if episode == 0:
                        model_s_backup.load_state_dict(model_s.state_dict())
                        pre_rmse, pre_score = validate_rmse_score(model_s_backup, val_x_batch, val_y_batch)
                    else: # save training time
                        pre_rmse = post_rmse
                        pre_score = post_score

                    if episode ==0:
                         state = torch.stack(feat_t_list).clone().detach().unsqueeze(dim=0)  # size(1, 10, 64, 64) = (1, T_num, sample_batch_size, feat_size)
                         next_state = state.clone().detach()
                    else:
                         state = next_state.clone().detach()

                    # Get action for current state;
                    action, output, _ = DQN.sample_action(state)

                    # set 1 to selected teacher in teacher_mask_off_list
                    if teacher_mask_off[action[0]] == 1:
                        output = F.softmax(output,dim=1)
                        output = output.masked_fill(torch.ByteTensor(teacher_mask_off).to(device).bool(),0)
                        action = (torch.argmax(output, dim=1)).detach().cpu().numpy()
                        teacher_mask_off[action[0]] = 1
                    else:
                        teacher_mask_off[action[0]] = 1

                    # Set next_state according to current action
                    next_state[:,action[0],:,:] = 0

                    # Get done mask for later Q_target calculation
                    if episode == episodes-1:
                        done = 0.0
                    else:
                        done = 1.0

                    # Calculate loss and update students weights with selected teachers
                    pred_t, feat_t = avg_over_selected_teacher(feat_t_list,soft_labels_list,teacher_mask_off, norm=False)
                    pkt_loss = cosine_similarity_loss(feat_s, feat_t)  # PKT_loss
                    soft_loss = nn.MSELoss()(pred_s, pred_t)
                    hard_loss = nn.MSELoss()(pred_s, batch_y)
                    beta = beta_decay(epoch, epochs)
                    alpha = 1 - beta
                    alpha_1 = 0.8 * alpha
                    alpha_2 = alpha - alpha_1
                    loss = alpha_1 * hard_loss + alpha_2 * soft_loss + beta * pkt_loss  # Early stage: PKT; Late stage: hardloss

                    loss.backward()
                    optimizer_s.step()
                    global_step+=1

                    # Get the post rmse and score after optimizaiton on validation set with selected teachers
                    model_s_backup.load_state_dict(model_s.state_dict())
                    post_rmse, post_score = validate_rmse_score(model_s_backup, val_x_batch, val_y_batch)

                    # Calculate the reward for the selected teacher
                    relative_rmse_improve = (pre_rmse - post_rmse) / pre_rmse
                    reward_rmse = torch.clamp(relative_rmse_improve, min=-1, max=1)
                    reward = reward_rmse.item()
                    reward = np.tanh(4 * reward)

                    if episode == episodes - 1:

                        memory.add((next_state.squeeze().tolist(), action, np.expand_dims(reward, 0),
                                    next_state.squeeze().tolist(), np.expand_dims(done, 0)))
                    else:
                        memory.add((state.squeeze().tolist(), action, np.expand_dims(reward, 0),
                                    next_state.squeeze().tolist(), np.expand_dims(done, 0)))


                    # update DQN amd DQN_target
                    if memory.size() > 50:
                        batch = memory.sample(32, False)
                        s, a, r, s_prime, done_mask = zip(*batch)
                        # double dqn
                        Q_out = DQN(torch.Tensor(s).to(device))
                        Q_est = Q_out.gather(1, ((torch.Tensor(a)).type(torch.LongTensor)).to(device))

                        a_max = DQN(torch.Tensor(s_prime).to(device)).detach().argmax(1).unsqueeze(1)
                        max_q_prime = DQN_target(torch.Tensor(s_prime).to(device)).gather(1, a_max)
                        Q_target = torch.Tensor(r).to(device) + 0.9 * max_q_prime * torch.Tensor(done_mask).to(device)

                        loss_dqn = F.smooth_l1_loss(Q_est, Q_target)

                        optimizer_dqn.zero_grad()
                        loss_dqn.backward()
                        optimizer_dqn.step()

                        if global_step % 100 == 0:
                            DQN_loss = loss_dqn.item() # for display only

                        for target_param, param in zip(DQN_target.parameters(), DQN.parameters()):
                            target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)

                if memory.size() > 50:
                    sum_teacher_mask_off = [sum(x) for x in zip(sum_teacher_mask_off, teacher_mask_off)]

                if global_step % 100 == 0:
                    model_s_backup.load_state_dict(model_s.state_dict())
                    with torch.no_grad():
                        val_loss = validate(model_s_backup, device, val_x, val_y)
                        print("Global Step - {} | val_loss = {:.6f}, DQN_loss = {:.6f}".format(global_step, val_loss, DQN_loss))

                        if val_loss < best_val_loss:
                            print("Epoch {} | val_loss improved from {:.6f} to {:.6f}".format(epoch, best_val_loss,
                                                                                                   val_loss))
                            logging.info("Epoch {} | Val_loss improved from {:.6f} to {:.6f}".format(epoch, best_val_loss,
                                                                                                   val_loss))
                            best_val_loss = val_loss
                            model_s_best = copy.deepcopy(model_s.state_dict())

                            if t_type=='ind':
                                model_s_name = model_s_path + data_identifier + '_multiple_teachers_ind_rl_kd_student_' + str(
                                    iteration) + '.pt'
                            elif t_type == 'ncl':
                                model_s_name = model_s_path + data_identifier + '_multiple_teachers_ncl_rl_kd_student_' + str(
                                    iteration) + '.pt'
                            elif t_type == 'snapshot':
                                model_s_name = model_s_path + data_identifier + '_multiple_teachers_snap_rl_kd_student_' + str(
                                    iteration) + '.pt'
                            torch.save(model_s.state_dict(), model_s_name)
            scheduler.step() # Update the learning rate every epoch
            print("total global step =", global_step)

        # Evaluate on test data set
        model_s.load_state_dict(model_s_best)
        rmse, score = test(model_s, device, test_x, test_y, max_RUL)
        print("{} | Itereation-{} | Test RMSE = {}, Test Score ={}".format(data_identifier, iteration, rmse, score))
        print()
        logging.info("{} | Itereation-{} | sum_teacher_mask_off={}".format(data_identifier, iteration, sum_teacher_mask_off))
        # save trained model
        if t_type == 'ind':
            model_s_name = model_s_path + data_identifier + '_multiple_teachers_ind_rl_kd_student_' + str(
                iteration) + '.pt'
        elif t_type == 'ncl':
            model_s_name = model_s_path + data_identifier + '_multiple_teachers_ncl_rl_kd_student_' + str(
                iteration) + '.pt'
        elif t_type == 'snapshot':
            model_s_name = model_s_path + data_identifier + '_multiple_teachers_snap_rl_kd_student_' + str(
                iteration) + '.pt'
        torch.save(model_s.state_dict(), model_s_name)
        test_rmse.append(rmse.item())
        test_score.append(score)
        del model_s
        torch.cuda.empty_cache()

    # round down
    test_rmse = [round(elem, 2) for elem in test_rmse]
    test_score = [round(elem, 2) for elem in test_score]
    print("{} | {} | Test RMSE={}".format(t_type, data_identifier, test_rmse))
    print("{} | {} | Test Score={}".format(t_type, data_identifier, test_score))
    logging.info("{} | {} | Test RMSE={}".format(t_type, data_identifier, test_rmse))
    logging.info("{} | {} | Test Score={}".format(t_type, data_identifier, test_score))
    mean_rmse = centered_average(test_rmse)
    mean_score = centered_average(test_score)
    print("{} | {} | Average RMSE = {:.2f}".format(t_type, data_identifier, mean_rmse))
    print("{} | {} | Average Score = {:.2f}".format(t_type, data_identifier, mean_score))
    logging.info("{} | {} | Average RMSE = {:.2f} ".format(t_type, data_identifier, mean_rmse))
    logging.info("{} | {} | Average Score = {:.2f} ".format(t_type, data_identifier, mean_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
                        help='1:FD001, 2: FD002, 3:FD003, 4:FD004',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=1)
    parser.add_argument('-t', '--teacher',
                        help='[ind, ncl, snapshot]',
                        type=str,
                        choices=['ind', 'ncl', 'snapshot'],
                        default='ncl')
    parser.add_argument('-g', '--GPU',
                        help='set GPU',
                        type=int,
                        choices=[0, 1, 2, 3],
                        default=0)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    main(args)