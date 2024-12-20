import logging
import os
import sys
sys.path.append("./src") #相对路径或绝对路径
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from dataset import Origin_data, pre_custom_collate_fn
# from Predict_model import RRCP_Model as my_model
from Pretrain_model import IB_Regressor as my_model
import random
import numpy as np
import pickle
import ipdb
from scipy.stats import spearmanr

BLUE = '\033[94m'
ENDC = '\033[0m'

def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")
    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")
    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")
    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")
    logger.info(BLUE + "Hidden Dim: " + ENDC + f"{args.hidden_dim}")
    logger.info(BLUE + "Beta: " + ENDC + f"{args.beta}")
    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"train_{args.model}_{args.dataset}_{args.metric}_{timestamp}"
    father_folder_name = args.save
    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)
    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    os.mkdir(os.path.join(folder_path, "trained_model"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return father_folder_name, folder_name, logger
    

def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")
    for i in range(len(model_name_list)):
        if model_name_list[i] != f'model_{min_turn}.pth':
            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):
    print(msg)
    sys.exit(1)


def train_val(args):

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)
    device = torch.device(args.device)
    
    train_data = Origin_data(os.path.join(args.dataset_path, args.dataset, 'origin/train.pkl'))
    valid_data = Origin_data(os.path.join(args.dataset_path, args.dataset, 'origin/valid.pkl'))
    test_data = Origin_data(os.path.join(args.dataset_path, args.dataset, 'origin/test.pkl'))
    
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=pre_custom_collate_fn)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=pre_custom_collate_fn)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, collate_fn=pre_custom_collate_fn)

    model = my_model(feature_dim = args.feat_dim, hidden_dim = args.hidden_dim)

    # model = my_model(retrieval_num=args.retrieval_num)
    model = model.to(device)

    if args.loss == 'BCE':
        loss_fn = torch.nn.BCELoss()
    elif args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        force_stop('Invalid parameter loss!')
        loss_fn = None

    loss_fn.to(device)
    if args.optim == 'Adam':
        optim = Adam(model.parameters(), args.lr)
    elif args.optim == 'SGD':
        optim = SGD(model.parameters(), args.lr)
    else:
        force_stop('Invalid parameter optim!')
        optim = None

    decayRate = args.decay_rate

    min_total_valid_loss = 1008611
    min_turn = 0

    print_init_msg(logger, args)
    for i in range(args.epochs):
        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")
        min_train_loss, total_valid_loss = run_one_epoch(args, model, loss_fn, optim, train_data_loader,
                                                         valid_data_loader,
                                                         device)
        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")
        logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")
        
        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss
            min_turn = i + 1
        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")
        torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")
        logger.info("Model has been saved successfully!")
        if (i + 1) - min_turn > args.early_stop_turns:
            break
    delete_model(father_folder_name, folder_name, min_turn)
    logger.info(BLUE + "Training is ended!" + ENDC)
    
    # 训练完成后获取所有样本的 z
    model.eval()  
    with torch.no_grad():
        z_train_all = []
        z_valid_all = []
        z_test_all = []
        # train_set
        for batch in tqdm(train_data_loader, desc='Saving Z_train Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            mean_pooling_vec, merge_text_vec, label = batch
            z, _, _ = model.forward(mean_pooling_vec, merge_text_vec)
            z_train_all.append(z)
        z_train_all = torch.cat(z_train_all, dim=0)  # 合并所有批次的 Z
        z_train_all_np = z_train_all.cpu().numpy()
        with open(os.path.join(args.dataset_path, args.dataset, f'origin/z_train_beta_{args.beta}.pkl'), 'wb') as f:
            pickle.dump(z_train_all_np, f)  # 保存为 pickle 文件
        # valid_set
        for batch in tqdm(valid_data_loader, desc='Saving Z_valid Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            mean_pooling_vec, merge_text_vec, label = batch
            z, _, _ = model.forward(mean_pooling_vec, merge_text_vec)
            z_valid_all.append(z)
        z_valid_all = torch.cat(z_valid_all, dim=0)  # 合并所有批次的 Z
        z_valid_all_np = z_valid_all.cpu().numpy()
        with open(os.path.join(args.dataset_path, args.dataset, f'origin/z_valid_beta_{args.beta}.pkl'), 'wb') as f:
            pickle.dump(z_valid_all_np, f)  # 保存为 pickle 文件
        # test_set
        for batch in tqdm(test_data_loader, desc='Saving Z_test Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            mean_pooling_vec, merge_text_vec, label = batch
            z, _, _ = model.forward(mean_pooling_vec, merge_text_vec)
            z_test_all.append(z)
        z_test_all = torch.cat(z_test_all, dim=0)  # 合并所有批次的 Z
        z_test_all_np = z_test_all.cpu().numpy()
        with open(os.path.join(args.dataset_path, args.dataset, f'origin/z_test_beta_{args.beta}.pkl'), 'wb') as f:
            pickle.dump(z_test_all_np, f)  # 保存为 pickle 文件        
        

        
def run_one_epoch(args, model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    model.train()
    min_train_loss = 1008611
    for batch in tqdm(train_data_loader, desc='Training Progress'):
        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
        mean_pooling_vec, merge_text_vec, label = batch
        z, output, z_dist = model.forward(mean_pooling_vec, merge_text_vec)
        target = label.type(torch.float32)
        ### ib_loss ###
        ib_loss = model.kl_divergence(z_dist)
        ### ib_loss ###
        mse_loss = loss_fn(output, target)
        # ipdb.set_trace()
        loss = mse_loss + args.beta * ib_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        if min_train_loss > loss:
            min_train_loss = loss
    
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_data_loader, desc='Validating Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            mean_pooling_vec, merge_text_vec, label = batch

            z, output, z_dist  = model.forward(mean_pooling_vec, merge_text_vec)

            target = label.type(torch.float32)

            mes_loss = loss_fn(output, target)

            total_valid_loss += loss

    return min_train_loss, total_valid_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in training')
    parser.add_argument('--metric', default='MSE', type=str, help='the judgement of the training')
    parser.add_argument('--save', default=r'RESULT', type=str,  help='folder to save the results')
    parser.add_argument('--epochs', default=1000, type=int, help='max number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--early_stop_turns', default=10, type=int, help='early stop turns of training')
    parser.add_argument('--loss', default='MSE', type=str, help='loss function, options: BCE, MSE')
    parser.add_argument('--optim', default='Adam', type=str, help='optim, options: SGD, Adam')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--dataset', default='ICIP', choices=["ICIP", "SMPD", "Instagram"], type=str, help='dataset')
    parser.add_argument('--dataset_path', default=r'datasets', type=str, help='path of dataset')
    parser.add_argument('--model', default='IB_pretrain', type=str, help='model')
    parser.add_argument('--feat_dim', default=1536, type=int, help='dims for all_modal_features')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dims for training')
    parser.add_argument('--beta', default=0.00005, type=float, help='parameter beta for ib_loss')   
    args = parser.parse_args()
    
    seed_init(args.seed)
    train_val(args)
    

if __name__ == '__main__':
    main()
