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
from sklearn.metrics import mean_absolute_error,mean_squared_error
from dataset import Origin_data, pre_custom_collate_fn, OwnData
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
    logger.info(BLUE + "Testing Starts!" + ENDC)

def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace(BLUE, '')
    content = content.replace(ENDC, '')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def test(args):
    device = torch.device(args.device)
    model = args.model
    dataset= args.dataset
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"test_{model}_{dataset}_{timestamp}"
    father_folder_name = args.save
    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)
    folder_path = os.path.join(father_folder_name, folder_name)
    os.makedirs(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
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

    test_data = Origin_data(os.path.join(args.dataset_path, args.dataset, 'origin/test.pkl'))
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, collate_fn=pre_custom_collate_fn)

    model = my_model(feature_dim = args.feat_dim, hidden_dim = args.hidden_dim)
    model = model.to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_test_step = 0
    total_MAE = 0
    total_MSE = 0
    total_SRC = 0
    print_init_msg(logger, args)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            mean_pooling_vec, merge_text_vec, label = batch

            label = label.type(torch.float32)

            z, output, z_dist  = model.forward(mean_pooling_vec, merge_text_vec)

            output = output.to('cpu')
            label = label.to('cpu')
            output = np.array(output)
            label = np.array(label)

            MAE = mean_absolute_error(label, output)
            SRC, _ = spearmanr(output, label)
            MSE = mean_squared_error(y_pred=output, y_true=label)
            total_test_step += 1
            total_MAE += MAE
            total_SRC += SRC
            total_MSE += MSE

    logger.warning(f"[ Test Result ]:  \n {args.metric[0]} = {total_MSE / total_test_step}" f"\n{args.metric[1]} = {total_SRC / total_test_step}\n{args.metric[2]} = {total_MAE / total_test_step}\n")
    logger.info("Test is ended!")
    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def main(args):
    seed_init(args.seed)
    test(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')
    parser.add_argument('--metric', default=['MSE', 'SRC', 'MAE'], type=list, help='the judgement of the testing')
    parser.add_argument('--save', default=r'RESULT', type=str, help='folder to save the results')
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
    parser.add_argument('--feat_dim', default=1536, type=int, help='dims for all_modal_features')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dims for training')
    parser.add_argument('--dataset', default='ICIP', choices=["ICIP", "SMPD", "Instagram"], type=str, help='dataset')
    parser.add_argument('--dataset_path', default=r'datasets', type=str, help='path of dataset')
    parser.add_argument('--model', default='Info-Hetero', type=str, help='model')
    parser.add_argument('--model_path', default=r"./RESULT/train_IB_pretrain_SMPD_MSE_2024-11-30_12-05-25/trained_model/model_180.pth", type=str, help='path of trained model')
    args = parser.parse_args()
    main(args)