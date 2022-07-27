import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from mymodel import T_VAE, A_VAE, My_Model
from sen import SEN
from train import train_mymodel, train_t_vae, train_a_vae


def main(args):

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_data = SEN(
        data_dir=args.data_dir,
        create_data=args.create_data,
        max_sequence_length=args.max_sequence_length
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # a_params = dict(
    #     E_in=args.annotator_dim,
    #     middle_size=20,
    #     hidden_size=20,
    #     latent_size=8,
    #     D_out=args.annotator_dim,
    #     device=device
    # )
    # # 初始化工人vae
    # a_vae = A_VAE(**a_params)
    # # 训练工人vae
    # train_a_vae(args, device, train_loader, a_vae)
    #
    #
    # t_params = dict(
    #     train_data=train_data,
    #     vocab_size=train_data.vocab_size,
    #     embedding_size=args.embedding_size,
    #     hidden_size=args.hidden_size,
    #     latent_size=args.latent_size,
    #     num_layers=args.num_layers,
    #     embedding_dropout=args.embedding_dropout,
    #     device=device
    # )
    # # 初始化任务vae
    # t_vae = T_VAE(**t_params)
    # # 训练任务vae
    # train_t_vae(args, device, train_loader, t_vae)


    a_vae = torch.load(args.model_dir + args.annotator_vae_name)
    t_vae = torch.load(args.model_dir + args.task_vae_name)


    a_vae.trainable = False
    t_vae.trainable = False

    mymodel = My_Model(a_vae, t_vae)
    train_mymodel(args, device, train_loader, mymodel)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)

    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-a_dim', '--annotator_dim', type=int, default=203)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-ls', '--latent_size', type=int, default=50)

    parser.add_argument('-taw', '--annotator_writer', type=str, default='logs/a_vae_new')
    parser.add_argument('-avn', '--annotator_vae_name', type=str, default='/a_vae_new')

    parser.add_argument('-ttw', '--task_writer', type=str, default='logs/t_vae_new')
    parser.add_argument('-tvn', '--task_vae_name', type=str, default='/t_vae_new')



    parser.add_argument('-mw', '--mymodel_writer', type=str, default='logs/mymodel7')
    parser.add_argument('-mn', '--mymodel_name', type=str, default='/mymodel7')

    args = parser.parse_args()

    main(args)

