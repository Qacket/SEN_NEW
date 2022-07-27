import json

import torch
import argparse

from utils import interpolate


def main(args):



    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    t_vae = torch.load(args.model_dir + args.task_vae_name).to(device)
    t_vae.eval()


    for i in range(100):
        z = torch.randn(1, 1, args.latent_size).to(device)
        sos = "<sos>"
        sample = t_vae.inference(30, sos, z)
        print(sample)

        print("----------------------------1")
        samples, text1, text2 = interpolate(t_vae, 20, sos, 10)
        print("First sentence:", text1)
        print("Second sentence:", text2)

        for sample in samples:
            print(sample)
        print("----------------------------2")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('-ls', '--latent_size', type=int, default=50)
    parser.add_argument('-tvn', '--task_vae_name', type=str, default='/t_vae_new_2')
    args = parser.parse_args()

    main(args)
