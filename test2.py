import random
from torch import optim, nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from mymodel import Encoder, VAE, Decoder, T_VAE, My_Model, latent_loss, reconstruction_loss
from sen import SEN
import torch.nn.functional as F

from utils import get_batch

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def train():

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_data = SEN(
        data_dir='./datasets',
        create_data=True,
        max_sequence_length=60
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=512,
        shuffle=True
    )

    # 初始化工人vae
    annotator_input_dim = 203
    annotator_encoder = Encoder(annotator_input_dim, 20, 20)
    annotator_decoder = Decoder(8, 20, annotator_input_dim)
    annotator_vae = VAE(annotator_encoder, annotator_decoder, 20, 8)


    # 初始化任务vae
    t_params = dict(
        train_data=train_data,
        vocab_size=train_data.vocab_size,
        embedding_size=300,
        hidden_size=128,
        latent_size=50,
        num_layers=1,
        embedding_dropout=0.5,
        device=device
    )
    # 初始化任务vae
    task_vae = T_VAE(**t_params)


    # 初始化mymodel
    mymodel = My_Model(annotator_vae, task_vae).to(device)
    criterion = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    nlloss = nn.NLLLoss()
    optimizer = optim.Adam(mymodel.parameters(), lr=0.001)
    l1 = None
    l2 = None
    l3 = None
    l4 = None
    l5 = None
    l = None


    torch.set_grad_enabled(True)
    mymodel.train()
    writer = SummaryWriter('logs/ssss')
    for epoch in range(5000):
        states = mymodel.t_vae.init_hidden(512)
        count = 0
        sum = 0
        for iteration, batch in enumerate(train_loader):
            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            # task_id = data[1]
            # annotator_inputs = data[2]
            # label_tensor = data[3]
            # task_inputs = data[4]
            optimizer.zero_grad()
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, 203).type(torch.float32)  # 工人input

            z1, dev_annotator = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator

            # 任务
            task = task.to(device)
            target = target.to(device)

            t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()


            z = torch.cat((z1, t_z.squeeze()), 1)  # z1 z2 结合

            dev_label = mymodel(z)  # 获得生成的标注^dev_label
            label_tensor = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(torch.LongTensor).to(
                device)

            loss1 = latent_loss(mymodel.a_vae.z_mean, mymodel.a_vae.z_sigma)
            l1 = loss1.data.item()

            loss2 = criterion(dev_annotator, annotator_inputs)
            l2 = loss2.data.item()

            loss3 = latent_loss(t_mean, t_logv)
            l3 = loss3.data.item()

            loss4 = reconstruction_loss(nlloss, t_output, target)
            l4 = loss4.data.item()

            loss5 = loss_fn(dev_label, label_tensor)
            l5 = loss5.data.item()

            loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss.backward()
            optimizer.step()
            l = loss.data.item()


            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()

            for i in range(len(annotator_id)):
                if pred_label[i] == target_label[i]:
                    count += 1
            sum += len(annotator_id)
        acc = count / sum
        # writer.add_scalar(tag='loss1', scalar_value=l1, global_step=epoch)
        # writer.add_scalar(tag='loss2', scalar_value=l2, global_step=epoch)
        # writer.add_scalar(tag='loss3', scalar_value=l3, global_step=epoch)
        # writer.add_scalar(tag='loss4', scalar_value=l4, global_step=epoch)
        writer.add_scalar(tag='loss5', scalar_value=l5, global_step=epoch)
        writer.add_scalar(tag='loss', scalar_value=l, global_step=epoch)
        print(epoch, l, l5, "ACC:" + str(acc))

    # torch.save(mymodel, model_dir + 'mymodel_%s' % dataset)

if __name__ == '__main__':
    # 训练
    train()
