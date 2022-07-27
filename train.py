import random
from multiprocessing import cpu_count
from torch.autograd import Variable
import numpy as np
import torch
from torch import optim, nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from loss import T_VAE_Loss, A_VAE_Loss
from mymodel import T_VAE, A_VAE
from utils import get_batch

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def train_a_vae(args, device, train_loader, a_vae):

    a_vae = a_vae.to(device)
    A_loss = A_VAE_Loss()
    optimizer = optim.Adam(a_vae.parameters(), lr=args.learning_rate)


    torch.set_grad_enabled(True)
    a_vae.train()
    writer = SummaryWriter(args.annotator_writer)

    for epoch in range(args.epochs):

        for batch in tqdm(train_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            optimizer.zero_grad()
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, args.annotator_dim).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = a_vae(annotator_inputs)

            # 工人 loss
            a_KL_loss, a_recon_loss = A_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)

            a_mloss = a_KL_loss + a_recon_loss

            a_mloss.backward()

            optimizer.step()


        writer.add_scalar(tag='a_KL_loss', scalar_value=a_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_recon_loss', scalar_value=a_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_mloss', scalar_value=a_mloss.data.item(), global_step=epoch)
        print(epoch, a_mloss.data.item())

    torch.save(a_vae, args.model_dir + args.annotator_vae_name)


def train_t_vae(args, device, train_loader, t_vae):

    t_vae = t_vae.to(device)
    T_loss = T_VAE_Loss()
    optimizer = optim.Adam(t_vae.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    t_vae.train()
    writer = SummaryWriter(args.task_writer)

    for epoch in range(args.epochs):
        states = t_vae.init_hidden(args.batch_size)
        for batch in tqdm(train_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            optimizer.zero_grad()

            # 任务
            task = task.to(device)
            target = target.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()

            # 任务loss
            t_KL_loss, t_recon_loss = T_loss(mu=t_mean, log_var=t_logv, x_hat_param=t_output, x=target)

            t_mloss = t_KL_loss + t_recon_loss

            t_mloss.backward()

            optimizer.step()

        writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_mloss', scalar_value=t_mloss.data.item(), global_step=epoch)
        print(epoch, t_KL_loss.data.item(), t_recon_loss.data.item(), t_mloss.data.item())

    torch.save(t_vae, args.model_dir + args.task_vae_name)


def train_mymodel(args, device, train_loader, mymodel):

    mymodel = mymodel.to(device)
    A_loss = A_VAE_Loss()
    T_loss = T_VAE_Loss()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(mymodel.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    mymodel.train()
    writer = SummaryWriter(args.mymodel_writer)

    for epoch in range(args.epochs):
        states = mymodel.t_vae.init_hidden(args.batch_size)
        count = 0
        sum = 0
        for iteration, batch in enumerate(train_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            optimizer.zero_grad()
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, args.annotator_dim).type(torch.float32)  # 工人input

            # 工人vae
            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)


            # 任务
            task = task.to(device)
            target = target.to(device)

            # 任务vae
            t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()


            z = torch.cat((a_z, t_z.squeeze()), 1)

            dev_label = mymodel(z)

            label_tensor = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(torch.LongTensor).to(device)


            # # 工人 loss
            # a_KL_loss, a_recon_loss = A_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)
            #
            # 任务loss
            # t_KL_loss, t_recon_loss = T_loss(mu=t_mean, log_var=t_logv, x_hat_param=t_output, x=target)


            # 监督loss
            sup_loss = loss_fn(dev_label, label_tensor)


            # loss = a_KL_loss + a_recon_loss + t_KL_loss + t_recon_loss + sup_loss

            # loss = t_KL_loss + t_recon_loss + sup_loss

            loss = sup_loss

            loss.backward()

            optimizer.step()

            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()

            for i in range(len(annotator_id)):
                if pred_label[i] == target_label[i]:
                    count += 1
            sum += len(annotator_id)

        acc = count / sum
        # writer.add_scalar(tag='a_KL_loss', scalar_value=a_KL_loss.data.item(), global_step=epoch)
        # writer.add_scalar(tag='a_recon_loss', scalar_value=a_recon_loss.data.item(), global_step=epoch)
        # writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        # writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='sup_loss', scalar_value=sup_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='loss', scalar_value=loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='acc', scalar_value=acc, global_step=epoch)
        print(epoch, sup_loss.data.item(), loss.data.item(), "ACC:" + str(acc))

    torch.save(mymodel, args.model_dir + args.mymodel_name)



