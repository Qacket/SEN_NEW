import math
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sen import SEN
from utils import idx_to_word, get_batch


class Mymodel_generate:
    def __init__(self, crowd_file, truth_file, **kwargs):
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()
        self.label_set = label_set

    def redundancy_distribution(self, count):
        sum = 0
        for item in count:
            sum += item
        miu = sum / len(count)
        # print(miu)
        s = 0
        for item in count:
            s += (item - miu) ** 2
        sigma = math.sqrt(s / len(count))
        # print(sigma)
        return miu, sigma
    def gete2wlandw2el(self):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, worker, label = line.split('\t')
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker,label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example,label])

            if label not in label_set:
                label_set.append(label)

        e2s = {}
        for example in e2wl:
            e2s[example] = len(e2wl[example])

        example_miu, example_sigma = self.redundancy_distribution(list(e2s.values()))
        self.example_miu = example_miu
        self.example_sigma = example_sigma

        return e2wl, w2el, label_set

    def run(self):
        pass

    def generate_fixed_task(self, exist_task, generate_file):

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # Load the data
        train_data = SEN(data_dir="./datasets", create_data=False, max_sequence_length=60)
        # Batchify the data
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)

        mymodel = torch.load('./model/mymodel7')
        mymodel.to(device)
        f_save = open(generate_file, 'w')


        taskidx2task = {}
        annotator_id_list = []
        for data in tqdm(train_loader):
            annotator_id, answer, task, target, task_lengths = get_batch(data)
            task_id = np.array(data["task_id"]).astype(dtype=int)[0]
            annotator_id = annotator_id[0]
            if task_id not in taskidx2task:
                taskidx2task[task_id] = task
            if annotator_id not in annotator_id_list:
                annotator_id_list.append(annotator_id)

        states = mymodel.t_vae.init_hidden(1)
        for task_id, task_input in taskidx2task.items():
            for annotator_id in annotator_id_list:

                annotator_id = np.array(annotator_id).astype(dtype=int)
                annotator_tensor = torch.from_numpy(annotator_id).unsqueeze(dim=0).to(device)
                annotator_inputs = F.one_hot(annotator_tensor, 203).type(torch.float32)  # ??????input


                # ??????vae
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)

                task_input = task_input.to(device)

                # ??????vae
                t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task_input, task_lengths, states)
                states = states[0].detach(), states[1].detach()

                # label
                z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)  # z1 z2 ??????
                dev_label = mymodel(z)

                # prediction = torch.max(F.softmax(dev_label), 1)[1]
                # pred_label = prediction.cpu().data.numpy().squeeze()

                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(2)), p=p.ravel())

                f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(label_new) + '\n')

        f_save.close()




        # states = mymodel.t_vae.init_hidden(1)
        # for data in tqdm(train_loader):
        #
        #     annotator_id, answer, task, target, task_lengths = get_batch(data)
        #
        #     task_id = np.array(data["task_id"]).astype(dtype=int)
        #     annotator_id = np.array(annotator_id).astype(dtype=int)
        #     label_np = np.array(answer).astype(dtype=int)
        #
        #     if task_id in list(map(int, exist_task)):  # current_task???????????? ????????????
        #         f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
        #     else:
        #         annotator_tensor = torch.from_numpy(annotator_id).to(device)
        #         annotator_inputs = F.one_hot(annotator_tensor, 203).type(torch.float32)  # ??????input
        #         # ??????vae
        #         a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
        #
        #         task = task.to(device)
        #         # ??????vae
        #         t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
        #         states = states[0].detach(), states[1].detach()
        #
        #         # label
        #         z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)  # z1 z2 ??????
        #
        #
        #         dev_label = mymodel(z)
        #
        #         # prediction = torch.max(F.softmax(dev_label), 1)[1]
        #         # pred_label = prediction.cpu().data.numpy().squeeze()
        #
        #         p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
        #         label_new = np.random.choice(np.array(range(2)), p=p.ravel())
        #
        #         # print(pred_label)
        #         #
        #         # p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
        #         # label_new = np.random.choice(np.array(range(2)), p=p.ravel())
        #
        #         f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        # f_save.close()

    # def generate_fixed_annotator(self, exist_annotator, generate_file):
    #
    #     device, train_iter, task_voc_size, task_pad_idx = deal_data(batch_size=1, shuffle=False)
    #
    #     mymodel = torch.load('/mnt/4T/scj/sentiment/Mymodel_generate/mymodel_sentiment')
    #     mymodel.to(device)
    #     f_save = open(generate_file, 'w')
    #     for data in tqdm(train_iter):
    #         batch_size = len(data.task)
    #
    #         task_id = data.task_id.cpu().detach().numpy().astype(np.int64)  # ???????????????current_task
    #         annotator_id = data.annotator_id.cpu().detach().numpy().astype(np.int64)
    #         label_np = data.answer.cpu().detach().numpy()
    #
    #         if annotator_id in list(map(int, exist_annotator)):  # current_annotator???????????? ????????????
    #             f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
    #         else:
    #             # task_id = data[1].to(device)
    #             annotator_inputs = F.one_hot(data.annotator_id, 203).type(torch.float32)
    #             # label_tensor = data[3].to(device)
    #             task_inputs = data.task
    #
    #             batch_task_length = [70] * batch_size - np.sum(data.task.cpu().detach().numpy() == 1, axis=1)
    #             batch_task_length = torch.from_numpy(batch_task_length)
    #
    #             z1, dev_annotator = mymodel.annotator_vae(annotator_inputs, device)  # ?????? ????????????z1  ???????????????^ dev_annotator
    #             z2, dev_task = mymodel.task_vae(task_inputs, batch_task_length, device)  # ?????? ????????????z2  ???????????????^ dev_task
    #             z = torch.cat((z1, z2), 1)  # z1 z2 ??????
    #             dev_label = mymodel(z)  # ?????????????????????^dev_label
    #
    #             p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
    #             label_new = np.random.choice(np.array(range(2)), p=p.ravel())
    #             # label_p = F.softmax(dev_label, dim=1)
    #             # label_new = label_p.argmax(1).cpu().numpy()
    #             f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
    #     f_save.close()
    #
    # def generate(self, sample_file, generate_file):
    #
    #     f_open = open(sample_file, 'r')
    #     reader = f_open.readlines()
    #     reader = [line.strip("\n") for line in reader]
    #     examples = []
    #     for line in reader:
    #         example, worker, label = line.split('\t')
    #         if example not in examples:
    #             examples.append(example)
    #     f_open.close()
    #
    #     device, train_iter, task_voc_size, task_pad_idx = deal_data(batch_size=1, shuffle=False)
    #
    #     mymodel = torch.load('/mnt/4T/scj/sentiment/Mymodel_generate/mymodel_sentiment')
    #     mymodel.to(device)
    #     f_save = open(generate_file, 'w')
    #
    #
    #
    #     for data in tqdm(train_iter):
    #         batch_size = len(data.task)
    #
    #         task_id = data.task_id.cpu().detach().numpy().astype(np.int64)
    #         annotator_id = data.annotator_id.cpu().detach().numpy().astype(np.int64)
    #         label_np = data.answer.cpu().detach().numpy()
    #         if task_id in list(map(int, examples)):  # current_task???????????? ????????????
    #             # task_id = data[1].to(device)
    #             annotator_inputs = F.one_hot(data.annotator_id, 203).type(torch.float32)
    #             # label_tensor = data[3].to(device)
    #             task_inputs = data.task
    #
    #             batch_task_length = [70] * batch_size - np.sum(data.task.cpu().detach().numpy() == 1, axis=1)
    #             batch_task_length = torch.from_numpy(batch_task_length)
    #
    #             z1, dev_annotator = mymodel.annotator_vae(annotator_inputs,
    #                                                           device)  # ?????? ????????????z1  ???????????????^ dev_annotator
    #             z2, dev_task = mymodel.task_vae(task_inputs, batch_task_length, device)  # ?????? ????????????z2  ???????????????^ dev_task
    #             z = torch.cat((z1, z2), 1)  # z1 z2 ??????
    #             dev_label = mymodel(z)  # ?????????????????????^dev_label
    #
    #             p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
    #             label_new = np.random.choice(np.array(range(2)), p=p.ravel())
    #             # label_p = F.softmax(dev_label, dim=1)
    #             # label_new = label_p.argmax(1).cpu().numpy()
    #             f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
    #     f_save.close()


