import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter
from VTP import VTP
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
import argparse

if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter()
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')
    # device = "cuda" if train_on_gpu else "cpu"
    device = "cpu"

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def test(model, net):

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').to(device)
            txt = input.get('txt').to(device)
            # print(txt.shape) # torch.Size([4, 96, 75, 4, 8])
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            txt_origin = input.get('txt_origin')
            encoder = StaticTokenizerEncoder(txt_origin, tokenize=lambda s: s.split())
            encoded_data = [encoder.encode(example) for example in txt_origin]
            if opt.with_vtp:
                attn, y = net(vid, encoded_data)
            else:
                y = net(vid, encoded_data)
            # y = net(vid)
            # print(y.shape)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={},loss={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean(), loss))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

def train(model, net):
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)

    print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss()
    tic = time.time()
    min_loss = 999999999.
    train_loss = 0
    train_wer = 0
    train_cer = 0
    train_wer_list = []
    train_cer_list = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').to(device)
            txt = input.get('txt').to(device)
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            txt_origin = input.get('txt_origin')
            encoder = StaticTokenizerEncoder(txt_origin, tokenize=lambda s: s.split())
            encoded_data = [encoder.encode(example) for example in txt_origin]
            optimizer.zero_grad()
            if opt.with_vtp:
                attn, y = net(vid, encoded_data)
            else:
                y = net(vid, encoded_data)
            # y = net(vid)
            # print('shape:', y.shape)
            # print(txt.shape)
            train_loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            train_loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer_list.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer_list.extend(MyDataset.cer(pred_txt, truth_txt))
            train_wer = MyDataset.wer(pred_txt, truth_txt)
            train_cer = MyDataset.cer(pred_txt, truth_txt)

            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                # writer.add_scalar('train loss', loss, tot_iter)
                # writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, train_loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (loss, wer, cer) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                if loss < min_loss:
                    # torch.save(model, '/home/max/Desktop/LipNet-PyTorch-master/weights/gru_weight.pkl')
                    min_loss = loss
                if(not opt.is_optimize):
                    exit()
        (test_loss, test_wer, test_cer) = test(model, net)

if(__name__ == '__main__'):
    print("Loading options...")
    model = VTP(opt.with_vtp)
    net = model.to(device)
    torch.manual_seed(opt.random_seed)
    # torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
