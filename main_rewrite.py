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
# from beam_search import Beam_search
from transformers import AutoTokenizer

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

def greedy_decoder(data):
    data = data.detach().cpu().numpy()
    # index for largest probability each row
    return [np.argmax(s) for s in data[0]]

# def beam_search_decoder(data, k):
#     sequences = [[list(), 1.0]]
#     # walk over each step in sequence
#     for row in data:
#         all_candidates = list()
#         # expand each current candidate
#         for i in range(len(sequences)):
#             seq, score = sequences[i]
#             for j in range(len(row)):
#                 candidate = [seq + [j], score * - np.log(row[j])]
#                 all_candidates.append(candidate)
#         # order all candidates by score
#         ordered = sorted(all_candidates, key=lambda tup :tup[1])
#         # select k best
#         sequences = ordered[:k]
#     return sequences


def test(model, net):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', bos_token='[BOS]', eos_token='[EOS]')
    tokenizer.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]'})
    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')
            
        print(f'num_test_data:{len(dataset.data)}')  
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
            truth_txt = input.get('origin_txt')
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            
            if opt.with_vtp:
                y, attn = net(vid, txt)
            else:
                y = net(vid, txt)
            
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred = greedy_decoder(y)
            pred_txt = tokenizer.decode(pred)
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
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
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

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', bos_token='[BOS]', eos_token='[EOS]')
    tokenizer.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]'})
    
    train_wer = []
    train_cer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').to(device)
            txt = input.get('txt').to(device)
            truth_txt = input.get('origin_txt')
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)
            
            optimizer.zero_grad()
            if opt.with_vtp:
                y, attn = net(vid, txt)
            else:
                y = net(vid, txt)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            # print(y.detach().cpu().numpy()[0].shape)
            tot_iter = i_iter + epoch*len(loader)
            pred = greedy_decoder(y)

            pred_txt = tokenizer.decode(pred)
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                writer.add_scalar('train wer', np.array(train_cer).mean(), tot_iter)  
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print(f'epoch={epoch},tot_iter={tot_iter},eta={eta},loss={loss},train_wer={np.array(train_wer).mean()}')
                print(''.join(101*'-'))
                
            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net)
                print(f'i_iter={tot_iter},lr={show_lr(optimizer)},loss={loss},wer={wer},cer={cer}')
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                torch.save(model, 'weights/vtp_model.pkl')
                if(not opt.is_optimize):
                    exit()
                
if(__name__ == '__main__'):
    print("Loading options...")
    model = VTP()
    model = model.to(device)
    # net = nn.DataParallel(model).to(device)
        
    torch.manual_seed(opt.random_seed)
    # torch.cuda.manual_seed_all(opt.random_seed)
    train(model, model)