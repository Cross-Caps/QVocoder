import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from   torchsummary import summary
from   torch.autograd.variable import Variable
from env import AttrDict
import os
import json

from meldatasetQ import MelDataset, mel_spectrogram, get_dataset_filelist
from modelsQ5B import QGenerator, QMultiPeriodDiscriminator, QMultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from torch.utils.data import DistributedSampler, DataLoader
import torchaudio
########### ll imports #######################

import numpy as np
from llhelper import get_weights,get_random_weights, normalize_direction, viz_histogram_weights, create_viz 
#################################################

config_file = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/cp_hifigan7/config.json"

with open(config_file) as f:
        data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

torch.manual_seed(h.seed)
    
if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cpu')
else:
        device = torch.device('cpu')


config_file = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/cp_hifigan7/config.json"
with open(config_file) as f:
        data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

torch.manual_seed(h.seed)


net = QGenerator(h).to(device)
checkpoint_file = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/cp_hifigan7/g_00775000"
state_dict_g = load_checkpoint(checkpoint_file, device)
net.load_state_dict(state_dict_g['generator'])

mpd = QMultiPeriodDiscriminator().to(device)
msd = QMultiScaleDiscriminator().to(device)
cp_do = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/cp_hifigan7/do_00775000"
state_dict_do = load_checkpoint(cp_do, device)
# generator.load_state_dict(state_dict_g['generator'])
mpd.load_state_dict(state_dict_do['mpd'])
msd.load_state_dict(state_dict_do['msd'])




with open('/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/training.txt', 'r', encoding='utf-8') as fi:
        training_files = [os.path.join('/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs/', x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]


trainset = MelDataset(training_files=training_files, segment_size=h.segment_size,n_fft= h.n_fft,num_mels= h.num_mels,
                         hop_size= h.hop_size,win_size= h.win_size,sampling_rate= h.sampling_rate,fmin= h.fmin,fmax= h.fmax, n_cache_reuse=0,
                          shuffle= True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=None, base_mels_path=None)
    
   

train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

correct = 0
total_loss = 0
total = 0 

wweight    = get_weights(net)

converged_weights = get_weights(net)

numebr_of_points = 21 ; small_range = -1.0 ; large_range =  1.0

xcoordinates = np.linspace(small_range, large_range, num=numebr_of_points) 
ycoordinates = np.linspace(small_range, large_range, num=numebr_of_points) 

xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
inds = np.array(range(numebr_of_points**2))
s1   = xcoord_mesh.ravel()[inds]
s2   = ycoord_mesh.ravel()[inds]

coordinate = np.c_[s1,s2]

print('From ',small_range,' to ',large_range,' with ',numebr_of_points,' total number of coordinate: ', numebr_of_points**2)


copy_of_the_weights = [ w.clone() for w in converged_weights]

random_direction1 = get_random_weights(copy_of_the_weights)
random_direction2 = [w.clone() for w in random_direction1]

for d,w in zip(random_direction1,copy_of_the_weights):
    d = d.to(device)
    w = w.to(device)
    # print("random direction wale code me ",w.shape, d.shape)
    normalize_direction(d,w,'filter')



# temp = []
# for d,w in zip(random_direction2,copy_of_the_weights):
#     d_re   = d.view((d.shape[0],-1))
#     d_norm = d_re.norm(dim=1, keepdim=True)
#     # d_norm = d_re.norm(dim=(1),keepdim=True)[:,:,None]
#     w_re   = w.view((w.shape[0],-1))
#     w_norm = w_re.norm(dim=1, keepdim=True)
#     # w_norm = w_re.norm(dim=(1),keepdim=True)[:,:,None]

#     temp.append(d.to(device) * (w_norm.to(device)/(d_norm.to(device)+1e-10)))
#     # temp.append(d * (w_norm/(d_norm+1e-10)))
#     d.data =  d.to(device) * (w_norm.to(device)/(d_norm.to(device)+1e-10))
#     # d.data =  d* (w_norm/(d_norm+1e-10))
    
# print("Outside1 ",w.shape, d.shape)

# checking if rad directions are same or not

# for x, xx in zip(random_direction1,random_direction2):
#     print(np.allclose(x.cpu().numpy(),xx.cpu().numpy()))
    
# for x, xx in zip(random_direction1,temp):
#     print(np.allclose(x.cpu().numpy(),xx.cpu().numpy()))



random_direction1 = get_random_weights(copy_of_the_weights)
random_direction2 = get_random_weights(copy_of_the_weights)

for d1,d2,w in zip(random_direction1,random_direction2,copy_of_the_weights):
    if(w.dim()==1):
            w_norm  = w.view((w.shape[0])).norm(dim=(0),keepdim=True)[:]
            d_norm1 = d1.view((d1.shape[0])).norm(dim=(0),keepdim=True)[:]
            d_norm2 = d2.view((d2.shape[0])).norm(dim=(0),keepdim=True)[:]
    
    else:
          w_norm  = w.view((w.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None]
          d_norm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None]
          d_norm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None]
    
    
    d1.data = d1.to(device) * (w_norm/(d_norm1.to(device)+1e-10))
    d2.data = d2.to(device) * (w_norm/(d_norm2.to(device)+1e-10))
    # print(d1.data.shape, d2.data.shape)

# print("Outside2 ",w.shape, d.shape)


# start the evaluation
loss_list = np.zeros((numebr_of_points,numebr_of_points)); acc_list  = np.zeros((numebr_of_points,numebr_of_points))
col_value = 0

loss_list = np.zeros((numebr_of_points,numebr_of_points)); acc_list  = np.zeros((numebr_of_points,numebr_of_points))
col_value = 0

for count, ind in enumerate(inds):
    
    # change the weight values
    coord   = coordinate[count]
    changes = [d0.to(device)*coord[0] + d1.to(device)*coord[1] for (d0, d1) in zip(random_direction1, random_direction2)]
    for (p, w, d) in zip(net.parameters(), wweight, changes): p.data = w + d

    # start the evaluation
    correct = 0; total_loss = 0; total = 0 
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            # x, y, _, y_mel = batch
            data, y, _, y_mel = batch
            # print(y.shape)
            data_first = torchaudio.functional.compute_deltas(data)
            data_second = torchaudio.functional.compute_deltas(data_first)
            data_third = torchaudio.functional.compute_deltas(data_second)
            x = torch.cat([data,data_first, data_second, data_third], dim=1)
           
            # print(y.shape)
            batch_size = y.size(0) 
            total = total + batch_size
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            # print(x.shape)
            y_g_hat = net(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            total_loss   = total_loss + loss_gen_all*batch_size

            print("Coord: "+str(coord)+"\tLoss: "+str(np.around(loss_gen_all,3))+"\r")
            # sys.stdout.flush()
            
            if batch_idx==10: break
            
        if count % 2 == 0 : print("count: "+str(count)+"\tCoord: "+"\tLoss: "+str(np.around(total_loss/total,3))+"\n")
        
    # store value 
    loss_list[col_value][ind%numebr_of_points] = total_loss/total
    # acc_list [col_value][ind%numebr_of_points] = 100.*correct/total
    ind_compare = ind + 1
    if ind_compare % numebr_of_points == 0 :  col_value = col_value + 1


import pickle
with open('Qloss_list.pkl', 'wb') as file:
    pickle.dump(loss_list, file)




# # # Loading the list from a binary file
# # with open('list.pkl', 'rb') as file:
# #     my_list = pickle.load(file)

# # for count, ind in enumerate(inds):

    
# #     # change the weight values
# #     coord   = coordinate[count]

# #     changes = [d0.to(device)*coord[0] + d1.to(device)*coord[1] for (d0, d1) in zip(random_direction1, random_direction2)]

# #     for (p, w, d) in zip(net.parameters(), wweight, changes):
# #         #   print(w.shape, d.shape) 
# #           p.data = w + d


# # #     # start the evaluation
# #     correct = 0
# #     total_loss = 0
# #     total = 0 
    
# #     with torch.no_grad():
# #         for batch_idx, batch in enumerate(train_loader):
# #             x, y, _, y_mel = batch
# #             print(y.shape)
# #             batch_size = y.size(0) 
# #             total = total + batch_size
# #             x = torch.autograd.Variable(x.to(device, non_blocking=True))
# #             y = torch.autograd.Variable(y.to(device, non_blocking=True))
# #             y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
# #             y = y.unsqueeze(1)
# #             print(x.shape)
# #             y_g_hat = net(x)
# #             y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
# #                                           h.fmin, h.fmax_for_loss)
# #             # MPD
# #             y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
# #             loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

# #             # MSD
# #             y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
# #             loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

# #             loss_disc_all = loss_disc_s + loss_disc_f

# #             loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

# #             y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
# #             y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
# #             loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
# #             loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
# #             loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
# #             loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
# #             loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

# #             total_loss   = total_loss + loss_gen_all*batch_size

# #             # print(total_loss)
# #             print("Coord: "+str(coord)+"\tLoss: "+str(np.around(loss_gen_all,3))+"\r")
           
            
# #             if batch_idx==10: 
# #                 break
            
# #         if count % 2 == 0 : 
# #             print("count: "+str(count)+"\tCoord: "+str(coord)+"\tLoss: "+str(np.around(total_loss/total,3))+"\n")
        
# #     # store value 
# #     loss_list[col_value][ind%numebr_of_points] = total_loss/total
# #     # acc_list [col_value][ind%numebr_of_points] = 100.*correct/total
# #     ind_compare = ind + 1
# #     if ind_compare % numebr_of_points == 0 :  
# #         col_value = col_value + 1

