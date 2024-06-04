import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldatasetQ import MelDataset, mel_spectrogram, get_dataset_filelist
from modelsQ5B import QGenerator, QMultiPeriodDiscriminator, QMultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import torchaudio

torch.backends.cudnn.benchmark = True


def train(a, h):
   
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:2')
    print(device)
    
    generator = QGenerator(h).to(device)
    mpd = QMultiPeriodDiscriminator().to(device)
    msd = QMultiScaleDiscriminator().to(device)
    print(generator)
    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']


 

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
   
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_files=training_filelist, segment_size=h.segment_size,n_fft= h.n_fft,num_mels= h.num_mels,
                         hop_size= h.hop_size,win_size= h.win_size,sampling_rate= h.sampling_rate,fmin= h.fmin,fmax= h.fmax, n_cache_reuse=0,
                          shuffle= True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
    
   

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
 

    
    validset = MelDataset(training_files=validation_filelist, segment_size=h.segment_size, n_fft=h.n_fft,num_mels= h.num_mels,
                            hop_size=  h.hop_size,win_size= h.win_size,sampling_rate= h.sampling_rate,fmin= h.fmin,fmax= h.fmax,split= False, shuffle=False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
   
    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch+1))
        for i, batch in enumerate(train_loader):
            start_b = time.time()
            data, y, _, y_mel = batch  # mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()
            #Quaterionic conversion
        
            data_first = torchaudio.functional.compute_deltas(data)
            data_second = torchaudio.functional.compute_deltas(data_first)
            data_third = torchaudio.functional.compute_deltas(data_second)
            x = torch.cat([data,data_first, data_second, data_third], dim=1)
           
            ## same script 
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            # print(data.shape)
            # print(x.shape)

            y_g_hat = generator(x)

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)
            
            # y_g_hat_mel_first  = torchaudio.functional.compute_deltas(y_g_hat_mel)
            # y_g_hat_mel_second  = torchaudio.functional.compute_deltas(y_g_hat_mel_first)
            # y_g_hat_mel_third = torchaudio.functional.compute_deltas(y_g_hat_mel_second)
            # y_g_hat_mel_quaternion = torch.cat([y_g_hat_mel,y_g_hat_mel_first, y_g_hat_mel_second, y_g_hat_mel_third], dim=1)


            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

           
                # STDOUT logging
            if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': generator.state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': mpd.state_dict(),
                                     'msd': msd.state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
            if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
            if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            data, y, _, y_mel = batch
                            data_first = torchaudio.functional.compute_deltas(data)
                            data_second = torchaudio.functional.compute_deltas(data_first)
                            data_third = torchaudio.functional.compute_deltas(data_second)
                            x = torch.cat([data,data_first, data_second, data_third], dim=1)
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan7')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=905, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.batch_size = h.batch_size
        print('Batch size per GPU :', h.batch_size)
        train(a,h)
    else:
        print("Bina GPU ke ghnta train hoga")
       

    # if h.num_gpus > 1:
    #     print("hehe2")
    #     mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    # else:
    #     train(0, a, h)


if __name__ == '__main__':
    main()
