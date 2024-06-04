# conda activate utm
import utmos
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# or model.calculate_wav(wav, sample_rate)


import os
# import torchaudio
source_real = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/generated_dir"
source_qnn = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/generated_dir"

# target = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/test_dir"

source_real_ = []
source_qnn_ = []

# target_ = []

# for file_name in os.listdir(target):
        
#         file_path = os.path.join(target, file_name)
#         target_.append(file_path)


# import torch
import librosa
for file_name in os.listdir(source_real):
        file_path = os.path.join(source_real, file_name)
        source_real_.append(file_path)

for file_name in os.listdir(source_qnn):
        file_path = os.path.join(source_qnn, file_name)
        source_qnn_.append(file_path)

utmos_ = []
model = utmos.Score()
for aud in source_real_:
        # print(aud1, aud2)
        target,sr = librosa.load(aud)
        # target = torch.rand(1,16000)
        target  = torch.from_numpy(target).to(device)
        print(target)
         # The model will be automatically downloaded and will automatically utilize the GPU if available.
        # utmos_.append(model.calculate_wav_file(target)) # -> Float 
        utmos_.append(model.calculate_wav(target,sr))

        # model.calculate_wav(target, sr)  

import numpy

print(numpy.mean(utmos_))


# Evaluation metrics: Mean cepstral distortion, F0 root mean square error, Bitrate, UTMOS


# import utmos
# model = utmos.Score() # The model will be automatically downloaded and will automatically utilize the GPU if available.
# model.calculate_wav_file('audio_file.wav') # -> Float
# # or model.calculate_wav(wav, sample_rate)