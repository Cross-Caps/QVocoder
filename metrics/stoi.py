# conda activate tmetrics
import torch
import os
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
g = torch.manual_seed(1)
import torchaudio
import statistics



source_real = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/generated_dir"

source_qnn = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/generated_dir"

target = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/test_dir"

source_real_ = []

source_qnn_ = []

target_ = []

for file_name in os.listdir(target):
        
        file_path = os.path.join(target, file_name)
        target_.append(file_path)



for file_name in os.listdir(source_real):
        
        file_path = os.path.join(source_real, file_name)
        source_real_.append(file_path)



for file_name in os.listdir(source_qnn):
        
        file_path = os.path.join(source_qnn, file_name)
        source_qnn_.append(file_path)



# print(target_,source_real_, source_qnn_)

stoi_ = []
for aud1, aud2 in zip(sorted(target_),sorted(source_real_)):
        # print(aud1, aud2)
        
        target,sr = torchaudio.load(aud1)
        preds,sr = torchaudio.load(aud2)

        length_diff = abs(target.shape[1] - preds.shape[1])


        if target.shape[1] < preds.shape[1]:
   
                target= torch.nn.functional.pad(target, (0, length_diff))
        elif target.shape[1] > preds.shape[1]:
    
                preds = torch.nn.functional.pad(preds, (0, length_diff))


        


        stoi = ShortTimeObjectiveIntelligibility(sr, False)
        score = stoi(preds, target)
        stoi_.append(score)


# print("Real STOI:",stoi_)

# # Example list of numbers
# numbers = stoi_

# # Calculate mean
# mean_value = statistics.mean(numbers)

# # Calculate variance
# variance_value = statistics.variance(numbers)
import numpy as np

print("Real STOI:",np.mean(stoi_))

Qstoi_ = []
for aud1, aud2 in zip(sorted(target_),sorted(source_qnn_)):
        # print(aud1, aud2)
        
        target,sr = torchaudio.load(aud1)
        
        preds,sr = torchaudio.load(aud2)

        length_diff = abs(target.shape[1] - preds.shape[1])


        if target.shape[1] < preds.shape[1]:
   
                target= torch.nn.functional.pad(target, (0, length_diff))
        elif target.shape[1] > preds.shape[1]:
    
                preds = torch.nn.functional.pad(preds, (0, length_diff))


        
        stoi = ShortTimeObjectiveIntelligibility(sr, False)
        score = stoi(preds, target)
        Qstoi_.append(score)


print("QNN STOI:",np.mean(Qstoi_))


# numbers = Qstoi_

# # Calculate mean
# mean_value = statistics.mean(numbers)

# # Calculate variance
# variance_value = statistics.variance(numbers)


# print("Real STOI:",mean_value, variance_value)