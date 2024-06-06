# QGAN: Low Footprint Quaternion Neural Vocoder for Speech Synthesis

### Aryan Chaudhary, Vinayak Abrol

In our [paper](), 
we proposed QGAN: a Quaternion GAN-based model capable of generating high fidelity speech efficiently.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
Neural vocoders have recently evolved to achieve superior syn- thesis quality by leveraging advancements in methods like dif- fusion, flow, transformers, GANs, etc. However, such mod- els have grown vastly in terms of space and time complex- ity, leading to challenges in the deployment of speech synthe- sis systems in resource-constraint scenarios. To address this, we present a novel low-footprint Quaternion Generative Adver- sarial Network (QGAN) for efficient and high-fidelity speech synthesis without compromising on the audio quality. QGAN achieves structural model compression over conventional GAN with quaternion convolutions in the generator and a modified multi-scale/period discriminator. To ensure model stability, we also propose weight-normalization in the quaternion domain. We show the effectiveness of QGAN with large-scale experi- ments on English and Hindi language datasets. In addition, us- ing loss landscape visualization, we provide an analysis of the learning behaviour of the proposed QGAN model.
Visit our [demo website](https://anonymousvocoders.github.io) for audio samples.


## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
5. Downlaod and extract the [Hindi dataset](https://openslr.org/118/)
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config config_v1.json
```
To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.


## Pretrained Model
You can also use pretrained models we provide.<br/>
[Download pretrained models]()<br/> 
<!-- 
## Fine-Tuning
1. Generate mel-spectrograms in numpy format using [Tacotron2](https://github.com/NVIDIA/tacotron2) with teacher-forcing.<br/>
The file name of the generated mel-spectrogram should match the audio file and the extension should be `.npy`.<br/>
Example:
    ```
    Audio File : LJ001-0001.wav
    Mel-Spectrogram File : LJ001-0001.npy
    ```
2. Create `ft_dataset` folder and copy the generated mel-spectrogram files into it.<br/>
3. Run the following command.
    ```
    python train.py --fine_tuning True --config config_v1.json
    ```
    For other command line options, please refer to the training section. -->


## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
    ```
    python inference.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Generate losslands
1. Set checkpoint path in the lossladns.py file and load the models and their wirgths accordingly.
2. Losslands code will dump the loss_list - a list of vlaues used for generating visualization 
    ```
    python losslands.py
    ```

<!-- ## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
    ```
    python inference_e2e.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option. -->


## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

