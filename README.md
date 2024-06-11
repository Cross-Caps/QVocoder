<h1 align="center">
<p>QGAN :speaker: :notes:</p>
<p align="center">
<img alt="GitHub" src="https://img.shields.io/github/license/cross-caps/AFLI?color=green&logo=GNU&logoColor=green">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="pytorch" src="https://img.shields.io/badge/pytorch-%3E%3D1.8-orange?logo=pytorch">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>QGAN: Low Footprint Quaternion Neural Vocoder for Speech Synthesis</p>
</h2> 

In this Interspeech-24 [paper](), 
we proposed QGAN: a Quaternion GAN-based model capable of generating high fidelity speech efficiently. We provide our open-source implementation and pretrained models in this repository.<br/>

**Demo:**
Visit our [demo website](https://anonymousvocoders.github.io) for audio samples.


## Pre-requisites
1. Python >= 3.8
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


## Generate Loss Landscapes
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

## Contact
Aryan Chaudhary: aryanc55@gmail.com

## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

