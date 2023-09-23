import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
import os 
import logging
import sys


if __name__ == '__main__':
    print('------------------Generating denoised files of train data--------------------------------')

    device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    model = pretrained.dns64().to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="input directory containing wav files", default = 'speech_modules/data/original_data/Train/')
    parser.add_argument('--output_dir', type=str,  help="output directory containing denoised files", default = 'speech_modules/data/Train_denoise/')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_names = [input_dir + f for f in listdir(input_dir) if isfile(join(input_dir, f))]

    i = 0
    for file_name in tqdm(file_names):
        name = file_name.split('/')[-1]
        # print(name)
        wav, sr = torchaudio.load(file_name)
        wav = convert_audio(wav.to(device), sr, model.sample_rate, model.chin)

        wav = torchaudio.transforms.Vol(gain=3, gain_type="amplitude")(wav)
        with torch.no_grad():
            denoised = model(wav[None])[0]
        path = f"{output_dir}/{name}"
        torchaudio.save(path, denoised.data.cpu(), 16000)