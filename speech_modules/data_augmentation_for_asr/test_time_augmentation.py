# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Class that aggregates the functions of the SpecAugmentation and AudioAugmentation classes.
#
#
# (C) 2021 Frederico Oliveira, UFMT/UFG
# Released under GNU Public License (GPL)
# email fred.santos.oliveira@gmail.com
#
import argparse
import numpy as np
import random
from librosa.core import load
from utils.config import load_config
from spec_augmentation import SpecAugmentation
from audio_augmentation import AudioAugmentation
from scipy.io.wavfile import write
from datetime import datetime
import os
from tqdm import tqdm
import librosa
# Setting seed.

# Constants
sampling_rate = 16000

class DataAugmentation():
    '''
    Class that performs data augmentation on spectrogram and waveform.
    '''

    def __init__(self,  sa: SpecAugmentation,
                        aa: AudioAugmentation,
                        sr,
                        noises_filepath,
                        noise_range_min,
                        noise_range_max,
                        pitch_range_min,
                        pitch_range_max,
                        stretch_range_min,
                        stretch_range_max,
                        external_noise_range_min,
                        external_noise_range_max,
                        shift_roll_range_min,
                        shift_roll_range_max,
                        num_freq_mask,
                        freq_masking_min_percentage,
                        freq_masking_max_percentage,
                        num_time_mask,
                        time_masking_min_percentage,
                        time_masking_max_percentage
    ):
        '''

        Args:
            sa: SpecAugmentation Object.
            aa: AudioAugmentation Object.
            sr: sampling rate used.
            noises_filepath: filepath of file containing list of audio noise files.
            noise_range_min:
            noise_range_max:
            pitch_range_min:
            pitch_range_max:
            stretch_range_min:
            stretch_range_max:
            external_noise_range_min:
            external_noise_range_max:
            shift_roll_range_min:
            shift_roll_range_max:
            num_freq_mask:
            freq_masking_min_percentage:
            freq_masking_max_percentage:
            num_time_mask:
            time_masking_min_percentage:
            time_masking_max_percentage:
        '''
        self.sa = sa
        self.aa = aa
        self.sr = sr
        self.noise_range_min = noise_range_min
        self.noise_range_max = noise_range_max
        self.pitch_range_min = pitch_range_min
        self.pitch_range_max = pitch_range_max
        self.stretch_range_min = stretch_range_min
        self.stretch_range_max = stretch_range_max
        self.external_noise_range_min = external_noise_range_min
        self.external_noise_range_max = external_noise_range_max
        self.shift_roll_range_min = shift_roll_range_min
        self.shift_roll_range_max = shift_roll_range_max
        self.num_freq_mask = num_freq_mask
        self.freq_masking_min_percentage = freq_masking_min_percentage
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.num_time_mask = num_time_mask
        self.time_masking_min_percentage = time_masking_min_percentage
        self.time_masking_max_percentage = time_masking_max_percentage
        self.noises_list = self.get_noises_list(noises_filepath)

    def get_noises_list(self, noise_filepath):
        '''
        Loads the list of audio noise files.
        Args:
            noise_filepath: filepath to the file.

        Returns:
            noises_list: list of noise audio files.

        '''
        f = open(noise_filepath, "r")
        noises_list = f.readlines()
        f.close()        
        return noises_list

    def read_audio(self, filepath):
        '''
        Load the clean audio file.
        Args:
            filepath: filepath to the audio file.

        Returns:
            data: numpy audio data time series.
        '''
        data, _ = load(filepath, self.sr) # Librosa function
        return data

    def write_audio(self, filepath, data):
        '''
        Saves the audio data to a file.
        Args:
            filepath: path to save the file
            data: audio data.

        '''
        data = (data * 32767).astype(np.int16) # Convert to int16
        write(filepath, self.sr, data) # Librosa function

    def insert_white_noise(self, filepath):
        '''
        Use the add_noise function from AudioAugmentation to insert white noise.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with white noise inserted.
        '''
        data = self.read_audio(filepath)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(data, noise_rate)
        return aug_data

    def shift_audio(self, filepath):
        '''
        Performs shitf audio data augmentation and adds white noise.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data shifted with white noise inserted.
        '''
        data = self.read_audio(filepath)
        # Shifting the sound
        shift_rate = random.uniform(self.shift_roll_range_min, self.shift_roll_range_max)
        aug_data = self.aa.shift(data, shift_rate)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(aug_data, noise_rate / 2)
        return aug_data

    def stretching_audio(self, filepath):
        '''
        Performs time stretch and adds some white noise.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data time stretched with white noise inserted.
        '''
        data = self.read_audio(filepath)
        # Stretching the sound
        stretch_rate = random.uniform(self.stretch_range_min, self.stretch_range_max)
        aug_data = self.aa.stretch(data, stretch_rate)
        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(aug_data, noise_rate / 10)
        return aug_data

    def changing_pitch(self, filepath):
        '''
        Performs pitch-shift and adds some white noise.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data pitch-shifted with white noise inserted.
        '''
        data = self.read_audio(filepath)
        # Changing pitch
        pitch_rate = random.uniform(self.pitch_range_min, self.pitch_range_max)
        aug_data = self.aa.pitch(data, pitch_rate)
        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(aug_data, noise_rate / 10)
        return aug_data

    def insert_external_noise(self, filepath):
        '''
        Performs external noise insertion and adds some white noise.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with random external noise and white noise inserted.
        '''
        data = self.read_audio(filepath)
        # randomly selects an audio to be inserted.
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        # Adding external noise to clean audio
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max)
        aug_data = self.aa.add_external_noise(data, noise, ex_noise_rate)
        # Adding noise to clean audio
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(aug_data, noise_rate / 10)
        return aug_data

    def insert_spectrogram_noise(self, filepath):
        '''
        Performs data augmentation masking on the time-frequency axis.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with data augmentation in the frequency domain.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on frequency axis.
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time axis.
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)
        return aug_data

    def insert_mix_noise(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on spectogram domain
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time domain
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)

        # Changing pitch
        pitch_rate = random.uniform(self.pitch_range_min, self.pitch_range_max)
        aug_data = self.aa.pitch(aug_data, pitch_rate)

        # Stretching the sound
        stretch_rate = random.uniform(self.stretch_range_min, self.stretch_range_max)
        daug_data = self.aa.stretch(aug_data, stretch_rate)

        # Shifting the sound
        shift_rate = random.uniform(self.shift_roll_range_min, self.shift_roll_range_max)
        daug_data = self.aa.shift(aug_data, shift_rate)

        # Adding noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
        aug_data = self.aa.add_noise(aug_data, noise_rate)

        # inserting asr-noises
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max)
        aug_data = self.aa.add_external_noise(aug_data, noise, ex_noise_rate)

        return aug_data


    def insert_lite_mix_noise(self, filepath):
        '''
        Performs several data augmentation functions on the same audio in a lite way. The number of functions is less than the insert_mix_noise function and the intensity of the noise (white and external) is attenuated.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        # Apply data augmentation on spectogram domain
        freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
        spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # Apply data augmentation on time domain
        time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
        spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        aug_data= self.sa.inv_spectrogram(spec_aug, phase)

        # Adding white noise to sound
        noise_rate = random.uniform(self.noise_range_min, self.noise_range_max) / 10 # noise_rate is divided by 10 in this lite version.
        aug_data = self.aa.add_noise(aug_data, noise_rate)

        # inserting external noises
        noise_filepath = random.choice(self.noises_list).strip()
        noise = self.aa.read_audio_file(noise_filepath)
        ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max) / 10 # noise_rate is divided by 10 in this lite version.
        aug_data = self.aa.add_external_noise(aug_data, noise, ex_noise_rate)

        return aug_data


    def augment_pipeline(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)

        spec_aug = spec
        # Apply data augmentation on spectrogram domain
        if random.uniform(0,1) < 0.3:
            freq_percentage = random.uniform(self.freq_masking_min_percentage, self.freq_masking_max_percentage)
            spec_aug = self.sa.freq_spec_augment(spec, self.num_freq_mask, freq_percentage)

        # print(random.uniform(0,1))

        if random.uniform(0,1) < 0.3:
            # Apply data augmentation on time domain
            time_percentage = random.uniform(self.time_masking_min_percentage, self.time_masking_max_percentage)   
            spec_aug = self.sa.time_spec_augment(spec_aug, self.num_time_mask, time_percentage)
        # Get wav
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)
        # aug_data = librosa.load(filepath, sr=22500)[0]
        # print(type(aug_data))
        # aug_data = data

        # print(random.uniform(0,1))

        # Changing pitch
        if random.uniform(0,1) < 0.4:
            pitch_rate = random.uniform(self.pitch_range_min, self.pitch_range_max)
            aug_data = self.aa.pitch(aug_data, pitch_rate)

        if random.uniform(0,1) < 0.4:
            # Stretching the sound
            stretch_rate = random.uniform(self.stretch_range_min, self.stretch_range_max)
            aug_data = self.aa.stretch(aug_data, stretch_rate)

        if random.uniform(0,1) < 0.3:

            # Shifting the sound
            shift_rate = random.uniform(self.shift_roll_range_min, self.shift_roll_range_max)
            aug_data = self.aa.shift(aug_data, shift_rate)

        if random.uniform(0,1) < 0.4:
            # Adding noise to sound
            noise_rate = random.uniform(self.noise_range_min, self.noise_range_max)
            aug_data = self.aa.add_noise(aug_data, noise_rate)

        # inserting asr-noises
        if random.uniform(0,1) < 0.4:

            noise_filepath = random.choice(self.noises_list).strip()
            noise = self.aa.read_audio_file(noise_filepath)
            ex_noise_rate = random.uniform(self.external_noise_range_min, self.external_noise_range_max)
            aug_data = self.aa.add_external_noise(aug_data, noise, ex_noise_rate)

        return aug_data

    def uppitch(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)
        spec_aug = spec
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)

        pitch_rate = random.uniform(0, self.pitch_range_max)
        aug_data = self.aa.pitch(aug_data, pitch_rate)
        return aug_data

    def downpitch(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)
        spec_aug = spec
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)

        pitch_rate = random.uniform( self.pitch_range_min, 0)
        aug_data = self.aa.pitch(aug_data, pitch_rate)
        return aug_data

    def speedup(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)
        spec_aug = spec
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)

        stretch_rate = random.uniform(0, self.stretch_range_max)
        aug_data = self.aa.stretch(aug_data, stretch_rate)
        return aug_data

    def speeddown(self, filepath):
        '''
        Performs several data augmentation functions on the same audio.
        Args:
            filepath: path to read clean audio file.

        Returns:
            aug_data: audio data with several data augmentation functions.
        '''
        # Read wavefile
        spec, phase = self.sa.get_spectrogram_phase(filepath)
        spec_aug = spec
        aug_data = self.sa.inv_spectrogram(spec_aug, phase)

        stretch_rate = random.uniform(0, self.stretch_range_max)
        aug_data = self.aa.stretch(aug_data, stretch_rate)
        return aug_data

def main():
    print('-------- Generating augmented audio files from the training set --------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('--input_dir', type=str, help="input directory containing wav files", default = 'speech_modules/data/original_data/Train/')
    parser.add_argument('--output_dir', type=str,  help="output directory containing augmented files", default = 'speech_modules/data/test_time_augmentation/Train/')
    parser.add_argument( '--config_path', type=str, help="json file with configurations", default = 'speech_modules/data_augmentation_for_asr/config/light.json')
    parser.add_argument( '--seed', type=int, help="Random seed", default = 42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config_path)  

    sa = SpecAugmentation(config.audio) 
    aa = AudioAugmentation(sampling_rate)
    da = DataAugmentation(sa, aa, sampling_rate, **config.aug_data)

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    uppitch_dir = os.path.join(output_dir,'uppitch/')
    downpitch_dir = os.path.join(output_dir,'downpitch/')
    speedup_dir = os.path.join(output_dir,'speedup/')
    speeddown_dir = os.path.join(output_dir,'speeddown/')

    if not os.path.exists(uppitch_dir):
        os.makedirs(uppitch_dir)

    if not os.path.exists(downpitch_dir):
        os.makedirs(downpitch_dir)

    if not os.path.exists(speedup_dir):
        os.makedirs(speedup_dir)

    if not os.path.exists(speeddown_dir):
        os.makedirs(speeddown_dir)

    all_audio_files = os.listdir(input_dir)
    all_audio_files = [file_name for file_name in all_audio_files if '.wav' in file_name]


    for file_name in tqdm(all_audio_files):
        file_path = os.path.join(input_dir,file_name)
        # print(file_path)
        # aug_data = da.augment_pipeline(file_path)
        # uppitch_data = da.uppitch(file_path) 
        # downpitch_data = da.downpitch(file_path)
        # speedup_data = da.speedup(file_path)
        speeddown_data = da.speeddown(file_path)

        # print(aug_data)
        # da.write_audio(os.path.join(uppitch_dir,file_name), uppitch_data)
        # da.write_audio(os.path.join(downpitch_dir,file_name), downpitch_data)
        # da.write_audio(os.path.join(speedup_dir,file_name), speedup_data)
        da.write_audio(os.path.join(speeddown_dir,file_name), speeddown_data)

if __name__ == "__main__":
    main()