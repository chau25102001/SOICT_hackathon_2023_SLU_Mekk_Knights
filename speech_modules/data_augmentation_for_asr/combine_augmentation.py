import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import wave
from pydub import AudioSegment
import json
from audio_augmentation import AudioAugmentation

def main():
    print('-------- Generating augmented audio files from the training set --------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('--num_combined', type=int,  help="Number of resulting combined files", default = 25000)
    parser.add_argument('--output_dir', type=str,  help="output directory containing combined files", default = 'speech_modules/data/Train_combined/')
    parser.add_argument( '--seed', type=int, help="Random seed", default = 42)
    parser.add_argument( '--lower_noise_rate', type=float, help="Lower range of noise rate", default = 0.1)
    parser.add_argument( '--upper_noise_rate', type=float, help="upper range of noise rate", default = 0.3)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_combined = args.num_combined
    output_dir = args.output_dir
    upper_noise_rate, lower_noise_rate = args.upper_noise_rate, args.lower_noise_rate

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_output_dir = os.path.join(output_dir,'audio')
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    with open('speech_modules/data/original_data/train_normalized_20230919.jsonl', 'r') as json_file:
        json_list = list(json_file)

    annotation_data = {}
    for json_str in json_list:
        result = json.loads(json_str)
        annotation_data[result['id']] = result['sentence']
    
    input_dirs = ['speech_modules/data/original_data/Train/','speech_modules/data/Train_augment/heavy/',
                  'speech_modules/data/Train_augment/moderate/','speech_modules/data/Train_augment/light/',
                  'speech_modules/data/Train_denoise']    
    
    all_audio_files = []
    for input_dir in input_dirs:
        input_dir_files =  os.listdir(input_dir)
        input_dir_files =  [os.path.join(input_dir,file_name) for file_name in input_dir_files if '.wav' in file_name]
        # print(input_dir_files)
        all_audio_files.extend(input_dir_files)
    
    random.shuffle(all_audio_files)
    # print(len(all_audio_files))
    
    aa = AudioAugmentation(16000)
    num_augmented = 0
    with open(output_dir + 'train_combined.jsonl','w+',encoding='utf-8') as f:

        for i in tqdm(range(len(all_audio_files))):
            current_file = all_audio_files[i]
            combine_file = random.sample(all_audio_files,1)[0]

            current_file_id = current_file.split('/')[-1].replace('.wav','')
            combine_file_id = combine_file.split('/')[-1].replace('.wav','')

            current_data_type = combine_data_type =  'train'
            if current_file.split('/')[-2] in ['Train_denoise','heavy','moderate','light']:
                current_data_type = current_file.split('/')[-2].lower()

            if combine_file.split('/')[-2] in ['Train_denoise','heavy','moderate','light']:
                combine_data_type = combine_file.split('/')[-2].lower()

            # print(current_file.split('/')[-2])
            # print(current_file, current_file_id, current_data_type)
            # print(combine_file, combine_file_id, combine_data_type)

            sound1 = AudioSegment.from_wav(current_file)
            sound2 = AudioSegment.from_wav(combine_file)
            sound2 = AudioSegment.from_wav(combine_file)

            current_audio = aa.read_audio_file(current_file)
            background = aa.read_audio_file(combine_file)
            noise_rate = random.uniform(lower_noise_rate, upper_noise_rate)
            combine_audio = aa.add_external_noise(current_audio, background,rate = noise_rate)

            # combined_audio = sound1 + sound2
            combined_file_id = current_file_id + '_' + current_data_type + '_' + combine_file_id + '_' + combine_data_type
            combined_file_name = os.path.join(audio_output_dir,combined_file_id + '.wav')
            aa.write_audio_file(combined_file_name, combine_audio)

            current_sentence = annotation_data[current_file_id]
            combine_sentence = annotation_data[combine_file_id]
            sentence = current_sentence + ' ' + combine_sentence
            entry = {'id':combined_file_id,'sentence':sentence}
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')





    # print(len(all_audio_files),all_audio_files[0], all_audio_files[10000], all_audio_files[20000])
    

if __name__ == "__main__":
    main()