import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import wave
from pydub import AudioSegment
import json

def main():
    print('-------- Generating augmented audio files from the training set --------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('--num_concat', type=int,  help="Number of resulting concatenation files", default = 25000)
    # parser.add_argument('--input_dir', type=str, help="input directory containing wav files", default = 'speech_modules/data/original_data/Train/')
    parser.add_argument('--output_dir', type=str,  help="output directory containing concatenated files", default = 'speech_modules/data/Train_concatenated_tts_final/')
    parser.add_argument( '--seed', type=int, help="Random seed", default = 42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_concat = args.num_concat
    output_dir = args.output_dir

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

    with open('speech_modules/data/Train_tts/stt_annotations.jsonl', 'r') as json_file:
        tts_json_list = list(json_file)

    for json_str in tts_json_list:
        result = json.loads(json_str)
        annotation_data[result['id']] = result['sentence']


    input_dirs = ['speech_modules/data/original_data/Train/', 'speech_modules/data/Train_tts/audio',
                  'speech_modules/data/Train_tts_augment/light/audio','speech_modules/data/Train_tts_augment/moderate/audio',
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
    
    num_augmented = 0
    with open(output_dir + 'train_concat.jsonl','w+',encoding='utf-8') as f:

        for i in tqdm(range(len(all_audio_files))):
            try:
                current_file = all_audio_files[i]
                concat_file = random.sample(all_audio_files,1)[0]

                current_file_id = current_file.split('/')[-1].replace('.wav','')
                concat_file_id = concat_file.split('/')[-1].replace('.wav','')

                current_data_type = concat_data_type =  'train'
                if current_file.split('/')[-2] in ['Train_denoise','heavy','moderate','light']:
                    current_data_type = current_file.split('/')[-2].lower()

                if concat_file.split('/')[-2] in ['Train_denoise','heavy','moderate','light']:
                    concat_data_type = concat_file.split('/')[-2].lower()

                sound1 = AudioSegment.from_wav(current_file)
                sound2 = AudioSegment.from_wav(concat_file)

                combined_audio = sound1 + sound2
                combined_file_id = current_file_id + '_' + current_data_type + '_' + concat_file_id + '_' + concat_data_type
                combined_file_name = os.path.join(audio_output_dir,combined_file_id + '.wav')
                combined_audio.export(combined_file_name, format="wav")

                current_sentence = annotation_data[current_file_id]
                concat_sentence = annotation_data[concat_file_id]
                sentence = current_sentence + ' ' + concat_sentence
                entry = {'id':combined_file_id,'sentence':sentence}
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
            except:
                print(1)

    

if __name__ == "__main__":
    main()