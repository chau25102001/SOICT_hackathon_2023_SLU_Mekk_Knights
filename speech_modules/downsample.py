import librosa  
import os
import soundfile as sf
from tqdm import tqdm
data_dir = 'speech_modules/data/original_data/private_test/'
output_dir = 'speech_modules/data/original_data/downsampled_private_test/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
all_file_names = os.listdir(data_dir)

for file_name in tqdm(all_file_names):
    file_path = os.path.join(data_dir,file_name)
    audio, sampling_rate = librosa.load(file_path, sr=16000) # Downsample 44.1kHz to 8kHz
    sf.write(os.path.join(output_dir,file_name), audio, sampling_rate)    
    # break