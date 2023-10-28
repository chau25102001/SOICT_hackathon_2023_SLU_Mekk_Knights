import os, shutil

tts_dir = "speech_modules/data/data_tts/"
new_tts_dir = "speech_modules/data/Train_tts/"
try:
    path = "speech_modules/data/stt_annotations.jsonl"
    moveto = os.path.join(new_tts_dir,"stt_annotations.jsonl")
    shutil.copyfile(path, moveto)
except:
    pass
new_audio_dir = os.path.join(new_tts_dir,'audio')
if not os.path.exists(new_audio_dir):
    os.makedirs(new_audio_dir)
all_sub_folders = [dir for dir in os.listdir(tts_dir) if '.jsonl' not in dir]
for folder in all_sub_folders:
    folder_dir = os.path.join(tts_dir,folder)
    audio_files_in_folder = os.listdir(folder_dir)
    for audio_file in audio_files_in_folder:
        audio_path = os.path.join(folder_dir,audio_file)
        new_audio_path = os.path.join( new_audio_dir,audio_file)
        shutil.copyfile(audio_path, new_audio_path)

        # print(audio_path)
    
    # print(audio_files_in_folder)