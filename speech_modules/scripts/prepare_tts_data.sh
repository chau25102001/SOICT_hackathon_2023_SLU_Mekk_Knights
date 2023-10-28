echo "-------------- Downloading ESC 50 Noise dataset --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/1K90F2E3DOlz_aPmmmfANhpu0DS7pSJY1/view?usp=sharing" \
--output_dir "speech_modules/data/"


# python speech_modules/download_file.py  \
# --download_url "https://drive.google.com/file/d/1sbFPxvANPkDJ-VSDSL-4FnDqGb9B5w5B/view?usp=sharing" \
# --output_dir "speech_modules/data/"

# echo "-------------- Unzipping ESC 50 Noise dataset --------------"
python speech_modules/unrar_data.py \
--rar_file_path "speech_modules/data/augmented_speech.tar.gz" \
--output_dir "speech_modules/data/"

# python speech_modules/unrar_data.py \
# --rar_file_path "speech_modules/data/4492_augmented_speech.tar.gz" \
# --output_dir "speech_modules/data/"

python speech_modules/prepare_tts_data.py \
