echo "-------------- Downloading best checkpoint --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/1PRG7dQb3rn2rQ9cm4BV27l5zf2e6xOMX/view?usp=drive_link" \
--output_dir "speech_modules/checkpoint/"

echo "-------------- Unzipping checkpoint --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/checkpoint/wav2vec2-2109.zip" \
--output_dir "speech_modules/checkpoint/"
