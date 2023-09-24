echo "-------------- Downloading best checkpoint --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/17EYwZsjD4g4orzy-LVgZLel6KSs2u-Il/view?usp=sharing" \
--output_dir "speech_modules/checkpoint/"

echo "-------------- Unzipping checkpoint --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/checkpoint/wav2vec2-2109.zip" \
--output_dir "speech_modules/checkpoint/"