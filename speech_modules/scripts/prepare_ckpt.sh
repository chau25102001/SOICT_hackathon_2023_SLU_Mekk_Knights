echo "-------------- Downloading best checkpoint --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/1fRXPX6dVooYOIvWCH0CVnbCkvkhg8QDw/view?usp=sharing" \
--output_dir "speech_modules/checkpoint/"

echo "-------------- Unzipping checkpoint --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/checkpoint/ensemble_2710.zip" \
--output_dir "speech_modules/checkpoint/"