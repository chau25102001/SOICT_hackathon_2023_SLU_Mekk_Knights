echo "-------------- Downloading ESC 50 Noise dataset --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/1hEr0iJ6ZJXFmHNxF3frC0nPWsAxYlZmR/view?usp=sharing" \
--output_dir "speech_modules/data/"

echo "-------------- Unzipping ESC 50 Noise dataset --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/data/ESC-50-master.zip" \
--output_dir "speech_modules/data/"
