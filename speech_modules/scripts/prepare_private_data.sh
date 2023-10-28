echo "-------------- Downloading private test dataset --------------"
python speech_modules/download_file.py  \
--download_url "https://drive.google.com/file/d/1p6W8GwRd0E0yC0BaWxA5gMB5fRJs11Y-/view?usp=sharing" \
--output_dir "speech_modules/data/original_data/"

echo "-------------- Unzipping private test set set --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/data/original_data/NAVER_SLU_private_test.zip" \
--output_dir "speech_modules/data/original_data"

echo "-------------- Downsampling private test set set to 16k--------------"
python speech_modules/downsample.py
