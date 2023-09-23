echo "-------------- Downloading original dataset --------------"
python speech_modules/download_data_folder.py  \
--download_url "https://drive.google.com/drive/folders/1FqCmmSjMMgkYjANXY7FD6tzqsfDwZJrY?usp=drive_link" \
--output_dir "speech_modules/data/original_data/"

echo "-------------- Unzipping training set --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/data/original_data/SLU/train_data.zip" \
--output_dir "speech_modules/data/original_data"

echo "-------------- Unzipping public test set --------------"
python speech_modules/unzip_data.py \
--zip_file_path "speech_modules/data/original_data/SLU/public_test.zip" \
--output_dir "speech_modules/data/original_data"

