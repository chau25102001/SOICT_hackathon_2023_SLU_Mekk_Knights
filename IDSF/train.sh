cd data
python3 process_data_bio.py
cd ..
CUDA_VISIBLE_DEVICES=0 python3 train_bio.py