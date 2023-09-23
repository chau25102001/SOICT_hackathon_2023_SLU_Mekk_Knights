cd data
python3 prepare_text_correction_data.py
cd ..
CUDA_VISIBLE_DEVICES=0 python3 train.py