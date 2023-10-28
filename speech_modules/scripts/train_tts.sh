CUDA_VISIBLE_DEVICES=1 python speech_modules/wav2vec2/train_tts.py --batch_size 4 \
                                                                --eval_batch_size 4 \
                                                                --num_epochs 25 \
                                                                --pretrained_model_name "nguyenvulebinh/wav2vec2-large-vi-vlsp2020" \
                                                                --checkpoint_path "speech_modules/checkpoint/wav2vec2-ckpt-tts" \
                                                                --learning_rate 1e-4 \
                                                                --weight_decay 1e-4 \
                                                                --use_augment \
                                                                --use_denoise \
                                                                --use_tts \
                                                                --use_tts_augment \
                                                                --use_augment_concat
                
