python speech_modules/normalize_text.py

echo "-------------- Generating denoised training file --------------"
CUDA_VISIBLE_DEVICES=0 python speech_modules/denoise.py

echo "-------------- Augmenting data. Mode: Heavy --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--output_dir "speech_modules/data/Train_augment/heavy" \
--config_path "speech_modules/data_augmentation_for_asr/config/heavy.json" \
--seed 42

echo "-------------- Augmenting data. Mode: Moderate --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--output_dir "speech_modules/data/Train_augment/moderate/" \
--config_path "speech_modules/data_augmentation_for_asr/config/moderate.json" \
--seed 43

echo "-------------- Augmenting data. Mode: Light --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--output_dir "speech_modules/data/Train_augment/light/" \
--config_path "speech_modules/data_augmentation_for_asr/config/light.json" \
--seed 44

echo "-------------- Augmenting TTS data. Mode: Heavy --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--input_dir "speech_modules/data/Train_tts_final/audio/" \
--output_dir "speech_modules/data/Train_tts_augment_final/heavy/audio" \
--config_path "speech_modules/data_augmentation_for_asr/config/heavy.json" \ 
--seed 42

echo "-------------- Augmenting TTS data. Mode: Moderate --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--input_dir "speech_modules/data/Train_tts/audio/" \
--output_dir "speech_modules/data/Train_tts_augment/moderate/audio" \
--config_path "speech_modules/data_augmentation_for_asr/config/moderate.json" \
--seed 43

echo "-------------- Augmenting TTS data. Mode: Light --------------"
python speech_modules/data_augmentation_for_asr/data_augmentation.py \
--input_dir "speech_modules/data/Train_tts/audio/" \
--output_dir "speech_modules/data/Train_tts/light/audio" \
--config_path "speech_modules/data_augmentation_for_asr/config/light.json" \
--seed 44

echo "-------------- Generating concatenated data from the available files (not including TTS)--------------"
python speech_modules/data_augmentation_for_asr/concat_augmentation.py 

echo "-------------- Generating concatenated data from the available files (including TTS)--------------"
python speech_modules/data_augmentation_for_asr/concat_augmentation_tts.py 