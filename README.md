# SOICT HACKATHON 2023 - SLU - Mekk-Knights
## Phase 1 - Speech Solutions
### Training procedure

Navigate to the project folder ```cd /SOICT_hackathon_2023_SLU_Mekk_Knights```

Run the following bash files in respective order to prepare for training:
- ```bash ./speech_modules/scripts/prepare_data.sh``` : Download the original training dataset, unzip the training and public audio files (currently saved at *speech_modules/data/original_data*)
- ```bash ./speech_modules/scripts/prepare_noise_data.sh```: Download and unzip the ESC-50 environmental noise dataset [https://github.com/karolpiczak/ESC-50] in order to augment the training audio files
- ```bash ./speech_modules/scripts/preprocess.sh```: Preprocess our training dataset
    + Normalize text utterances: lowercase, remove punctuations, convert numbers and special symbols to spoken words ("28%" to "hai mươi tám phần trăm")
    + Data augmentation: Apply various data augmentation methods on the training audio files (SpecAugment, changing pitch, changing speed, adding white noise, adding ESC 50 noise...) to obtain an augmented dataset, stored at *speech_modules/data/Train_augment/*)
    + Generate denoised training audio files to remove noises. These denoised audio files are stored at *speech_modules/data/Train_denoise*.
- ```bash ./speech_modules/scripts/train.sh```: Train a wav2vec2 model using the original training data, augmented data, and denoised data. The best checkpoint will be saved at "speech_modules/checkpoint/wav2vec2-ckpt"

### Inference Procedure
- ```bash ./speech_modules/scripts/prepare_ckpt.sh```: Download our trained checkpoint and unzip it at *speech_modules/checkpoint/wav2vec2-2109*.
- ```bash ./speech_modules/scripts/infer.sh```: Generating transcripts of the public test sets using the saved best checkpoint (at *speech_modules/checkpoint/wav2vec2-2109*). This will save the inference results in a file named ```/SOICT_hackathon_2023_SLU_Mekk_Knights/stt_pred.jsonl```


## Phase 1.5 - Text correction model

We use a BartPho model to correct the text predicted from ASR

**TRAINING**
* navigate to the folder```cd /SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction```
* run ```bash train.sh```

This will generate corrupted text dataset from the train transcript file and train a bartpho model for text correction. The trained model will be saved in ```./log/bartpho_text_correction```.
You can change the train config in ```/SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction/configs/config.py```
**INFER TEXT CORRECTION FROM ASR MODEL**
* navigate to the folder```cd /SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction```

* run 

```
python correct_prediction.py --stt_pred_path [path-to-asr-prediction-jsonl] --model_checkpoint [path-to-bartpho-checkpoint-pt]
```

This will output a ```correction.jsonl``` file in ```/SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction```.
```model_checkpoint``` default is set to ```checkpoint/checkpoint_best.pt``` which is our best submission, you do not have to parse this argument.


## Phase 2 - Intent detection and slot filling model

We use a JointIDSF model with PhoBert backbone for intent detection and slot filling 

**TRAINING**

* navigate to the folder ```cd /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF```
* run ```bash train.sh```

This will generate an augmented dataset from the train transcript file and train a intent detection and slot filling model.
You can change the train config in ```/SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/configs/bio_config.py```
The trained model will be saved in ```./log/phobert_bio```. 

**INFER**
* navigate to the folder ```cd /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF```
* run 
```
python infer_bio.py --stt_pred_path [path-to-asr-prediction-jsonl] --model_checkpoint [path-to-bartpho-checkpoint-pt]
```
this will create a ```predictions.jsonl``` file in ```/SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF```, which is the inference result from the ```[path-to-asr-prediction-jsonl]``` file. 
```model_checkpoint``` default is set to ```checkpoint/checkpoint_best.pt``` which is our best submission, you do not have to parse this argument.



