# SOICT HACKATHON 2023 - SLU - Mekk-Knights
## Text correction model

We use a BartPho model to correct the text predicted from ASR

**TRAINING**
* navigate to the folder```cd text_correction```
* run ```bash train.sh```

This will generate corrupted text dataset from the train transcript file and train a bartpho model for text correction. The trained model will be saved in ```./log/bartpho_text_correction```

**INFER TEXT CORRECTION FROM ASR MODEL**
* navigate to the folder```cd text_correction```

* run 

```
python correct_prediction.py --stt_pred_path [path-to-asr-prediction-jsonl] --model_checkpoint [path-to-bartpho-checkpoint-pt]
```

This will output a ```correction.jsonl``` file

## Intent detection and slot filling model

We use a JointIDSF model with PhoBert backbone for intent detection and slot filling 

**TRAINING**

* navigate to the folder ```cd IDSF```
* run ```bash train.sh```

This will generate an augmented dataset from the train transcript file and train a intent detection and slot filling model.
The trained model will be saved in ```./log/phobert_bio```

**INFER**
* navigate to the folder ```cd IDSF```
* run 
```
python infer_bio.py --stt_pred_path [path-to-asr-prediction-jsonl] --model_checkpoint [path-to-bartpho-checkpoint-pt]
```
this will create a ```predictions.jsonl``` file, which is the inference result from the ```[path-to-asr-prediction-jsonl]``` file
