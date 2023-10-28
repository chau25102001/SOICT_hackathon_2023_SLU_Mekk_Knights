from datasets import load_dataset
import json

pred_old = load_dataset("json", data_files="predictions.jsonl", split='train')
pred_new = load_dataset("json", data_files="/home/chau/predictions.jsonl", split='train')
transcript = load_dataset("json",
                          data_files="/home/chau/PycharmProjects/SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction/correction.jsonl",
                          split='train')
pred_old = pred_old.sort("file")
pred_new = pred_new.sort("file")
transcript = transcript.sort("id")
with open("pred_diff.jsonl", 'w') as f:
    for i in range(len(pred_new)):
        sample_new = pred_new[i]
        sample_old = pred_old[i]
        if sample_new['intent'] != sample_old['intent']:
            write_sample = {"id": transcript[i]['id'],
                            "transcript": transcript[i]['sentence'],
                            "pred_old": sample_old['intent'],
                            "pred_new": sample_new['intent']}
            f.write(json.dumps(write_sample, ensure_ascii=False))
            f.write("\n")
