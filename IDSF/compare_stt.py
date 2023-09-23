from datasets import load_dataset
import json

dataset_base = load_dataset("json", data_files="data/max255_infer.jsonl", split='train')
dataset_large = load_dataset("json", data_files="/home/chaunm/workspace/bartpho/correction.jsonl", split='train')

dataset_base = dataset_base.sort("id")
id_list_base = [s['id'] for s in dataset_base]
dataset_large = dataset_large.filter(lambda x: x['id'] in id_list_base).sort("id")

print(dataset_large[1])
print(dataset_base[1])
diff_file = open("diff.jsonl", 'w')

for i in range(len(dataset_base)):
    sample_base = dataset_base[i]
    sample_large = dataset_large[i]
    if sample_large['sentence'].replace(".", "").replace("mươi", "").replace("  ", " ").lower() != sample_base[
        'sentence'].replace(".", "").replace("mươi", "").replace("  ", " ").lower():
        write_sample = {"id": sample_base['id'], "sentence_base": sample_base['sentence'],
                        "sentence_large": sample_large['sentence']}
        diff_file.write(json.dumps(write_sample, ensure_ascii=False))
        diff_file.write("\n")
