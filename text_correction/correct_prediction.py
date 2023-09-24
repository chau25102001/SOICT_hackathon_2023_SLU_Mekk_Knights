import tqdm
from transformers import MBartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, \
    DataCollatorForSeq2Seq
import torch
import json
from datasets import load_dataset
import os
import termcolor
from argparse import ArgumentParser

parser = ArgumentParser(description='testing')
parser.add_argument("--stt_pred_path", type=str, default='/SOICT_hackathon_2023_SLU_Mekk_Knights/stt_pred.jsonl',
                    help="path to the jsonl stt prediction")
parser.add_argument("--model_checkpoint", type=str, default='checkpoint/checkpoint_best.pt')
args = parser.parse_args()

if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")
    data_path = args.stt_pred_path
    if data_path is None or not os.path.exists(data_path):
        print(termcolor.colored(f"ERROR: cannot find stt prediction: {data_path}"))
        exit()
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location='cpu')[
            'state_dict'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print("loaded state dict")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    stt_dataset = load_dataset("json", data_files=data_path,
                               split="train")
    with open("correction.jsonl", 'w', encoding='utf8') as f:
        for sample in tqdm.tqdm(stt_dataset, total=len(stt_dataset)):
            id_ = sample['id']
            sentence: str = sample['sentence']
            if 'của' in sentence:
                index = sentence.find('của')
                s_index = index + 4
                sentence = sentence[:s_index] + sentence[s_index].upper() + sentence[s_index + 1:]
            # sentence.replace(".", '')
            encoded_source = tokenizer(sentence.replace(".", ''), return_tensors='pt')
            new_length = encoded_source['input_ids'].shape[1]
            for k, v in encoded_source.items():
                encoded_source[k] = v.to(device)
            new_sentence = tokenizer.decode(
                model.generate(**encoded_source, max_new_tokens=new_length, num_beams=5, ).cpu().numpy().tolist()[0],
                skip_special_tokens=True, clean_up_tokenization_space=False)
            save_string = json.dumps({"id": id_,
                                      "sentence": new_sentence, "original_sentence": sentence}, ensure_ascii=False)
            f.write(save_string)
            f.write('\n')
