import os.path
import termcolor
import torch
import tqdm

from models.phobert_jointidsf import *
from transformers import AutoTokenizer
from utils.utils import ner
from configs.bio_config import get_config
from datasets import load_dataset
import json
from data.constants import text_num_mapping
import re
from argparse import ArgumentParser

parser = ArgumentParser(description='testing')
parser.add_argument("--stt_pred_path", type=str, default='/SOICT_hackathon_2023_SLU_Mekk_Knights/stt_pred.jsonl',
                    help="path to the jsonl stt prediction")
parser.add_argument("--slot_checkpoint", type=str)
parser.add_argument("--intent_checkpoint", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    import warnings

    data_path = args.stt_pred_path
    if data_path is None or not os.path.exists(data_path):
        print(termcolor.colored(f"ERROR: cannot find stt prediction: {data_path}"))
        exit()
    test_dataset = load_dataset("json", data_files=data_path,
                                split='train')
    # test_dataset = load_dataset("json", data_files="data/wav2vec_2109_lasthope.jsonl", split='train')

    warnings.filterwarnings('ignore')
    config = get_config(False)
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(config.model_card)
    intent_model = PhobertBIO(config.model_card, config.num_intent_classes,
                              config.num_slot_classes, use_etf=config.use_etf, use_attn=config.use_attn,
                              drop_out=config.drop_out,
                              attention_embedding_size=config.attention_embedding_size, use_crf=config.use_crf,
                              tag_mapping=config.slot_mapping)
    slot_model = PhobertBIO(config.model_card, config.num_intent_classes,
                            config.num_slot_classes, use_etf=config.use_etf, use_attn=config.use_attn,
                            drop_out=config.drop_out,
                            attention_embedding_size=config.attention_embedding_size, use_crf=config.use_crf,
                            tag_mapping=config.slot_mapping)

    intent_checkpoint = torch.load(args.intent_checkpoint, map_location='cpu')['ema']
    slot_checkpoint = torch.load(args.slot_checkpoint, map_location='cpu')['state_dict']

    intent_model.load_state_dict(intent_checkpoint)
    slot_model.load_state_dict(slot_checkpoint)

    intent_model = intent_model.to(config.device)
    intent_model.eval()

    slot_model = slot_model.to(config.device)
    slot_model.eval()

    sorted_keys = sorted(text_num_mapping.keys(), key=len, reverse=True)

    correct_intent = {'giảm mức độ của thiết bị': 'giảm độ sáng của thiết bị',
                      'giảm nhiệt độ của thiết bị': 'giảm độ sáng của thiết bị',
                      'giảm âm lượng của thiết bị': 'giảm độ sáng của thiết bị',
                      'giảm độ sáng của thiết bị': 'giảm độ sáng của thiết bị',
                      'tăng mức độ của thiết bị': 'tăng độ sáng của thiết bị',
                      'tăng nhiệt độ của thiết bị': 'tăng độ sáng của thiết bị',
                      'tăng âm lượng của thiết bị': 'tăng độ sáng của thiết bị',
                      'tăng độ sáng của thiết bị': 'tăng độ sáng của thiết bị'
                      }

    correct_intent_temp = {'giảm mức độ của thiết bị': 'giảm nhiệt độ của thiết bị',
                           'giảm nhiệt độ của thiết bị': 'giảm nhiệt độ của thiết bị',
                           'giảm âm lượng của thiết bị': 'giảm nhiệt độ của thiết bị',
                           'giảm độ sáng của thiết bị': 'giảm nhiệt độ của thiết bị',
                           'tăng mức độ của thiết bị': 'tăng nhiệt độ của thiết bị',
                           'tăng nhiệt độ của thiết bị': 'tăng nhiệt độ của thiết bị',
                           'tăng âm lượng của thiết bị': 'tăng nhiệt độ của thiết bị',
                           'tăng độ sáng của thiết bị': 'tăng nhiệt độ của thiết bị'
                           }


    def replace_longest_substring(match):
        return str(text_num_mapping[match.group(0)])


    pattern = "|".join(re.escape(key) for key in sorted_keys)
    pattern = f"\\b({pattern})\\b"
    save_list = []
    with torch.no_grad():
        for sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
            context = sample['sentence']
            id = sample['id']
            context = context.strip()
            context = context.replace(",", " , ").replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(
                "%", " % ").replace("không giờ", "0 giờ").replace("không phút", "0 phút").replace(
                "  ", " ")
            context = re.sub(pattern, replace_longest_substring, context)
            context = context.replace(" phần trăm", "%").strip()
            context_save = context
            context = context.split(" ")
            intent_result = ner(context, tokenizer=tokenizer, model=intent_model, device=config.device)
            intent = intent_result['intent'][0]
            slots = ner(context, tokenizer=tokenizer, model=slot_model, device=config.device)['slot']
            context = intent_result['context']
            ids = intent_result['ids']

            slot_list = []
            current_filler = []
            i = 0
            cond1 = False
            cond2 = True
            cond3 = False
            while i < len(slots):
                slot = slots[i]
                if slot.startswith("B"):  # begin of a new slot
                    current_slot = slot[2:]
                    if current_slot == 'device':
                        cond2 = False
                    current_filler.append(ids[i])
                    count = 0
                    for j in range(i + 1, len(slots)):
                        next_slot = slots[j]
                        if (next_slot.startswith('B') and next_slot[2:] != current_slot) or next_slot.startswith('O'):
                            break
                        elif next_slot.startswith('I') and next_slot[2:] != current_slot:
                            break
                        else:
                            current_filler.append(ids[j])
                            count += 1

                    value: str = tokenizer.decode(current_filler).replace("_", " ").replace("độ khoảng", "").replace(
                        "khoảng", '').replace("tầm",
                                              '').replace(
                        "  ", " ").strip()
                    if "%" in value and current_slot.lower() in ['changing value', 'target number']:
                        cond1 = True
                        pos = value.find("%")
                        if value[pos - 1] == " ":
                            value = value[:pos - 1] + value[pos:]
                    if 'độ' in value and current_slot.lower() in ['changing value', 'target number']:
                        cond3 = True

                    # value.replace(" phần trăm", "%").strip()
                    slot_list.append(
                        {"type": current_slot.strip(), "filler": value})
                    current_filler = []
                    i = i + count
                i += 1
            if cond1 and cond2 and intent in correct_intent:
                intent = correct_intent[intent]
            if cond3 and intent in correct_intent:
                intent = correct_intent_temp[intent]
            save_list.append(
                {"intent": intent, "entities": slot_list, "file": f"{id}.wav"})
        with open("predictions.jsonl", 'w', encoding='utf8') as f:
            for item in save_list:
                save_string = json.dumps(item, ensure_ascii=False)
                f.write(save_string)
                f.write("\n")
