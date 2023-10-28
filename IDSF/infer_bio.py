import os.path

import termcolor
import torch
import tqdm

from models.phobert_jointidsf import *
from transformers import AutoTokenizer
from utils.utils import ner, ensemble_checkpoints
from configs.bio_config import get_config
from datasets import load_dataset
import json
import re
from argparse import ArgumentParser
import json

parser = ArgumentParser(description='testing')
parser.add_argument("--stt_pred_path", type=str, default='/SOICT_hackathon_2023_SLU_Mekk_Knights/stt_pred.jsonl',
                    help="path to the jsonl stt prediction")
parser.add_argument("--model_checkpoint", type=str, default='checkpoint/checkpoint_best.pt')
parser.add_argument("--ema", action='store_true', default=False)
args = parser.parse_args()
if __name__ == "__main__":
    import warnings

    data_path = args.stt_pred_path
    with open("data/text_to_num.json", 'r') as f:
        text_num_mapping = json.load(f)
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
    model = PhobertBIO(config.model_card, config.num_intent_classes,
                       config.num_slot_classes, use_etf=config.use_etf, use_attn=config.use_attn,
                       drop_out=config.drop_out,
                       attention_embedding_size=config.attention_embedding_size, use_crf=config.use_crf,
                       tag_mapping=config.slot_mapping)
    checkpoints = args.model_checkpoint.split(",")
    if len(checkpoints) == 1:
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict):
            if args.ema and 'ema' in checkpoint.keys():
                print("ema")
                checkpoint = checkpoint['ema']
            elif 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
    else:
        checkpoints = [torch.load(cp) for cp in checkpoints]
        if args.ema:
            checkpoints = [cp['ema'] if 'ema' in cp else cp['state_dict'] for cp in checkpoints]
        else:
            checkpoints = [cp['state_dict'] for cp in checkpoints]
        checkpoint = ensemble_checkpoints(checkpoints, weights=list(range(len(checkpoints), 0, -1)))
        # checkpoint = ensemble_checkpoints(checkpoints, weights=[3, 7])

    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    model.eval()
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
            # original_context = None
            # if 'original_sentence' in sample:
            #     original_context = sample['original_sentence']
            #     original_context = original_context.strip()
            #     original_context = original_context.replace(",", " , ").replace(".", " . ").replace("?", " ? ").replace(
            #         "!",
            #         " ! ").replace(
            #         "%", " % ").replace("không giờ", "0 giờ").replace("không phút", "0 phút").replace(
            #         "  ", " ")
            #     original_context = re.sub(pattern, replace_longest_substring, original_context)
            #     original_context = original_context.replace(" phần trăm", "%").strip()
            #     original_context = original_context.split(" ")

            id = sample['id']
            context = context.strip()
            context = context.replace(",", " , ").replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(
                "%", " % ").replace("không giờ", "0 giờ").replace("không phút", "0 phút").replace(
                "  ", " ")
            context = re.sub(pattern, replace_longest_substring, context)
            context = context.replace(" phần trăm", "%").strip()
            context_save = context
            context = context.split(" ")
            # original_context = None
            # if original_context is not None:
            #     o_ner_results = ner(original_context, tokenizer=tokenizer, model=model, device=config.device)
            #     if 'B_scene' in o_ner_results['slot']:
            #         intent = o_ner_results['intent'][0]
            #         slots = o_ner_results['slot']
            #         context = o_ner_results['context']
            #         ids = o_ner_results['ids']
            #     else:
            #         ner_results = ner(context, tokenizer=tokenizer, model=model, device=config.device)
            #         intent = ner_results['intent'][0]
            #         slots = ner_results['slot']
            #         context = ner_results['context']
            #         ids = ner_results['ids']
            # else:
            ner_results = ner(context, tokenizer=tokenizer, model=model, device=config.device)
            intent = ner_results['intent'][0]
            slots = ner_results['slot']
            context = ner_results['context']
            ids = ner_results['ids']
            slot_list = []
            current_filler = []
            i = 0
            cond1 = False
            cond2 = True
            cond3 = False
            cond4 = True
            while i < len(slots):
                slot = slots[i]
                if slot.startswith("B"):  # begin of a new slot
                    current_slot = slot[2:]
                    if current_slot == 'device':
                        cond2 = False
                    if current_slot == 'command':
                        cond4 = False
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
                    if current_slot == 'chếch':
                        current_slot = 'check'
                    # value.replace(" phần trăm", "%").strip()
                    if not (('hoạt cảnh' in intent and current_slot.lower() == 'command') or (
                            'hoạt cảnh' not in intent and current_slot.lower() == 'scene')):
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
