import termcolor
import torch
import random
import numpy as np
import os

import transformers


def ensemble_checkpoints(checkpoints, weights=None):
    normalized_weights = [1 / len(checkpoints) for _ in checkpoints]
    if weights is not None:
        assert len(checkpoints) == len(weights)
        normalized_weights = np.array(weights) / np.sum(weights)
    print(termcolor.colored(f"ensemble {len(checkpoints)} models, with weights: {normalized_weights}"))
    result = {}
    for i, cp in enumerate(checkpoints):
        for k in cp.keys():
            if k not in result:
                result[k] = cp[k] * normalized_weights[i]
            else:
                result[k] = result[k] + normalized_weights[i] * cp[k]
    return result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def accumulate(self):
        return self.sum


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def iou(pred_start, pred_end, gt_start, gt_end):
    '''
    :param pred_start: tensor: b
    :param pred_end: tensor: b
    :param gt_start: tensor: b
    :param gt_end: tensor: b
    :return:
    '''
    len1 = pred_end - pred_start + 1
    ignore_index = torch.where(len1 < 0)
    len2 = gt_end - gt_start + 1
    overlap = torch.minimum(pred_end, gt_end) - torch.maximum(pred_start, gt_start) + 1
    overlap = torch.where(overlap > 0, overlap, 0)
    union = len1 + len2 - overlap
    iou = (overlap + 1e-5) / (union + 1e-5)
    iou[ignore_index] = 0.0
    return torch.mean(iou)  # mean over batch


def accuracy(pred, gt, mask=None):
    '''
    :param pred: bsize x d x c
    :param gt: bsize x d
    :param mask: bsize x d
    :return:
    '''
    pred = torch.argmax(pred, dim=-1)
    count = torch.ones_like(pred)
    if mask is not None:
        count = count * mask
        tp = torch.sum(torch.eq(pred, gt) * mask)
    else:
        tp = torch.sum(torch.eq(pred, gt))
    return tp / torch.sum(count)


def seed_everything(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def question_answering(question, context, tokenizer, model, device, topk=1):
    encoded_inputs = tokenizer(question,
                               context,
                               truncation="only_second",
                               max_length=50,
                               # stride=128,
                               return_tensors='np',
                               return_token_type_ids=True,
                               return_overflowing_tokens=True,
                               return_offsets_mapping=True,
                               return_special_tokens_mask=True,
                               padding="max_length"
                               )
    num_spans = len(encoded_inputs['input_ids'])
    p_mask = np.asarray(
        [
            [tok != 1 for tok in encoded_inputs.sequence_ids(span_id)]
            for span_id in range(num_spans)
        ]
    )

    if tokenizer.cls_token_id is not None:
        cls_index = np.nonzero(encoded_inputs["input_ids"] == tokenizer.cls_token_id)
        p_mask[cls_index] = 0
    inputs = {"input_ids": torch.tensor(encoded_inputs['input_ids'], dtype=torch.long).to(device),
              "attention_mask": torch.tensor(encoded_inputs["attention_mask"], dtype=torch.long).to(device)}
    outputs = model(**inputs)
    starts = outputs[0].cpu().numpy()
    ends = outputs[1].cpu().numpy()
    cls = outputs[2]
    cls = torch.softmax(cls, dim=-1)[0]
    cls_predict = torch.argmax(cls)
    for k, v in inputs.items():
        inputs[k] = v.cpu().numpy()
    answers = []
    for i, (start_, end_) in enumerate(zip(starts, ends)):
        offsets = encoded_inputs["offset_mapping"][i]
        undesired_tokens = p_mask[i] - 1
        undesired_tokens = undesired_tokens & inputs['attention_mask'][i]
        undesired_tokens_mask = undesired_tokens == 0.0
        start_ = np.where(undesired_tokens_mask, -10000.0, start_)
        end_ = np.where(undesired_tokens_mask, -10000.0, end_)
        start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
        end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
        start_[0] = end_[0] = 0.0
        start_ids, end_ids, scores = decode(start_, end_, topk, undesired_tokens=undesired_tokens,
                                            max_answer_len=50)
        for s, e, score in zip(start_ids, end_ids, scores):
            start_index = offsets[s][0]
            end_index = offsets[e][1]
            answers.append({
                "score": score,
                "start": start_index,
                "end": end_index,
                "answer": context[start_index:end_index],
                "label": [cls_predict.item(), cls[cls_predict].item()]
            })
    answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:topk]
    # if len(answers) == 1:
    #     return answers[0]
    return answers


def decode(start, end, topk, max_answer_len=15, undesired_tokens=None):
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
    starts = starts[desired_spans]
    ends = ends[desired_spans]
    scores = candidates[0, starts, ends]

    return starts, ends, scores


intent_mapping = {0: 'bật thiết bị',
                  1: 'giảm mức độ của thiết bị',
                  2: 'giảm nhiệt độ của thiết bị',
                  3: 'giảm âm lượng của thiết bị',
                  4: 'giảm độ sáng của thiết bị',
                  5: 'hủy hoạt cảnh',
                  6: 'kiểm tra tình trạng thiết bị',
                  7: 'kích hoạt cảnh',
                  8: 'mở thiết bị',
                  9: 'tăng mức độ của thiết bị',
                  10: 'tăng nhiệt độ của thiết bị',
                  11: 'tăng âm lượng của thiết bị',
                  12: 'tăng độ sáng của thiết bị',
                  13: 'tắt thiết bị',
                  14: 'đóng thiết bị'}

intent_mapping_for_correction = {
    2: 1,
    3: 1,
    4: 1,
    10: 9,
    11: 9,
    12: 9
}

slot_mapping = {
    0: 'B_changing value',
    1: 'B_command',
    2: 'B_device',
    3: 'B_duration',
    4: 'B_location',
    5: 'B_scene',
    6: 'B_target number',
    7: 'B_time at',
    8: 'I_changing value',
    9: 'I_command',
    10: 'I_device',
    11: 'I_duration',
    12: 'I_location',
    13: 'I_scene',
    14: 'I_target number',
    15: 'I_time at',
    16: 'O',
}


def ner(context, tokenizer: transformers.PreTrainedTokenizer, model, device):
    encoded_inputs = tokenizer(context, is_split_into_words=True, return_tensors='pt')
    inputs = {"input_ids": torch.tensor(encoded_inputs['input_ids'], dtype=torch.long).to(device),
              "attention_mask": torch.tensor(encoded_inputs["attention_mask"], dtype=torch.long).to(device)}
    outputs = model(**inputs)
    intent_pred = outputs['intent_pred'][0]
    slot_pred = outputs['slot_pred'][0]
    # print(outputs['intent_logits'].shape)

    intent_pred_id = torch.argmax(intent_pred, dim=-1).item()
    intent_pred_score = intent_pred[intent_pred_id].item()
    # if intent_pred_id in intent_mapping_for_correction.keys() and 2 not in slot_pred:  # nếu intent là tăng giảm, nhưng không thấy device thì nó sẽ được sửa thành tăng giảm mức độ
    #     intent_pred_id = intent_mapping_for_correction[intent_pred_id]
    intent_pred = intent_mapping[intent_pred_id]
    slot_pred = [slot_mapping[s] for s in slot_pred]

    return {"context": tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'].cpu().numpy().tolist()[0]),
            "ids": encoded_inputs['input_ids'].cpu().numpy().tolist()[0],
            'intent': [intent_pred, intent_pred_score],
            'slot': slot_pred}
