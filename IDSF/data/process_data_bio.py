import copy

from datasets import load_dataset, concatenate_datasets
import re
import sys
import random

sys.path.append("..")
from augment_data_bio import *
import datasets

random.seed(42)
intent_mapping = {'bật thiết bị': 0,
                  'giảm mức độ của thiết bị': 1,
                  'giảm nhiệt độ của thiết bị': 2,
                  'giảm âm lượng của thiết bị': 3,
                  'giảm độ sáng của thiết bị': 4,
                  'hủy hoạt cảnh': 5,
                  'kiểm tra tình trạng thiết bị': 6,
                  'kích hoạt cảnh': 7,
                  'mở thiết bị': 8,
                  'tăng mức độ của thiết bị': 9,
                  'tăng nhiệt độ của thiết bị': 10,
                  'tăng âm lượng của thiết bị': 11,
                  'tăng độ sáng của thiết bị': 12,
                  'tắt thiết bị': 13,
                  'đóng thiết bị': 14}

slot_type_mapping = {'changing value': 0,
                     'command': 1,
                     'device': 2,
                     'duration': 3,
                     'location': 4,
                     'scene': 5,
                     'target number': 6,
                     'time at': 7}

slot_type_question_mapping = {'changing value': [0, 1],
                              'command': [2, 3],
                              'device': [4, 5],
                              'duration': [6, 7],
                              'location': [8, 9],
                              'scene': [10, 11],
                              'target number': [12, 13],
                              'time at': [14, 15],
                              'O': 16
                              }


def find_annotation_indices(sentence, annotation):
    indices = []
    accumulated_len = 0
    for match in re.finditer(r'\[ ([^:]+) : ([^\]]+) \]', annotation):
        # slot_type = match.span(1)  # start and end index of slot type
        slot = match.span(2)
        annotation_type = match.group(1).strip()
        annotation_value = match.group(2).strip()

        start_index = slot[0] - accumulated_len - 5 - len(annotation_type)
        end_index = start_index + len(annotation_value)
        accumulated_len += 7 + len(annotation_type)

        indices.append([annotation_type, start_index, end_index])
    assert len(indices) == annotation.count("["), f"{indices}, {annotation}, {sentence}"
    return indices


def get_slot_label(segmented_context, slot_indices):
    '''
    :param segmented_context: segmented context sentence
    :param slot_indices: list of [slot_type, char_start_index, char_end_index]
    :return: list of BIO tags
    '''
    context_words = segmented_context.split(" ")  # convert to word segments
    labels = []
    current_len = 0
    first_tag_index = 0  # start with the frst tag
    for word in context_words:
        if first_tag_index < len(slot_indices):
            if current_len == slot_indices[first_tag_index][1]:  # match the start current tag
                fill_slot = "B_" + slot_indices[first_tag_index][0]
            elif slot_indices[first_tag_index][1] < current_len < slot_indices[first_tag_index][
                2]:  # in between start and end
                fill_slot = "I_" + slot_indices[first_tag_index][0]
            else:
                fill_slot = 'O'
            if current_len + len(word) + 1 > slot_indices[first_tag_index][2]:  # end of tag
                first_tag_index += 1
        else:
            fill_slot = "O"
        current_len += len(word) + 1  # space
        labels.append(fill_slot)

    return context_words, labels


def process_sample(sample):
    new_sample = {}
    new_sample['context'] = []
    new_sample['slot_label'] = []
    new_sample['intent_label'] = []
    new_sample['audio_file'] = []
    new_sample['original_sentence'] = []
    new_sample['original_sentence_annotation'] = []
    for sample_id, _ in enumerate(sample['sentence']):
        context = sample['sentence'][sample_id]
        context = context.strip()  # remove space at start and end

        annotation = sample['sentence_annotation'][sample_id].strip()
        intent = sample['intent'][sample_id]
        slot_indices = find_annotation_indices(sentence=context, annotation=annotation)
        slot_indices = sorted(slot_indices, key=lambda x: x[1])
        intent_label = intent_mapping[intent]
        try:
            segmented_context = context
        except:
            print(context)
            continue
        # new_context, new_slot_indices = refine_context(segmented_context, slot_indices)
        new_context, new_slot_indices = segmented_context, slot_indices

        word_level_context, slot_label = get_slot_label(new_context, new_slot_indices)
        assert len(word_level_context) == len(slot_label)
        new_sample['context'].append(word_level_context)
        new_sample['slot_label'].append(slot_label)
        new_sample['intent_label'].append(intent_label)
        new_sample['audio_file'].append(sample['file'][sample_id])
        new_sample['original_sentence'].append(sample['sentence'][sample_id])
        new_sample['original_sentence_annotation'].append(sample['sentence_annotation'][sample_id])
    return new_sample


def refine_context(context, answers):
    new_context = []
    added = 0
    for i, c in enumerate(context):
        if c == ',' or c == '.' or c == '?' or c == '!':
            if context[i - 1] != " " and i >= 1:  # need to separate
                new_context.append(" ")
                added += 1
                for j, a in enumerate(answers):
                    if i <= a[1] - added:
                        answers[j][1] = answers[j][1] + 1
                    if i <= a[2] - added:
                        answers[j][2] = answers[j][2] + 1
        new_context.append(c)
    new_context = "".join(new_context)
    # for a in answers:
    #     a[0] = new_context[a[1]:a[2]]
    return new_context, answers


def filter_invalid_annotation(sample):
    annotation = sample['slot_label']
    is_invalid = False
    for j, s in enumerate(annotation):
        if s.startswith("I"):
            if j >= 1 and annotation[j - 1].startswith("O"):
                is_invalid = True
            elif j >= 1 and annotation[j - 1].startswith("B") and annotation[j - 1].split("_")[-1] != s.split("_")[
                -1]:
                is_invalid = True
    return not is_invalid


if __name__ == "__main__":
    dataset = load_dataset('json', data_files='train_final_20230919.jsonl', split='train')
    dataset = dataset.map(refine_dataset, batched=True, load_from_cache_file=False)
    dataset = dataset.map(refine_slot_label_changing_value, batched=True, load_from_cache_file=False)
    clean_train = copy.deepcopy(dataset)
    train_split = []
    val_split = []

    for v in intent_mapping.keys():
        intent_split = dataset.filter(lambda x: x['intent'] == v, load_from_cache_file=False)
        train_len = int(len(intent_split) * 0.9)
        train_partition = intent_split.select(list(range(train_len)))
        val_partition = intent_split.select(list(range(train_len, len(intent_split))))
        train_split.append(train_partition)
        val_split.append(val_partition)

    train_set = datasets.concatenate_datasets(train_split)
    val_set = datasets.concatenate_datasets(val_split)
    # train_set = copy.deepcopy(dataset)

    train_set = train_set.map(add_location, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(partial(add_duration, times=1), batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)

    train_set = train_set.map(random_change_command, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(random_change_device, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(random_change_number, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(partial(random_change_duration, times=1), batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(random_change_time_at, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    # train_set = train_set.map(replace_with_synonym, batched=True, load_from_cache_file=False)
    # train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)

    train_set = train_set.map(random_scene_aug, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(add_confusing_device, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(add_confusing_slot, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)

    train_set = train_set.map(partial(generate_yes_no, prob=0.5), batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(partial(reverse_intent, prob=0.2), batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)
    train_set = train_set.map(clean_command, batched=True, load_from_cache_file=False)
    train_set = train_set.map(strip_spaces, batched=True, load_from_cache_file=False)

    ratio = int(len(train_set) / len(clean_train))
    aux_datasets = [clean_train] if ratio <= 1 else [clean_train for i in range(ratio // 2)]
    print(f"concatenating augmented dataset: {len(train_set)}, and original dataset: {len(clean_train)}")
    train_set = concatenate_datasets([train_set] + aux_datasets)

    processed_train_set = train_set.map(process_sample, batched=True,
                                        remove_columns=train_set.column_names,
                                        load_from_cache_file=False)
    processed_val_set = val_set.map(process_sample, batched=True,
                                    remove_columns=val_set.column_names,
                                    load_from_cache_file=False
                                    )

    print(len(processed_train_set))
    print(len(processed_val_set))
    processed_train_set = processed_train_set.filter(lambda x: filter_invalid_annotation(x), load_from_cache_file=False)
    processed_val_set = processed_val_set.filter(lambda x: filter_invalid_annotation(x), load_from_cache_file=False)
    print(len(processed_train_set))
    print(len(processed_val_set))
    processed_train_set.save_to_disk("data_bio/processed_train")
    processed_val_set.save_to_disk("data_bio/processed_val")
