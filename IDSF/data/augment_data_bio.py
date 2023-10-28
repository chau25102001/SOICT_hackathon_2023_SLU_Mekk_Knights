import random

from datasets import load_from_disk, load_dataset
from numpy import finfo, inexact

from constants import *
from pprint import pprint
import re
from functools import partial


# dataset = load_dataset('json', data_files='/home/hadoop/Downloads/train.jsonl', split='train')


def find_annotation_indices(sentence, annotation):
    indices = []
    accumulated_len = 0
    for match in re.finditer(r'\[ ([^:]+) : ([^\]]+) \]', annotation):
        annotation_span = match.span(2)

        slot = match.span(2)
        annotation_type = match.group(1).strip()
        annotation_value = match.group(2).strip()

        start_index = slot[0] - accumulated_len - 5 - len(annotation_type)
        end_index = start_index + len(annotation_value)
        accumulated_len += 7 + len(annotation_type)

        indices.append([annotation_type, start_index, end_index, annotation_span[0], annotation_span[1]])
    assert len(indices) == annotation.count("["), f"{sentence}, {annotation}"
    return indices


def random_change_device(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "device":  # found a device slot
                current_slot_value = sentence[sentence_start:sentence_end]
                device_choices = possible_intent_device_mapping[intent.lower().strip()]
                if len(device_choices) == 0:
                    break
                while True:  # choose a new device to replace
                    new_slot_value = random.choice(device_choices)
                    if new_slot_value != current_slot_value:
                        break
                if random.uniform(0, 1) < 0.3:
                    new_slot_value = new_slot_value + " số " + str(random.randint(0, 50))
                new_sentence = sentence[:sentence_start] + new_slot_value + sentence[sentence_end:]
                new_sentence_annotation = sentence_annotation[:annotation_start] + new_slot_value + sentence_annotation[
                                                                                                    annotation_end:]
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    else:
                        new_sample[k].append(new_sentence_annotation)
    return new_sample


def random_change_command(sample):
    new_sample = sample.copy()
    rate = 0.1
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)

        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "command" and random.uniform(0, 1) < rate:  # found a device slot
                current_slot_value = sentence[sentence_start:sentence_end]
                command_choices = possible_intent_command_mapping[intent.lower().strip()]
                if ('tăng' in intent or 'giảm' in intent) and ('lên' in sentence or 'xuống' in sentence):
                    command_choices.extend(['chỉnh', 'điều chỉnh'])
                if len(command_choices) == 0:
                    break
                while True:  # choose a new device to replace
                    new_slot_value = random.choice(command_choices)
                    if new_slot_value != current_slot_value:
                        break
                new_sentence = sentence[:sentence_start] + new_slot_value + sentence[sentence_end:]
                new_sentence_annotation = sentence_annotation[:annotation_start] + new_slot_value + sentence_annotation[
                                                                                                    annotation_end:]
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    else:
                        new_sample[k].append(new_sentence_annotation)
    return new_sample


def random_change_number(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        if 'nhiệt độ' in intent:
            value_type = 0  # temperature
        elif 'mức độ' in intent:
            value_type = 1  # value
        elif 'âm lượng' in intent:
            value_type = 2  # volume
        elif 'độ sáng' in intent:
            value_type = 3  # illumination
        else:
            value_type = 4

        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == 'changing value':
                rate += 0.2
            if slot == "target number" or slot == 'changing value' and random.uniform(0,
                                                                                      1) < rate:  # found a number slot
                current_slot_value = sentence[sentence_start:sentence_end]
                number, unit, unit_index = find_unit_in_number(current_slot_value)
                try:
                    int(number)
                except:
                    print(current_slot_value)
                    continue
                prefix = ' '
                postfix = ' '
                if value_type == 0:  # C degree
                    rate = random.uniform(0, 1)
                    if rate < 0.33:  # percentage
                        new_slot_value = str(random.randint(10, 300)) + "%"
                    elif rate < 0.88:  # temperature
                        new_slot_value = str(random.randint(20, 300)) + random.choice([" độ C", " độ", ' '])
                    else:  # level
                        new_slot_value = str(random.randint(0, 10))
                        if slot == 'target number':
                            prefix = random.choice([' mức ', ' nấc ', ' '])
                            postfix = random.choice([' mức ', ' nấc ', ' '])
                elif value_type != 1:
                    rate = random.uniform(0, 1)
                    if rate < 0.5:
                        new_slot_value = str(random.randint(10, 300)) + "%"
                    else:
                        new_slot_value = str(random.randint(1, 10))
                        if slot == 'target number':
                            prefix = random.choice([' mức ', ' nấc ', ' '])
                            postfix = random.choice([' mức ', ' nấc ', ' '])
                else:  # integer value for level
                    new_slot_value = str(random.randint(1, 10))
                    if slot == 'target number':
                        prefix = random.choice([' mức ', ' nấc ', ' '])
                        postfix = random.choice([' mức ', ' nấc ', ' '])
                if prefix != ' ' and postfix != ' ':
                    if random.random() < 0.5:
                        postfix = ' '
                    else:
                        prefix = ' '

                new_sentence = sentence[:sentence_start] + prefix + new_slot_value + postfix + sentence[sentence_end:]
                new_sentence_annotation = sentence_annotation[
                                          :annotation_start - 5 - len(slot)] + prefix + sentence_annotation[
                                                                                        annotation_start - 5 - len(
                                                                                            slot):annotation_start] + new_slot_value + sentence_annotation[
                                                                                                                                       annotation_end:annotation_end + 2] + postfix + sentence_annotation[
                                                                                                                                                                                      annotation_end + 2:]

                new_sentence.replace("mức mức", "mức")
                new_sentence.replace("mức nấc", "mức")
                new_sentence.replace("nấc nấc", "nấc")
                new_sentence.replace("nấc mức", "nấc")

                new_sentence_annotation.replace("mức mức", "mức")
                new_sentence_annotation.replace("mức nấc", "mức")
                new_sentence_annotation.replace("nấc nấc", "nấc")
                new_sentence_annotation.replace("nấc mức", "nấc")
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    else:
                        new_sample[k].append(new_sentence_annotation)
                break
    return new_sample


def random_change_location(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "location" and random.uniform(0, 1) < rate:  # found a location slot
                current_slot_value = sentence[sentence_start:sentence_end]
                location_choices = location_list
                while True:  # choose a new device to replace
                    new_slot_value = replace_digits_randomly(random.choice(location_choices))
                    if new_slot_value != current_slot_value:
                        break
                found_digits = False
                for c in new_slot_value:
                    if c.isdigit():
                        found_digits = True
                if not found_digits and 'của' not in new_slot_value and 'số' not in new_slot_value and 'tầng' not in new_slot_value and random.uniform(
                        0, 1) < 0.3:
                    new_slot_value = new_slot_value + " " + random.choice(['số', 'tầng', '']) + " " + str(
                        random.randint(0,
                                       10))
                new_sentence = sentence[:sentence_start] + new_slot_value + sentence[sentence_end:]
                new_sentence_annotation = sentence_annotation[:annotation_start] + new_slot_value + sentence_annotation[
                                                                                                    annotation_end:]
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    else:
                        new_sample[k].append(new_sentence_annotation)
    return new_sample


def generate_time(time_at):
    hour = random.randint(0, 25)
    hour_unit = "giờ" if time_at else random.choice(["giờ", "tiếng"])
    minute = random.randint(0, 60)
    second = random.randint(0, 60)
    chance = random.uniform(0, 1)
    if time_at:
        if chance < 0.3:  # hour only
            return_string = str(hour) + f" {hour_unit}"
        elif chance < 0.6:
            return_string = str(hour) + random.choice([" rưỡi", " giờ rưỡi"])
        elif chance < 0.9:
            return_string = str(hour) + f" {hour_unit}" + random.choice([" kém", ""]) \
                            + f" {str(minute)}" + random.choice([" phút", ""])
        else:
            return_string = str(hour) + f" {hour_unit}" + random.choice([" kém", ""]) + f" {str(minute)}" + " phút" \
                            + f" {str(second)}" + random.choice([" giây", ''])
        freq_postfix_chance = random.uniform(0, 1)
        if freq_postfix_chance <= 0.5:
            return return_string
        else:
            return return_string + " " + random.choice(time_repeat_freq)
    else:  # Duration
        if chance < 0.3:  # hour only
            return_string = str(hour) + f" {hour_unit}"
        elif chance < 0.6:
            return_string = str(hour) + f" {hour_unit} rưỡi"
        elif chance < 0.9:
            return_string = str(hour) + f" {hour_unit}" + f" {str(minute)}" + random.choice([" phút", ""])
        else:
            return_string = str(hour) + f" {hour_unit}" + f" {str(minute)}" + " phút" \
                            + f" {str(second)}" + random.choice([" giây", ''])
        return return_string


def random_change_time_at(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "time at" and random.uniform(0, 1) < rate:  # found a device slot

                new_slot_value = generate_time(time_at=True)
                new_sentence = sentence[:sentence_start] + new_slot_value + sentence[sentence_end:]
                new_sentence_annotation = sentence_annotation[
                                          :annotation_start] + new_slot_value + sentence_annotation[
                                                                                annotation_end:]
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    else:
                        new_sample[k].append(new_sentence_annotation)
    return new_sample


def random_change_duration(sample, times=1):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        for t in range(times):
            for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
                if slot == "duration" and random.uniform(0, 1) < rate:  # found a device slot
                    new_slot_value = generate_time(time_at=False)
                    new_sentence = sentence[:sentence_start] + new_slot_value + sentence[sentence_end:]
                    new_sentence_annotation = sentence_annotation[
                                              :annotation_start] + new_slot_value + sentence_annotation[
                                                                                    annotation_end:]
                    for k in new_sample.keys():
                        if k not in ['sentence', 'sentence_annotation']:
                            new_sample[k].append(sample[k][i])
                        elif k == 'sentence':
                            new_sample[k].append(new_sentence)
                        else:
                            new_sample[k].append(new_sentence_annotation)
    return new_sample


def replace_with_synonym(sample):
    new_sample = sample.copy()
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        sentence = sentence.split(" ")
        sentence_annotation = sentence_annotation.split(" ")
        replaced = False
        for word in synonym:
            replace_word = random.choice(synonym[word])
            if word in sentence and random.uniform(0, 1) < 0.5:
                sentence = [w if w != word else replace_word for w in sentence]
                sentence_annotation = [w if w != word else replace_word for w in sentence_annotation]
                replaced = True
            # print(sentence)

        sentence = " ".join(sentence)
        sentence_annotation = " ".join(sentence_annotation)
        if replaced:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(sentence)
                else:
                    new_sample[k].append(sentence_annotation)
    return new_sample


def replace_digits_randomly(input_string):
    result = ""

    for char in input_string:
        if char.isdigit():
            random_digit = str(random.randint(0, 9))
            result += random_digit
        else:
            result += char

    return result


def add_duration(sample, times=1):  # add duration slot, before and after command slot, after device slot
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for _ in range(times):
        for i, _ in enumerate(sample['sentence']):
            intent = sample['intent'][i]
            rate = intent_to_rate[intent]
            sentence = sample['sentence'][i]
            sentence_annotation = sample['sentence_annotation'][i]
            indices = find_annotation_indices(sentence, sentence_annotation)
            added = False  # added or not
            template = "[ duration : {} ]"
            exists = False
            for slot, _, _, _, _ in indices:
                if slot == 'duration':
                    exists = True
            if exists:
                continue
            for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
                if slot == "command" and not added:  # insert before or after
                    add = random.uniform(0, 1)
                    if add < rate:
                        new_slot_value = generate_time(time_at=False)  # generate time
                        prefix = random.choice(["trong vòng ", "trong ", "khoảng ", "trong khoảng ", 'trong suốt '])
                        sentence = sentence[:sentence_end] + " " + prefix + new_slot_value + " " + sentence[
                                                                                                   sentence_end:]  # add before command
                        sentence_annotation = sentence_annotation[
                                              :annotation_end + 2] + " " + prefix + template.format(
                            new_slot_value) + " " + sentence_annotation[
                                                    annotation_end + 2:]
                        added = True
                elif slot == "device" or slot == "target number" or slot == "changing value" or slot == 'location' and not added:
                    add = random.uniform(0, 1)
                    if add < rate:  # 50 % add
                        new_slot_value = generate_time(time_at=False)  # generate time
                        prefix = random.choice(["trong vòng ", "trong ", "khoảng ", "trong khoảng ", 'trong suốt '])
                        sentence = sentence[:sentence_end] + " " + prefix + new_slot_value + " " + sentence[
                                                                                                   sentence_end:]  # add before command
                        sentence_annotation = sentence_annotation[
                                              :annotation_end + 2] + " " + prefix + template.format(
                            new_slot_value) + " " + sentence_annotation[
                                                    annotation_end + 2:]
                        added = True
                if added:

                    while "  " in sentence:
                        sentence = sentence.replace("  ", " ")
                    while "  " in sentence_annotation:
                        sentence_annotation = sentence_annotation.replace("  ", " ")

                    break
            if added:
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(sentence)
                    else:
                        new_sample[k].append(sentence_annotation)
    return new_sample


def add_location(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        added = False  # added or not
        template = "[ location : {} ]"
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        exists = False
        for slot, _, _, _, _ in indices:
            if slot == 'location':
                exists = True
        if exists:
            continue
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "device" and not added:
                add = random.uniform(0, 1)
                if add < rate:  # 50 % add
                    new_slot_value = replace_digits_randomly(random.choice(location_list))  # generate time
                    found_digits = False
                    for c in new_slot_value:
                        if c.isdigit():
                            found_digits = True
                    if found_digits and 'của' not in new_slot_value and 'số' not in new_slot_value and 'tầng' not in new_slot_value and random.uniform(
                            0, 1) < 0.3:
                        new_slot_value = new_slot_value + " " + random.choice(
                            ['số', 'tầng', '']) + " " + random.randint(0,
                                                                       10)

                    prefix = random.choice(["trong ", "ở ", "chỗ ", "ở chỗ "])
                    sentence = sentence[:sentence_end] + " " + prefix + new_slot_value + " " + sentence[
                                                                                               sentence_end:]  # add before command
                    sentence_annotation = sentence_annotation[
                                          :annotation_end + 2] + " " + prefix + " " + template.format(
                        new_slot_value) + sentence_annotation[
                                          annotation_end + 2:]
                    while "  " in sentence:
                        sentence = sentence.replace("  ", " ")
                    while "  " in sentence_annotation:
                        sentence_annotation = sentence_annotation.replace("  ", " ")

                    added = True
                if added:
                    break
        if added:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(sentence)
                else:
                    new_sample[k].append(sentence_annotation)
    return new_sample


def add_time_at(sample):
    new_sample = sample.copy()
    intent_to_rate = {'bật thiết bị': 0.2,
                      'giảm mức độ của thiết bị': 0.5,
                      'giảm nhiệt độ của thiết bị': 1.,
                      'giảm âm lượng của thiết bị': 1.,
                      'giảm độ sáng của thiết bị': 0.1,
                      'hủy hoạt cảnh': 1.,
                      'kiểm tra tình trạng thiết bị': 0.2,
                      'kích hoạt cảnh': 1.,
                      'mở thiết bị': 0.5,
                      'tăng mức độ của thiết bị': 1.,
                      'tăng nhiệt độ của thiết bị': 1.,
                      'tăng âm lượng của thiết bị': 1.,
                      'tăng độ sáng của thiết bị': 0.1,
                      'tắt thiết bị': 0.2,
                      'đóng thiết bị': 0.5}
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        added = False  # added or not
        template = "[ time at : {} ]"
        intent = sample['intent'][i]
        rate = intent_to_rate[intent]
        exists = False
        for slot, _, _, _, _ in indices:
            if slot == 'time at':
                exists = True
        if exists:
            continue
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "command" and not added:  # insert before or after
                add = random.uniform(0, 1)
                if random.uniform(0, 1) < 0.5 and add < rate:
                    new_slot_value = generate_time(time_at=True)  # generate time
                    prefix = random.choice(["lúc ", "vào "])
                    sentence = sentence[:sentence_start] + prefix + new_slot_value + " " + sentence[
                                                                                           sentence_start:]  # add before command
                    sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                        slot)] + prefix + template.format(new_slot_value) + " " + sentence_annotation[
                                                                                  annotation_start - 5 - len(slot):]
                    added = True
                elif add < rate:
                    new_slot_value = generate_time(time_at=True)  # generate time
                    prefix = random.choice(["lúc ", "vào "])
                    sentence = sentence[:sentence_end] + " " + prefix + new_slot_value + sentence[
                                                                                         sentence_end:]  # add before command
                    sentence_annotation = sentence_annotation[
                                          :annotation_end + 2] + " " + prefix + template.format(
                        new_slot_value) + sentence_annotation[
                                          annotation_end + 2:]
                    added = True
            elif slot == "device" or slot == "target number" or slot == "changing value" or slot == 'location' and not added:
                add = random.uniform(0, 1)
                if add < rate:  # 50 % add
                    new_slot_value = generate_time(time_at=True)  # generate time
                    prefix = random.choice(["lúc ", "vào "])
                    sentence = sentence[:sentence_end] + " " + prefix + new_slot_value + sentence[
                                                                                         sentence_end:]  # add before command
                    sentence_annotation = sentence_annotation[
                                          :annotation_end + 2] + " " + prefix + template.format(
                        new_slot_value) + sentence_annotation[
                                          annotation_end + 2:]
                    added = True
            elif slot == 'duration' and not added:
                new_slot_value = generate_time(time_at=True)  # generate time
                prefix = random.choice(["lúc ", "vào "])
                sentence = sentence[:sentence_start] + prefix + new_slot_value + " " + sentence[
                                                                                       sentence_start:]  # add before command
                sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                    slot)] + prefix + template.format(new_slot_value) + " " + sentence_annotation[
                                                                              annotation_start - 5 - len(slot):]
                added = True
            if added:
                break

        if added:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(sentence)
                else:
                    new_sample[k].append(sentence_annotation)
    return new_sample


def refine_dataset(sample):
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        annotation = sample['sentence_annotation'][i]
        annotation = annotation.replace("'[", "[")
        annotation = annotation.replace("]'", "]")
        annotation = annotation.replace("[", " [ ")
        annotation = annotation.replace("]", " ] ")
        sentence = sentence.replace(",", " , ")
        annotation = annotation.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        annotation = annotation.replace(".", " . ")
        sentence = sentence.replace("/", " / ")
        annotation = annotation.replace("/", " / ")
        sentence = sentence.replace("!", " ! ")
        annotation = annotation.replace("!", " ! ")
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        while "  " in annotation:
            annotation = annotation.replace("  ", " ")

        sample['sentence'][i] = sentence
        sample['sentence_annotation'][i] = annotation
    return sample


def strip_spaces(sample):
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        annotation = sample['sentence_annotation'][i]
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        while "  " in annotation:
            annotation = annotation.replace("  ", " ")
        sample['sentence'][i] = sentence.strip()
        sample['sentence_annotation'][i] = annotation.strip()
    return sample


def refine_slot_label_changing_value(sample):
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        annotation = sample['sentence_annotation'][i]
        slots = find_annotation_indices(sentence, annotation)
        slots = sorted(slots, key=lambda x: x[1])  # sort by start index in sentence
        for s, slot in enumerate(slots):
            slot_type = slot[0]
            if slot_type == 'changing value':
                prefix_list = need_to_change_prefix[slot_type]

                # find 2 word prefix to the slot
                start = slot[1] - 2
                prefix = ''
                count_space = 0
                while count_space != 2 and start >= 0:
                    prefix = sentence[start] + prefix
                    start -= 1
                    if sentence[start] == " ":
                        count_space += 1
                if prefix.lower().strip() in prefix_list or 'về' in prefix.lower().strip() or 'còn' in prefix.lower().strip() or 'đến' in prefix.lower().strip():  # need to change slot type
                    print("correcting: ", sentence, annotation)
                    annotation_start = slot[3]
                    annotation = annotation[:annotation_start - 3 - len(slot_type)] + "target number : " + annotation[
                                                                                                           annotation_start:]
                    sample['sentence_annotation'][i] = annotation
                    break
    return sample
    # print("dataset ", len(dataset))


# TODO: handle máy tính ơi, multiple time at (từ ... đến ...), multiple time at and duration (à ừm)
def add_confusing_device(sample):
    new_sample = sample.copy()
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        added = False
        rate = 0.05
        for pos, (slot, sentence_start, sentence_end, annotation_start, annotation_end) in enumerate(indices):
            if slot == "command" and not added and not any([w in sample['sentence'] for w in [' ơi ', ' à ']]):
                add = random.uniform(0, 1)
                if add < rate:
                    add_string = random.choice(possible_confusing_device) + " " + random.choice(possible_verbal_words)
                    sentence = sentence[:sentence_start] + add_string + " " + sentence[sentence_start:]
                    sentence_annotation = sentence_annotation[
                                          :annotation_start - 5 - len(slot)] + add_string + " " + sentence_annotation[
                                                                                                  annotation_start - 5 - len(
                                                                                                      slot):]
                    added = True
            elif pos == len(indices) - 1 and not added and intent.lower() not in ['kích hoạt cảnh',
                                                                                  'hủy hoạt cảnh']:
                add = random.uniform(0, 1)
                if add < rate:
                    add_string = generate_sentence()
                    sentence = sentence[:sentence_end] + " " + random.choice(
                        [", ", '']) + add_string.strip() + ' ' + sentence[sentence_end:]
                    sentence_annotation = sentence_annotation[:annotation_end + 2] + " " + random.choice(
                        [", ", '']) + add_string.strip() + ' ' + sentence_annotation[annotation_end + 2:]
                    added = True

            if added:
                break

        if not added and not any([w in sample['sentence'] for w in [' ơi ', ' à ']]):
            add = random.uniform(0, 1)
            if add < rate:  # add at the start or end of the sentence
                if random.uniform(0, 1) < 0.5:  # add to start
                    add_string = random.choice(possible_confusing_device) + " " + random.choice(
                        possible_verbal_words) + random.choice([" ,", ''])
                    sentence = add_string + " " + sentence
                    sentence_annotation = add_string + " " + sentence_annotation
                else:
                    add_string = random.choice([", ", ""]) + random.choice(
                        possible_confusing_device) + " " + random.choice(possible_verbal_words)
                    sentence = sentence + " " + add_string
                    sentence_annotation = sentence_annotation + " " + add_string
                added = True
        if not added and intent.lower() not in ['kích hoạt cảnh', 'hủy hoạt cảnh']:
            add = random.uniform(0, 1)
            if add < rate:  # add at the start or end of the sentence
                if random.uniform(0, 1) < 0.5:  # add to start
                    add_string = generate_sentence()
                    sentence = add_string + " " + sentence
                    sentence_annotation = add_string + " " + sentence_annotation
                else:
                    add_string = generate_sentence()
                    sentence = sentence + " " + add_string
                    sentence_annotation = sentence_annotation + " " + add_string
                added = True

        if added:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(sentence)
                else:
                    new_sample[k].append(sentence_annotation)
    return new_sample


# 'B_duration': 3,
#         'B_location': 4,
#         'B_target number': 6,
#         'B_time at': 7,
def add_confusing_slot(sample):
    new_sample = sample.copy()
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        intent = sample['intent'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        added = False
        rate = 0.05
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            add = random.uniform(0, 1)
            if slot == 'device' and not added and add < rate:
                template = "[ device : {} ]"
                current_slot_value = sentence[sentence_start:sentence_end]
                device_choices = possible_intent_device_mapping[intent.lower().strip()]
                if len(device_choices) == 0:
                    break
                while True:
                    new_slot_value = random.choice(device_choices)
                    if new_slot_value != current_slot_value:
                        break

                if random.uniform(0, 1) < 0.3:
                    new_slot_value = new_slot_value + " số " + str(random.randint(0, 50))
                prefix = random.choice(possible_confusion_words)
                sentence = sentence[:sentence_end] + " " + prefix + " " + new_slot_value + " " + sentence[sentence_end:]
                sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                    slot)] + current_slot_value + " " + prefix + " " + template.format(
                    new_slot_value) + " " + sentence_annotation[annotation_end + 2:]
                added = True
            elif slot == "location" and not added and add < rate:
                template = "[ location : {} ]"
                current_slot_value = sentence[sentence_start:sentence_end]
                while True:
                    new_slot_value = replace_digits_randomly(random.choice(location_list))
                    if new_slot_value != current_slot_value:
                        break

                found_digits = False
                for c in new_slot_value:
                    if c.isdigit():
                        found_digits = True
                if not found_digits and 'của' not in new_slot_value and 'số' not in new_slot_value and 'tầng' not in new_slot_value and random.uniform(
                        0, 1) < 0.3:
                    new_slot_value = new_slot_value + " " + random.choice(['số', 'tầng', '']) + " " + str(
                        random.randint(0,
                                       10))
                prefix = random.choice(possible_confusion_words)
                sentence = sentence[:sentence_end] + " " + prefix + " " + new_slot_value + " " + sentence[sentence_end:]
                sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                    slot)] + current_slot_value + " " + prefix + " " + template.format(
                    new_slot_value) + " " + sentence_annotation[annotation_end + 2:]
                added = True

            elif slot == "changing value" or slot == "target number" and not added and add < rate:
                template = f"[ {slot}" + " : {} ]"
                current_slot_value = sentence[sentence_start:sentence_end]
                number, unit, unit_index = find_unit_in_number(current_slot_value)
                # if unit != "" and random.uniform(0, 1) < 0.5:  # delete unit from slot
                sentence = sentence[:sentence_start + unit_index] + sentence[sentence_end:]
                sentence_annotation = sentence_annotation[:annotation_start + unit_index] + sentence_annotation[
                                                                                            annotation_end:]
                try:
                    number = int(number)
                except:
                    print(current_slot_value)
                    print(number)
                    print(unit)
                    print(unit_index)
                    print(sentence, sentence_annotation)
                    continue
                if random.uniform(0, 1) < 0.5:  # minus
                    random_minus = random.randint(1, 5)
                    if random_minus > number:
                        new_number = number + 1
                    else:
                        new_number = number - random_minus
                else:
                    random_add = random.randint(1, 5)
                    new_number = number + random_add
                new_slot_value = str(new_number) + " " + unit
                prefix = random.choice(possible_confusion_words)

                sentence = sentence[
                           :sentence_start + unit_index] + " " + prefix + " " + new_slot_value + " " + sentence[
                                                                                                       sentence_start + unit_index:]
                sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                    slot)] + sentence_annotation[
                             annotation_start:annotation_start + unit_index] + " " + prefix + " " + template.format(
                    new_slot_value) + " " + sentence_annotation[annotation_start + unit_index + 2:]

                added = True
            elif slot == 'time at' or slot == 'duration' and not added and add < rate:
                template = f"[ {slot}" + " : {} ]"
                prefix = random.choice(possible_confusion_words)
                current_slot_value = sentence[sentence_start:sentence_end]
                new_slot_value = generate_time(time_at=slot == 'time at')
                sentence = sentence[:sentence_end] + " " + prefix + " " + new_slot_value + " " + sentence[sentence_end:]
                sentence_annotation = sentence_annotation[:annotation_start - 5 - len(
                    slot)] + current_slot_value + " " + prefix + " " + template.format(
                    new_slot_value) + " " + sentence_annotation[annotation_end + 2:]
                added = True
            if added:
                break

        if added:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(sentence)
                else:
                    new_sample[k].append(sentence_annotation)
    return new_sample


def find_unit_in_number(number_slot):
    unit_index = None
    for i, c in enumerate(number_slot):
        if not c.isdigit():
            unit_index = i
            break
    if unit_index is not None:
        return number_slot[:unit_index].strip(), number_slot[unit_index:].strip(), unit_index
    else:
        return number_slot, "", len(number_slot)


def random_scene_aug(sample, prob=.7):
    new_sample = sample.copy()
    count = 0
    for i in range(len(sample['sentence'])):
        intent = sample['intent'][i]
        if intent.lower() in ['bật thiết bị', 'tắt thiết bị'] and random.random() < prob:
            sentence = sample['sentence'][i]

            # Use re.sub() to remove the surrounding [ command : ] and keep only the <txt>
            remove_command_pattern = r'\[ command : (.*?) \]'
            sentence_annotation = sample['sentence_annotation'][i]
            sentence_annotation = re.sub(remove_command_pattern, r'\1', sentence_annotation)

            indices = find_annotation_indices(sentence, sentence_annotation)
            new_sentence = None
            new_sentence_annotation = None
            for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
                if slot == "device":  # found a device slot
                    new_slot_value = random.choice(scenes)
                    slot_prefix = random.choice(["chế độ ", "không khí ", "sự ", "hoạt cảnh "])

                    new_sentence = sentence[:sentence_start] + slot_prefix + new_slot_value + sentence[sentence_end:]

                    new_sentence_annotation = sentence_annotation[:annotation_start - len('[ device : ')] \
                                              + slot_prefix + f'[ scene : {new_slot_value} ]' + \
                                              sentence_annotation[annotation_end + 2:]
            if new_sentence is not None and new_sentence_annotation is not None:
                count += 1
                for k in new_sample.keys():
                    if k not in ['sentence', 'sentence_annotation', 'intent']:
                        new_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        new_sample[k].append(new_sentence)
                    elif k == 'sentence_annotation':
                        new_sample[k].append(new_sentence_annotation)
                    else:
                        if intent.lower() == "bật thiết bị":
                            new_intent = "kích hoạt cảnh"
                        else:
                            new_intent = "hủy hoạt cảnh"
                        new_sample[k].append(new_intent)
    return new_sample


def generate_yes_no(sample, prob=0.7):
    new_sample = sample.copy()
    for i in range(len(sample['sentence'])):
        intent = random.choice(list(possible_intent_command_mapping.keys()))
        if intent in ['kích hoạt cảnh', 'hủy hoạt cảnh', 'kiểm tra tình trạng thiết bị']:
            continue
        if random.random() < 0.7:
            new_intent = opposite_intent_mapping[intent]
            if new_intent is None:
                continue
        else:
            new_intent = random.choice(neutral_intent)
        include_postfix = random.random() < 0.5
        prefix, type, subject, annotation = create_prefix(include_postfix=include_postfix)
        middle, label = create_middle(intent, include_postfix=include_postfix)
        if include_postfix:
            post, label_post, type = create_postfix(new_intent, subject=subject, type=type)
        else:
            post, label_post, type = '', '', 2
        new_sentence = prefix + " " + middle + " " + post
        new_sentence_annotation = annotation + " " + label + " " + label_post
        if type == 2:
            new_intent = 'kiểm tra tình trạng thiết bị'
        if random.random() < prob:
            for k in new_sample.keys():
                if k not in ['sentence', 'sentence_annotation', 'intent']:
                    new_sample[k].append(sample[k][i])
                elif k == 'sentence':
                    new_sample[k].append(new_sentence)
                elif k == 'sentence_annotation':
                    new_sample[k].append(new_sentence_annotation)
                elif k == 'intent':
                    new_sample[k].append(new_intent)
    return new_sample


def reverse_intent(sample, prob=.7):
    reversed_sample = sample.copy()
    for i in range(len(reversed_sample['sentence'])):
        intent = reversed_sample['intent'][i]
        if intent.lower() in reversed_intent_mapping and random.random() < prob:
            sentence_annotation = reversed_sample['sentence_annotation'][i]
            sentence = reversed_sample['sentence'][i]
            indices = find_annotation_indices(sentence, sentence_annotation)
            reversed_sentence = None
            reversed_annotation = None
            for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
                if slot == 'command':
                    # reversed_sample['intent'][i] = reversed_intent_mapping[intent]
                    orig_command = sentence[sentence_start:sentence_end]
                    reversed_command = f"{random.choice(reversed_command_prefix)} {orig_command}"

                    reversed_sentence = sentence[:sentence_start] + reversed_command + sentence[sentence_end:]
                    reversed_annotation = sentence_annotation[
                                          :annotation_start - len('[ command : ')] + reversed_command \
                                          + sentence_annotation[annotation_end + len(' ]'):]

            if reversed_sentence is not None and reversed_annotation is not None:
                for k in reversed_sample.keys():
                    if k not in ['sentence', 'sentence_annotation', 'intent']:
                        reversed_sample[k].append(sample[k][i])
                    elif k == 'sentence':
                        reversed_sample[k].append(reversed_sentence)
                    elif k == 'sentence_annotation':
                        reversed_sample[k].append(reversed_annotation)
                    elif k == 'intent':
                        reversed_sample[k].append(reversed_intent_mapping[intent])
    return reversed_sample



if __name__ == "__main__":
    from process_data_bio import process_sample

    dataset = load_dataset('json', data_files='/home/chaunm/workspace/SLU/IDSF/data/train_final_20230919.jsonl',
                           split='train')
    dataset = dataset.map(refine_dataset, batched=True, load_from_cache_file=False)
    dataset = dataset.map(refine_slot_label_changing_value, batched=True, load_from_cache_file=False
                          )
    dataset = dataset.map(strip_spaces, batched=True)

    # dataset = dataset.filter(lambda x: x['intent'] == "Bật thiết bị")
    # dataset = dataset.map(add_location, batched=True, load_from_cache_file=False)
    # dataset = dataset.map(strip_spaces, batched=True)
    # {'context': ['nóng', 'quá', 'đừng', 'có', 'mở', 'cho', 'anh', 'cái', 'điều', 'hòa', 'ở', 'tầng', 'hầm', 'nhớ'], 'slot_label': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_device', 'I_device', 'O', 'B_location', 'I_location', 'O'], 'intent_label': 14, 'audio_file': '648710731ae6761d9db63ca3.wav', 'original_sentence': 'nóng quá đừng có mở cho anh cái điều hòa ở tầng hầm nhớ', 'original_sentence_annotation': 'nóng quá đừng có mở cho anh cái [ device : điều hòa ] ở [ location : tầng hầm ] nhớ'}

    # dataset = dataset.map(partial(add_duration, times=1), batched=True, load_from_cache_file=False)
    # dataset = dataset.map(strip_spaces, batched=True)
    #
    # dataset = dataset.map(random_scene_aug, batched=True, load_from_cache_file=False)
    # dataset = dataset.map(strip_spaces, batched=True)
    # dataset = dataset.map(add_confusing_device, batched=True, load_from_cache_file=False)
    # dataset = dataset.map(strip_spaces, batched=True)
    # dataset = dataset.map(add_confusing_slot, batched=True)
    # dataset = dataset.map(strip_spaces, batched=True)
    #
    # # dataset = dataset.map(random_scene_aug, batched=True)
    # dataset = dataset.map(random_change_command, batched=True)
    # dataset = dataset.map(random_change_number, batched=True)
    # dataset = dataset.map(random_change_time_at, batched=True)
    # dataset = dataset.map(random_change_duration, batched=True)
    # dataset = dataset.map(random_change_location, batched=True)
    # dataset = dataset.map(generate_yes_no, batched=True)
    dataset = dataset.map(add_time_at, batched=True)
    dataset = dataset.map(strip_spaces, batched=True)
    dataset = dataset.map(reverse_intent, batched=True)
    dataset = dataset.map(process_sample, batched=True,
                          remove_columns=['id', 'sentence', 'intent', 'sentence_annotation', 'entities',
                                          'file'],
                          load_from_cache_file=False)
    print(len(dataset))
    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        annotation = sample['slot_label']
        is_invalid = False
        for j, s in enumerate(annotation):
            if s.startswith("I"):
                if j >= 1 and annotation[j - 1].startswith("O"):
                    is_invalid = True
                elif j >= 1 and annotation[j - 1].startswith("B") and annotation[j - 1].split("_")[-1] != s.split("_")[
                    -1]:
                    is_invalid = True

        if is_invalid:
            print(sample)
            count += 1
            print(" ")

    print(count)

    dataset = dataset.shuffle()

    subset = dataset.filter(lambda x: 'đừng' in x['context'])
    print(len(subset))
    for i, sample in enumerate(subset):
        print(sample)
        print("\n")
        if i >= 10:
            break
