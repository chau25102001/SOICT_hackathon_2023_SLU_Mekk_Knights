import typing
import re
from datasets import load_dataset
from constants import *

dataset = load_dataset('json', data_files='train_final_20230919.jsonl', split='train')
van_file = open("van", 'r')
lines = van_file.readlines()
list_van = []
# print("a".isnumeric())
for l in lines:
    l = l.replace("\n", '')
    # print(l.i())
    if not l.isnumeric():
        list_van.append(l.strip())

list_non_vowel = ["b", 'c', 'd', 'đ', 'g', 'h', 'l', 'm', 'n', 'ph', 'r', 's', 't', 'v', 'x', 'ch', 'tr', '']
list_accent = ['s', 'f', 'x', 'j', 'r', '']

vowel_with_accent_to_vowel_mapping = {
    "a": "a_",
    "á": "a_s",
    "à": "a_f",
    "ả": "a_r",
    "ạ": "a_j",
    "ã": "a_x",

    "ă": "ă_",
    "ắ": "ă_s",
    "ằ": "ă_f",
    "ẳ": "ă_r",
    "ặ": "ă_j",
    "ẵ": "ă_x",

    "â": "â_",
    "ấ": "â_s",
    "ầ": "â_f",
    "ẩ": "â_r",
    "ậ": "â_j",
    "ẫ": "â_x",

    "e": "e_",
    "é": "e_s",
    "è": "e_f",
    "ẻ": "e_r",
    "ẹ": "e_j",
    "ẽ": "e_x",

    "ê": "ê_",
    "ế": "ê_s",
    "ề": "ê_f",
    "ể": "ê_r",
    "ệ": "ê_j",
    "ễ": "ê_x",

    "i": "i_",
    "í": "i_s",
    "ì": "i_f",
    "ỉ": "i_r",
    "ị": "i_j",
    "ĩ": "i_x",

    "o": "o_",
    "ó": "o_s",
    "ò": "o_f",
    "ỏ": "o_r",
    "ọ": "o_j",
    "õ": "o_x",

    "ô": "ô_",
    "ố": "ô_s",
    "ồ": "ô_f",
    "ổ": "ô_r",
    "ộ": "ô_j",
    "ỗ": "ô_x",

    "ơ": "ơ_",
    "ớ": "ơ_s",
    "ờ": "ơ_f",
    "ở": "ơ_r",
    "ợ": "ơ_j",
    "ỡ": "ơ_x",

    "u": "u_",
    "ú": "u_s",
    "ù": "u_f",
    "ủ": "u_r",
    "ụ": "u_j",
    "ũ": "u_x",

    "ư": "ư_",
    "ứ": "ư_s",
    "ừ": "ư_f",
    "ử": "ư_r",
    "ự": "ư_x",
    "ữ": "ư_j",

    "y": "y_",
    "ý": "y_s",
    "ỳ": "y_f",
    "ỷ": "y_r",
    "ỵ": "y_j",
    "ỹ": "y_x",
}

vowel_to_accent_mapping = {v: k for k, v in vowel_with_accent_to_vowel_mapping.items()}


def extract_van_and_accent_from_word(word):
    for i, c in enumerate(word):
        if i >= 1 and word[i - 1:i + 1] in ['gi', 'qu']:
            continue
        if c in vowel_with_accent_to_vowel_mapping:
            return_word = []
            accent = ''
            non_vowel = word[:i]
            for cm in word[i:]:
                if cm in vowel_with_accent_to_vowel_mapping:
                    cm_accent = vowel_with_accent_to_vowel_mapping[cm]
                    cm, accent = cm_accent.split("_")

                if cm.isalpha():
                    return_word.append(cm)
            return non_vowel, ''.join(return_word), accent
    return None, None, None


def valid_van(van):
    sep = False
    valid = True
    for i, c in enumerate(van):
        if c not in vowel_with_accent_to_vowel_mapping:  # meet a non-vowel
            sep = True
        if sep and c in vowel_with_accent_to_vowel_mapping:
            valid = False
        if c in ['z', 'k', 'j', 'x', 'v', 'b', 'q', 'w', 'r', 's', 'd', 'f', 'l']:
            valid = False
    if van.startswith("y") and len(van) >= 3:
        valid = False
    return valid


# get the van list
for s in dataset:
    sentence = s['sentence']
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("/", " / ")
    sentence = sentence.replace("!", " ! ")
    while "  " in sentence:
        sentence = sentence.replace("  ", " ")
    sentence_words: typing.List[str] = sentence.split(" ")
    for w in sentence_words:
        nonvowel, van, accent = extract_van_and_accent_from_word(w.lower())
        if van is not None and valid_van(van.strip().lower()):
            list_van.append(van.strip().lower())


def create_word(nonvowel, van, accent):
    count_vowel = 0
    accent_index = 0
    for i, c in enumerate(van):
        if c in vowel_with_accent_to_vowel_mapping:  # found a vowel
            count_vowel += 1
            accent_index = i
        if count_vowel == 2:
            break
    if count_vowel == 2 and len(van) == 2:
        accent_index -= 1
    try:
        van = van[:accent_index] + vowel_to_accent_mapping[van[accent_index] + "_" + accent] + van[accent_index + 1:]
    except:
        print(van, type(van), accent_index)
    return nonvowel + van


list_eng_vowel = ['u', 'e', 'o', 'a', 'i']


def augment_word(word):
    '''
    :param word: str: vietnamese word, should not contain non-alphabetic character
    :return: augmented word
    '''
    if not word.isalpha():
        return word
    nonvowel, van, accent = extract_van_and_accent_from_word(word)
    if van is None:
        return word
    van = van.lower()
    if not valid_van(van) and accent == '':  # not a vietnamese van
        new_van = []
        for v in van:
            if v not in list_eng_vowel:
                new_van.append(v)
            else:
                if random.uniform(0, 1) < 0.5:
                    new_v = ''
                    while True:
                        new_v = random.choice(list_eng_vowel)
                        if new_v != v:
                            break
                    new_van.append(new_v)
                else:
                    new_van.append(v)
        new_van = ''.join(new_van)
        return create_word(nonvowel, new_van, accent)

    else:
        augment_choice = random.uniform(0, 1)
        new_nonvowel = None
        while True:
            new_nonvowel = random.choice(list_non_vowel)
            if new_nonvowel != nonvowel:
                break

        new_van = None
        while True:
            new_van = random.choice(list_van)
            if new_van != van:
                break

        new_accent = None
        while True:
            new_accent = random.choice(list_accent)
            if new_accent != accent:
                break

        if augment_choice < 0.7:  # augment 1 component
            index = random.sample([0, 1, 2], k=1)
        else:
            index = random.sample([0, 1, 2], k=2)
            if accent == "":
                index = random.choice([[0, 2], [1, 2]])

        if 0 in index:  # augment non vowel
            nonvowel = new_nonvowel
        if 1 in index:  # augment van
            van = new_van
        if 2 in index:  # augment accent
            accent = new_accent
        new_word = create_word(nonvowel, van, accent)
        return new_word


PATTERN = r'\d+'
BASIC = {0: 'không', 1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín",
         10: "mười"}


def num_to_text(num: int):
    if num in BASIC:
        return BASIC[num]

    chuc = num // 10
    donvi = num % 10
    if chuc == 1:
        return "mười " + BASIC[donvi]
    else:
        first = BASIC[chuc]
        prob = random.uniform(0, 1)
        if prob < 0.5:
            middle = " "
        else:
            middle = " mươi "
        if donvi == 4:
            another_prob = random.uniform(0, 1)
            if another_prob < 0.5:
                final = "bốn"
            else:
                final = "tư"
        elif donvi == 1:
            final = "mốt"
        elif donvi == 5:
            final = 'lăm'
        elif donvi == 0:
            if middle == ' mươi ':
                final = ''
                middel = ' mươi'
            else:
                final = 'mươi'
        else:
            final = BASIC[donvi]
        return first + middle + final


def num_convert(sentence):
    match = re.finditer(PATTERN, sentence)
    lech = 0

    for something in match:
        start, end = something.span()
        # print(start, end)
        word = sentence[start + lech:end + lech]

        num = int(word)
        text_num = num_to_text(num)
        sentence = sentence.replace(word, text_num, 1)
        lech += len(text_num) - len(word)
    sentence = sentence.replace("%", " phần trăm")

    return sentence


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
                if random.uniform(0, 1) < 0.3:
                    new_slot_value = new_slot_value + " của " + random.choice(human_names)
                if random.uniform(0, 1) < 0.3:
                    new_slot_value = new_slot_value + " " + random.choice(directions)
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
    rate = 0.3
    for i, _ in enumerate(sample['sentence']):
        intent = sample['intent'][i]
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        device = ''
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == 'device':
                device = sentence[sentence_start: sentence_end]
                break
        additional_command = []
        if intent == 'đóng thiết bị':
            for key in device_keyword_command_mapping_dong_thiet_bi:
                if key in device:
                    additional_command = device_keyword_command_mapping_dong_thiet_bi[key]
                    break
        elif intent == 'mở thiết bị':
            for key in device_keyword_command_mapping_mo_thiet_bi:
                if key in device:
                    additional_command = device_keyword_command_mapping_mo_thiet_bi[key]
                    break
        elif intent == 'bật thiết bị':
            for key in device_keyword_command_mapping_bat_thiet_bi:
                if key in device:
                    additional_command = device_keyword_command_mapping_bat_thiet_bi[key]
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == "command" and random.uniform(0, 1) < rate:  # found a device slot
                current_slot_value = sentence[sentence_start:sentence_end]
                command_choices = possible_intent_command_mapping[intent.lower().strip()] + additional_command
                if ('tăng' in intent or 'giảm' in intent) and ('lên' in sentence or 'xuống' in sentence):
                    command_choices.extend(['chỉnh', 'điều chỉnh'])
                if len(command_choices) == 0:
                    break
                # while True:  # choose a new device to replace
                new_slot_value = random.choice(command_choices)
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
                prefix = ''
                postfix = ''
                if value_type == 0:
                    rate = random.uniform(0, 1)
                    if rate < 0.33:  # percentage
                        new_slot_value = str(random.randint(10, 90)) + "%"
                    elif rate < 0.88:  # temperature
                        new_slot_value = str(random.randint(20, 25)) + random.choice([" độ C", " độ"])
                    else:  # level
                        new_slot_value = str(random.randint(0, 10))
                        if slot == 'target number':
                            prefix = random.choice([' mức ', ' nấc ', ''])
                            postfix = random.choice([' mức ', ' nấc ', ''])
                elif value_type != 1:
                    rate = random.uniform(0, 1)
                    if rate < 0.5:
                        new_slot_value = str(random.randint(10, 90)) + "%"
                    else:
                        new_slot_value = str(random.randint(1, 10))
                        if slot == 'target number':
                            prefix = random.choice([' mức ', ' nấc ', ''])
                            postfix = random.choice([' mức ', ' nấc ', ''])
                else:
                    new_slot_value = str(random.randint(1, 10))
                    if slot == 'target number':
                        prefix = random.choice([' mức ', ' nấc ', ''])
                        postfix = random.choice([' mức ', ' nấc ', ''])
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
                if not found_digits and 'của' not in new_slot_value and 'số' not in new_slot_value and 'tầng' not in new_slot_value:
                    if random.uniform(
                            0, 1) < 0.3:
                        new_slot_value = new_slot_value + " " + random.choice(['số', 'tầng', '']) + " " + str(
                            random.randint(0,
                                           10))
                    if random.uniform(0, 1) < 0.3:
                        new_slot_value = new_slot_value + ' của ' + random.choice(human_names)
                    if random.uniform(0, 1) < 0.3:
                        new_slot_value = new_slot_value + " " + random.choice(directions)

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
                        prefix = random.choice(["trong vòng ", "trong ", "khoảng ", "trong khoảng "])
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
                        prefix = random.choice(["trong vòng ", "trong ", "khoảng ", "trong khoảng "])
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
                new_slot_value = generate_time(time_at=slot=='time at')
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


def generate_corrupted_dataset(sample, num_augment=2):
    new_sample = {}
    new_sample['sentence_source'] = []
    new_sample['sentence_target'] = []
    new_sample['file'] = []
    for time in range(num_augment):
        for i, _ in enumerate(sample['sentence']):
            sentence = sample['sentence'][i]
            try:
                sentence = num_convert(sentence)  # normalize text
            except:
                continue
            sentence_words = sentence.split(" ")
            word_indices = [idx for idx in range(len(sentence_words)) if sentence_words[idx].isalpha()]
            num_augment = random.uniform(0.15, 0.3)
            augment_index = random.sample(word_indices,
                                          k=min(max(int(num_augment * len(word_indices)), 2), len(word_indices)))
            for ai in augment_index:
                sentence_words[ai] = augment_word(sentence_words[ai])
            new_sample['sentence_source'].append(" ".join(sentence_words))
            new_sample['sentence_target'].append(sentence)
            new_sample['file'].append(sample['file'][i])
    # for k, v in new_sample.items():
    #     print(k, len(v), v[-1])
    return new_sample


def generate_clean_dataset(sample, num_augment=1):
    new_sample = {}
    new_sample['sentence_source'] = []
    new_sample['sentence_target'] = []
    new_sample['file'] = []
    for time in range(num_augment):
        for i, _ in enumerate(sample['sentence']):
            sentence = sample['sentence'][i]
            try:
                sentence = num_convert(sentence)
            except:
                print(sentence)
                continue
            new_sample['sentence_source'].append(sentence)
            new_sample['sentence_target'].append(sentence)
            new_sample['file'].append(sample['file'][i])
    return new_sample


def clean_command(sample):
    new_sample = sample.copy()
    for i, _ in enumerate(sample['sentence']):
        sentence = sample['sentence'][i]
        sentence_annotation = sample['sentence_annotation'][i]
        indices = find_annotation_indices(sentence, sentence_annotation)
        for slot, sentence_start, sentence_end, annotation_start, annotation_end in indices:
            if slot == 'command':
                current_slot = sentence[sentence_start:sentence_end]
                if current_slot in bad_command:
                    sentence_annotation = sentence_annotation[
                                          :annotation_start - len('[ command : ')] + current_slot \
                                          + sentence_annotation[annotation_end + len(' ]'):]
                    current_slot = ''
                found_special_command = False
                for c in special_command:
                    if c in current_slot:
                        found_special_command = True
                        break
                if found_special_command and len(current_slot.split(" ")) > 1:
                    offset = len(current_slot.split(" ")[0])
                    sentence_annotation = sentence_annotation[:annotation_start] + current_slot[
                                                                                   :offset] + " ]" + current_slot[
                                                                                                     offset:] + sentence_annotation[
                                                                                                                annotation_end + 2:]
        new_sample['sentence_annotation'][i] = sentence_annotation
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
        new_sentence = new_sentence.strip()
        new_sentence_annotation = new_sentence_annotation.strip()
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
