import string
import re
import random
import json
import logging 
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

PATTERN = r'\d+'
BASIC = {0: 'không', 1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}
random.seed(42)

def num_to_text(num: int):
    if num <= 100:
        return two_digits_num_to_text(num)
    else:
        tram = num // 100
        two_digit_part  = num - tram * 100
        print(two_digit_part)
        return BASIC[tram] + ' trăm ' + two_digits_num_to_text(two_digit_part)


def two_digits_num_to_text(num: int):
    """Convert number with 2 or"""
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
            another_prob = random.uniform(0,1)
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
            middel  = ' mươi'
          else:
            final = 'mươi'
        else: final = BASIC[donvi]
        return first + middle + final

def num_convert(sentence):

    match = re.finditer(PATTERN, sentence)
    lech = 0

    for something in match:

        start, end = something.span()
        # print(start, end)
        word = sentence[start+lech:end+lech]

        num = int(word)
        text_num = num_to_text(num)
        sentence = sentence.replace(word, text_num, 1)
        lech += len(text_num) - len(word)
    sentence = sentence.replace("%", " phần trăm")

    return sentence

def normalize_sentence(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split()).lower()
    text = text.replace('"', '')
    try:
        if text[-1] not in [".", "?", "!"]:
            text = text + "."
    except:
        pass
    return num_convert(text)

if __name__ == '__main__':
    logging.info('-------- Normalize text data for ASR training --------')
    # print(normalize_sentence('tôi có 260 đô'))
    for i in range(1000):
        print(i)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, help="input path of annotation file", default = 'speech_modules/data/original_data/SLU/train_final_20230919.jsonl')
    # parser.add_argument('--output_dir', type=str,  help="output file", default = 'speech_modules/data/original_data/train_normalized_20230919.jsonl')
    # args = parser.parse_args()

    # input_dir = args.input_dir
    # output_dir = args.output_dir

    # with open(input_dir, 'r') as json_file:
    #     json_list = list(json_file)
    # annotation_data = []
    # annotation_id = []
    # annotation_text = []
    # for json_str in json_list:
    #     result = json.loads(json_str)
    #     annotation_data.append(result)
    #     annotation_id.append(result['id'])
    #     annotation_text.append(result['sentence'])

    # normalized_data = []
    # for row in annotation_data:
    #     id,sentence,intent,entities,annotation, file_name = row['id'], row['sentence'], row['intent'], row['entities'], row['sentence_annotation'], row['file']
    #     sentence = remove_punctuations_and_spaces(num_convert(sentence))
    #     normalized_entities = []
    #     for entity in entities:
    #         # print(entity)
    #         normalized_entity = {'type':entity['type'],'filler': num_convert(entity['filler'])}
    #         normalized_entities.append(normalized_entity)
    #     annotation = num_convert(annotation)
    #     normalized_data.append({'id':id, 'sentence':sentence,'sentence_annotation':annotation,'entities':normalized_entities,'file':file_name})

    # with open(output_dir, 'w') as outfile:
    #     for entry in normalized_data:
    #         json.dump(entry, outfile, ensure_ascii = False)
    #         outfile.write('\n')