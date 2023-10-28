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
    all_spellings = []
    if num <= 100:
        all_spellings.extend(two_digits_num_to_text(num))
    else:
        tram = num // 100
        two_digit_part  = num - tram * 100
        if two_digit_part == 0:
            # print(two_digit_part)
            all_spellings.append(BASIC[tram] + ' trăm')
        elif two_digit_part < 10:
            two_digit_part_spellings = two_digits_num_to_text(two_digit_part)
            for spelling in two_digit_part_spellings:
            
                all_spellings.append(BASIC[tram] + ' trăm ' + 'linh ' + spelling)
                all_spellings.append(BASIC[tram] + ' trăm ' + 'lẻ ' + spelling)
                all_spellings.append(BASIC[tram] + ' lẻ ' + spelling)

                if spelling == 'một':
                    all_spellings.append(BASIC[tram] + ' trăm ' + 'linh ' + 'mốt')
                    all_spellings.append(BASIC[tram] + ' trăm ' + 'lẻ ' + 'mốt')
                    all_spellings.append(BASIC[tram] + ' lẻ ' + 'mốt')


        else:
            two_digit_part_spellings = two_digits_num_to_text(two_digit_part)
            for spelling in two_digit_part_spellings:
                all_spellings.append(BASIC[tram] + ' trăm ' + spelling)

    
    return all_spellings


def two_digits_num_to_text(num: int):
    """Convert number with 2 or"""
    two_digit_spellings = []
    if num in BASIC:
        print(num)
        two_digit_spellings.append(BASIC[num])
        return two_digit_spellings
    

    chuc = num // 10
    donvi = num % 10
    if chuc == 1:
        two_digit_spellings.append("mười " + BASIC[donvi])
        if donvi == 5:
            two_digit_spellings.append("mười " + 'lăm')

        return two_digit_spellings
    else:
        first = BASIC[chuc]
        middles = [" ", ' mươi ']
        if donvi == 4:
            finals = ['bốn' , 'tư']
        elif donvi == 1:
            finals = ["mốt"]
        elif donvi == 5:
            finals = ['lăm','năm']
        elif donvi == 0:
                finals = [ 'mươi']
        else: 
            finals  = [BASIC[donvi]]
        
        for middle in middles:
            for final in finals:
                spelling = first + middle + final
                if 'mươi mươi' in spelling:
                    spelling = spelling.replace('mươi mươi','mươi')
                # print("Final", final)
                two_digit_spellings.append(spelling)
                # print(two_digit_spellings)
        
        return two_digit_spellings

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
    all_spellings = {}
    for i in range(1000):
        all_spellings_i = num_to_text(i)
        # print(all_spellings_i)
        for spelling in all_spellings_i:
            # print(spelling,i)
            all_spellings[spelling.strip()] = str(i)
    # print(all_spellings)
    with open('speech_modules/text_to_num.json','w') as outfile:
        json.dump(all_spellings, outfile, ensure_ascii = False)
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