import re
import random
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
import string 

import soundfile as sf

from .hifigan.mel2wave import mel2wave
from .nat.config import FLAGS
from .nat.text2mel import text2mel
import json
import os
import random
from datasets import load_from_disk

intent_mapping = { 'bật thiết bị': 0,
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


def nat_normalize_text(text):
	text = unicodedata.normalize('NFKC', text)
	text = text.lower().strip()
	sp = FLAGS.special_phonemes[FLAGS.sp_index]
	text = re.sub(r'[\n.,:]+', f' {sp} ', text)
	text = text.replace('"', " ")
	text = re.sub(r'\s+', ' ', text)
	text = re.sub(r'[.,:;?!]+', f' {sp} ', text)
	text = re.sub('[ ]+', ' ', text)
	text = re.sub(f'( {sp}+)+ ', f' {sp} ', text)
	return text.strip()


# text = nat_normalize_text(args.text)
# print('Normalized text input:', text)
# mel = text2mel(text, args.lexicon_file, args.silence_duration, speaker=args.speaker)
# wave = mel2wave(mel)
# print('writing output to file', args.output)
# sf.write(str(args.output), wave, samplerate=args.sample_rate)

PATTERN = r'\d+'
BASIC = {0: 'không', 1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}
random.seed(42)

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
				middle  = ' mươi'
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


def stutter_augmentation(context: list, slot_label: list):
	stutter_or_not, stutter_non_slot = 0.85, 0.95
	
	augmented_sentence = ''
	if random.random() > stutter_or_not:
		for word_id in range(len(context)-1):
			if (("B" in slot_label[word_id+1]) and random.random() > 0.3) or ("O" == slot_label[word_id] and random.random() > stutter_non_slot):
				stutter_word = num_convert(context[word_id])
				stutter_word += stutter_word[-1]*3 + " spn " if(random.random() > 0.3) else " "
				augmented_sentence += stutter_word
			else:
				augmented_sentence += f'{num_convert(context[word_id])} '
		augmented_sentence += num_convert(context[-1])
		augmented_sentence = augmented_sentence.strip()
	
	if len(augmented_sentence) > 0:
		return augmented_sentence
	return num_convert(" ".join(context))

def remove_punctuations_and_spaces(text):
	text = text.translate(str.maketrans("", "", string.punctuation))
	text = " ".join(text.split()).lower()
	text = text.replace('"', '')
	try:
		if text[-1] not in [".", "?", "!"]:
			text = text + "."
	except:
		pass
	return text

def loanwords_subs(text):
	dct = {
		"champagne": "sâm banh",
		"steak": "sờ tếch",
		"beefsteak": "bít tết",
		"beef steak": "bít tết",
		"beer": "bia",
		"olive": "ô liu",
		"sauce": "sốt",
		"salad": "xa lát",
		"sandwich": "xăng quých",
		"soda": "sô đa",
		"yogurt": "da ua",
		"menu": "me nu",
		"radio": "ra đi ô",
		"camera": "cam mê ra",
		"internet": "in tơ nét",
		"laptop": "láp tốp",
		"email": "i meo",
		"cinema": "xi nê",
		"fax": "phách",
		"cafe": "cà phê",
		"garage": "ga ra",
		"compact": "com pắc",
		"wc": "vê kép xê",
		"gym": "dim",
		"game": "gêm",
		"led": "lét"
	}
	for key, value in dct.items():
		text = text.replace(key, value)
	return text

if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("-ip", "--input_path", default="data_bio/processed_train", type=str, help="input huggingface format dataset")
	parser.add_argument("-op", "--output_folder", default="/Train_tts", type=str, help="output data folder")
	args = parser.parse_args()

	# input_path = "data_bio/processed_train"
	lexicon_file = "assets/infore/lexicon.txt"
	silence_duration = random.random()*1.5
	os.makedirs(f"{args.output_folder}", exist_ok=True)
	os.makedirs(f"{args.output_folder}/audio", exist_ok=True)

	# for key in intent_mapping.keys():
	# 	os.makedirs(f'{args.output_folder}/{intent_mapping[key]}', exist_ok=True)
	
	data = load_from_disk(args.input_path)

	for sent_idx in range(len(data)):
		speaker = random.randint(0, 256)
		context = data[sent_idx]['context']
		slot_label = data[sent_idx]['slot_label']

		augmented_sentence = loanwords_subs(nat_normalize_text(stutter_augmentation(context, slot_label)))

		mel = text2mel(augmented_sentence, lexicon_file, silence_duration, speaker=speaker)
		wave = mel2wave(mel)

		prefix, postfix = data[sent_idx]['audio_file'].split(".")
		file_name = f'{speaker}_{prefix}_{sent_idx}'

		output = f"{args.output_folder}/audio/{file_name}.{postfix}"

		print('writing output to file', output)
		sf.write(str(output), wave, samplerate=16000)

		annotation = {"id": file_name, 'sentence': remove_punctuations_and_spaces(num_convert(" ".join(context))), 'augmented_sentence':augmented_sentence, 'speaker':speaker}
		
		with open(f"{args.output_folder}/stt_annotation.jsonl", "a") as f:
			f.write(f'{json.dumps(annotation, ensure_ascii=False)}\n')
