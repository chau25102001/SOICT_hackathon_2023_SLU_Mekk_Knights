import gdown
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--download_url', type=str, help="URL of download file")
parser.add_argument('--output_dir', type=str,  help="Download path", default = 'speech_modules/data/')
args = parser.parse_args()

gdown.download(args.download_url, output = args.output_dir, quiet=True, fuzzy = True)

# print('-------- Downloading original dataset --------')
# url = 'https://drive.google.com/drive/folders/1FqCmmSjMMgkYjANXY7FD6tzqsfDwZJrY?usp=drive_link'
# gdown.download_folder(url, output = 'speech_modules/data/original_data/', quiet=True, remaining_ok=True, use_cookies=False)


# print('-------- Downloading noise dataset ESC-50 --------')
# url = 'https://drive.google.com/drive/folders/1cbMd7NPwKhynOEDSMBVuisD8g-xDtHr5?usp=sharing'
# # gdown.download_folder(url, output = 'data/slu_data/', quiet=True, remaining_ok=True, use_cookies=False)
# gdown.download_folder(url, output = 'speech_modules/data/', quiet=True, remaining_ok=True, use_cookies=False)
