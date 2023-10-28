import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--zip_file_path', type=str, help="Path of zip file")
parser.add_argument('--output_dir', type=str,  help="Output directory", default = 'speech_modules/data/')
args = parser.parse_args()

# zip_file_path = 'speech_modules/data/original_data/SLU/train_data.zip'
with zipfile.ZipFile(args.zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(args.output_dir)

# zip_file_path = 'speech_modules/data/original_data/SLU/public_test.zip'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall('speech_modules/data')


# zip_file_path = 'speech_modules/data/ESC 50 Data/ESC-50-master.zip'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall('speech_modules/data/ESC 50 Data')