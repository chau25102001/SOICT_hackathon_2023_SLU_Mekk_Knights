import zipfile
import argparse
import tarfile
parser = argparse.ArgumentParser()
parser.add_argument('--rar_file_path', type=str, help="Path of rar file")
parser.add_argument('--output_dir', type=str,  help="Output directory", default = 'speech_modules/data/')
args = parser.parse_args()

# zip_file_path = 'speech_modules/data/original_data/SLU/train_data.zip'
tar = tarfile.open(args.rar_file_path, "r:gz")
tar.extractall(path=args.output_dir)
tar.close()
# zip_file_path = 'speech_modules/data/original_data/SLU/public_test.zip'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall('speech_modules/data')


# zip_file_path = 'speech_modules/data/ESC 50 Data/ESC-50-master.zip'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall('speech_modules/data/ESC 50 Data')