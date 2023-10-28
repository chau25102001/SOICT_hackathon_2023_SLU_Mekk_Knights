if [ ! -f assets/infore/hifigan/g_01135000 ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p assets/infore/{nat,hifigan}
  gdown --id 1zh6axidF1H9TAz9t45IrSe_mQ8KZMgNr -O assets/infore/nat/nat_ckpt_latest.pickle
  gdown --id 1-NTQc4RDK-DgBIQVk-t4dgtYI1MUbBIk -O assets/infore/hifigan/g_01135000
  python3 -m vietTTS.hifigan.convert_torch_model_to_haiku --config-file=assets/hifigan/config.json --checkpoint-file=assets/infore/hifigan/g_01135000
fi
