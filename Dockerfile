FROM python:3.8.16-slim-buster
LABEL authors="mekk-knights"

ARG INCUBATOR_VER=unknown
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git  &&\
    apt-get install -y wget &&\
    apt-get install -y libsndfile1 &&\
    apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev

RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
RUN mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
RUN pip3 install pyctcdecode
RUN pip3 install https://github.com/kpu/kenlm/archive/master.zip
RUN git clone https://github.com/chau25102001/SOICT_hackathon_2023_SLU_Mekk_Knights.git
WORKDIR SOICT_hackathon_2023_SLU_Mekk_Knights
RUN pip3 install -r /SOICT_hackathon_2023_SLU_Mekk_Knights/requirements.txt
RUN git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
RUN cd transformers && pip3 install -e . && cd ..
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install gdown
RUN pip3 install accelerate -U

RUN gdown --fuzzy https://drive.google.com/file/d/18ny_Nru2iem0eAyebkdjZHuY3LrJjMfB/view?usp=sharing -O /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/data/train_final_20230919.jsonl
RUN cp /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/data/train_final_20230919.jsonl /SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction/data/train_final_20230919.jsonl
RUN mkdir -p /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/checkpoint
RUN mkdir -p /SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction/checkpoint

RUN gdown --fuzzy https://drive.google.com/file/d/1_V-xCcfaUoNkEKVvGVZ9Ktw3SEA6Z0AA/view?usp=sharing -O /SOICT_hackathon_2023_SLU_Mekk_Knights/text_correction/checkpoint/checkpoint_best.pt
RUN gdown --fuzzy https://drive.google.com/file/d/1h9Z-gcIqWu5nzc-tH_OZiDbZgdkdzXpp/view?usp=sharing -O /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/checkpoint/checkpoint_best.pt

ENV LANG C.UTF-8
WORKDIR /SOICT_hackathon_2023_SLU_Mekk_Knights