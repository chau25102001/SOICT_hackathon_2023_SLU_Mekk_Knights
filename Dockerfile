FROM python:3.8.16-slim-buster
LABEL authors="mekk-knights"

ARG INCUBATOR_VER=unknown
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/chau25102001/SOICT_hackathon_2023_SLU_Mekk_Knights.git
WORKDIR SOICT_hackathon_2023_SLU_Mekk_Knights
RUN pip3 install --no-cache-dir -r /SOICT_hackathon_2023_SLU_Mekk_Knights/requirements.txt
RUN git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
RUN cd transformers && pip3 install -e . && cd ..
RUN pip3 install gdown

RUN gdown --fuzzy https://drive.google.com/file/d/18ny_Nru2iem0eAyebkdjZHuY3LrJjMfB/view?usp=sharing -O /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/data
RUN mkdir "SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/checkpoint"
RUN gdown --fuzzy https://drive.google.com/file/d/1h9Z-gcIqWu5nzc-tH_OZiDbZgdkdzXpp/view?usp=sharing -O /SOICT_hackathon_2023_SLU_Mekk_Knights/IDSF/checkpoint

ENV LANG C.UTF-8
WORKDIR /SOICT_hackathon_2023_SLU_Mekk_Knights