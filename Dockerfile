FROM python:3.8.16-slim-buster
LABEL authors="mekk-knights"

ARG INCUBATOR_VER=unknown

RUN git clone https://github.com/chau25102001/SOICT_hackathon_2023_SLU_Mekk_Knights.git
RUN cd SOICT_hackathon_2023_SLU_Mekk_Knights

COPY requirements.txt /SLU/requirements.txt

RUN pip3 install --no-cache-dir -r /SLU/requirements.txt
RUN git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
RUN cd transformers
RUN pip3 install -e .
RUN cd ..
RUN pip3 install gdown

COPY . /SLU

RUN gdown --fuzzy https://drive.google.com/file/d/18ny_Nru2iem0eAyebkdjZHuY3LrJjMfB/view?usp=sharing -O /SLU/IDSF/data
RUN mkdir "/SLU/IDSF/checkpoint"
RUN gdown --fuzzy https://drive.google.com/file/d/1h9Z-gcIqWu5nzc-tH_OZiDbZgdkdzXpp/view?usp=sharing -O /SLU/IDSF/checkpoint
RUN cd /SLU/IDSF/data
ENV LANG C.UTF-8
WORKDIR /SLU/requirements.txt