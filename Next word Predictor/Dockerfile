FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt


EXPOSE 8501

CMD [ "streamlit","run","next_word_predictor.py"]