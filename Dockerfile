FROM python:3.11-slim

ARG NUMERAI_MODEL_ID

RUN if [ -z ${NUMERAI_MODEL_ID} ]; then echo "NUMERAI_MODEL_ID is not set (docker build --build-arg NUMERAI_MODEL_ID=)"; exit 1; fi

ENV NUMERAI_MODEL_ID=$NUMERAI_MODEL_ID

WORKDIR /model

COPY ./model /model
COPY .env .env

RUN pip install -r requirements.txt

CMD ["python", "model.py"]