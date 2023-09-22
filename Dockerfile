FROM python:3.11-slim

ARG NUMERAI_MODEL_ID
RUN if [ -z ${NUMERAI_MODEL_ID} ]; then echo "NUMERAI_MODEL_ID is not set (docker build --build-arg NUMERAI_MODEL_ID=sample_model .)"; exit 1; fi
ENV NUMERAI_MODEL_ID=$NUMERAI_MODEL_ID

ARG GIT_COMMIT
ENV GIT_COMMIT_HASH=$GIT_COMMIT

WORKDIR /model

COPY .env .env
COPY ./models/${NUMERAI_MODEL_ID} /model

RUN apt update \
 && apt install -y --no-install-recommends libgomp1 \
 && apt -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python", "model.py"]