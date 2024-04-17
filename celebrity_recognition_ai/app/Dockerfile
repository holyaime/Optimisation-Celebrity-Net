FROM --platform=linux/amd64 python:3.9-slim as builder
ARG MODEL
ENV MODEL=${MODEL}

COPY requirements.txt dist/*.whl /tmp/
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir /tmp/*.whl && \
    pip install torch torchvision \ 
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install timm && rm -rf /tmp


RUN groupadd -g 1000 mlgroup && useradd -u 1000 -g mlgroup mluser
RUN mkdir -p /usr/src/app && chown mluser:mlgroup /usr/src/app

USER mluser:mlgroup
WORKDIR /usr/src/app

COPY --chown=mluser:mlgroup celebrity_recognition_ai/app/utils.py celebrity_recognition_ai/app/api.py \
    celebrity_recognition_ai/app/trained-models/${MODEL} celebrity_recognition_ai/configs/celebrity-config.yaml ./

COPY --chown=mluser:mlgroup celebrity_recognition_ai/app/static ./static
COPY --chown=mluser:mlgroup celebrity_recognition_ai/app/templates ./templates

EXPOSE 5001
CMD ["gunicorn", "--bind", ":5001", "api:app", "-k", "gthread", "-w", "3", "--threads", "2"]




