FROM mambaorg/micromamba:1.5.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1
ENV MAMBA_DOCKERFILE_ACTIVATE=1

USER root
WORKDIR /app

COPY requirements.txt .

RUN micromamba install -y -n base -c conda-forge python=3.10 pip lightfm && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    micromamba clean --all --yes

COPY movielens_recsys ./movielens_recsys
COPY static ./static
COPY README.md .

EXPOSE 8000

CMD ["python", "-m", "movielens_recsys.serve", "--artifacts-dir", "artifacts", "--host", "0.0.0.0", "--port", "8000"]
