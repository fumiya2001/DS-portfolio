FROM python:3.12

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]