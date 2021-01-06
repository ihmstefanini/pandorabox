ARG PYTHON_VERSION=3.8.1-slim

FROM python:${PYTHON_VERSION} as build

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --upgrade pip

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /pandora_app

COPY requirements.txt /etc

RUN pip install -r /etc/requirements.txt 

COPY . /pandora_app

FROM python:${PYTHON_VERSION} as prod

COPY --from=build /opt/venv /opt/venv

RUN apt-get update && apt-get install --no-install-recommends -y \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /pandora_app

RUN mkdir -p /data/log

EXPOSE 8501

COPY . /pandora_app

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["streamlit","run"]

CMD ["pandora_app.py"]
