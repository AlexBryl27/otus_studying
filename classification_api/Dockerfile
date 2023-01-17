FROM python:3.8-slim
WORKDIR /app 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt \
    && rm -r /root/.cache/pip

COPY . .
ENV ROOT_DIR /app/src/
CMD [ "python3", "./src/run.py"]