FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN apt-get update \
    && apt-get -y install libpq-dev gcc
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./app.py