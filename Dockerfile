FROM python:3.7

RUN adduser ml_app

WORKDIR /home/ml_app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install tensorflow==2.1.0
RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY app app
COPY migrations migrations
COPY saved_models saved_models
COPY data_modules data_modules
COPY data_csv data_csv
COPY loaded_models.py loaded_models.py
COPY ml_app.py config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP ml_app.py

RUN chown -R ml_app:ml_app ./
USER ml_app

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
