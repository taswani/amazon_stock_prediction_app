FROM python:3.7-alpine

RUN adduser -D ml_app

WORKDIR /home/ml_app

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY app app
COPY migrations migrations
COPY ml_app.py config.py boot.sh ./
COPY classical_model.joblib data_pipeline.py data.py ff_model.h5 ff_nn_model.py final_amazon.csv loaded_models.py lstm_model.h5 lstm_model.py model.py text_classification.py
RUN chmod +x boot.sh

ENV FLASK_APP ml_app.py

RUN chown -R ml_app:ml_app ./
USER ml_app

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
