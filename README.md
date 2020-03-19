# Amazon Stock Prediction Project

In this project, I have set up two models for prediction of Amazon's stock prices based off of headline sentiment and the date, opening price, high, and low values for AMZN on that day.

All the dependencies are within the requirements.txt file, and in order to run the models you will have to navigate to the folder with the models and run the files associated with them. Mainly: model.py for the stacked XGBoost and Linear Regression model, and ff_nn_model.py for feed forward neural network model.

My goal with this project is to compare the architecture and approach between classical machine learning and neural networks, using stock market prediction as the backdrop to this comparison.

In order to start the application `pip3 install -r requirements.txt` and `flask run` in your terminal or command prompt. You can navigate to localhost:5000 to view the site. After that, create an account and enter your queries!

If you would rather start the application as a docker image, start by typing this into your terminal/command prompt `docker build -t ml_app:latest .`, followed by `docker run --name ml_app -d -p 8000:5000 --rm ml_app:latest`, to have it run in the background and removing it once you are finished. Navigate to localhost:8000 to view the site. After that, create an account and enter your queries!
