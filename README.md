# Amazon Stock Prediction Project

In this project, I have set up two models for prediction of Amazon's stock prices based off of headline sentiment and the date, opening price, high, and low values for AMZN on that day.

All the dependencies are within the requirements.txt file, and in order to run the models you will have to navigate to the folder with the models and run the files associated with them. Mainly: model.py for the stacked XGBoost and Linear Regression model, and ff_nn_model.py for feed forward neural network model.

My goal with this project is to compare the architecture and approach between classical machine learning and neural networks, using stock market prediction as the backdrop to this comparison.

In order to start the application `pip3 install -r requirements.txt` and `flask run` in your terminal or command prompt. You can navigate to localhost:5000 to view the site. After that, create an account and enter your queries!

*There are known issues with Tensorflow and Mac OSX, relating to what TF looks for. I am looking into it to fix the dependency issue*
