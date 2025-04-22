This project aims to predict sleep quality based on various daily habits like physical activity,
screen time, and stress levels. Using machine learning algorithms, we analyze how these factors 
 influence the quality of sleep and make predictions based on user input.
To run this project, you'll need to install the following dependencies:
pip install -r requirements.txt

Create a requirements.txt file that includes:

nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
streamlit  # (if you're deploying)

The dataset used in this project is from Kaggle - Sleep Health and Lifestyle Dataset.

The dataset contains information on:

Physical Activity Levels

Caffeine Intake

Screen Time

Stress Levels

Sleep Duration

Sleep Quality (Target variable)

Usage
After setting up the environment, you can run the project by executing the Jupyter notebook or Python script.

Run the Jupyter Notebook: Open the notebook and execute each cell to follow the analysis and prediction steps.

Run the Python script: To run the entire script from start to finish, use:

bash
Copy
Edit
python sleep_prediction.py

Data Preprocessing
This step includes handling missing values, encoding categorical variables, and splitting the dataset into features (X) and target (y).

Key preprocessing steps:

Handling missing data

One-hot encoding categorical variables

Splitting data into training and testing sets using a stratified approach

After training the model, we evaluate its performance using various metrics like accuracy, precision, recall, and f1-score.
