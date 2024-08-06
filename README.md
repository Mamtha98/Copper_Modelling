# Copper_Modelling
**Objective**
To enhance pricing and lead management in the copper industry by leveraging machine learning models. The goal is to:

Predict the Selling_Price of copper products using a regression model.
Classify leads as WON or LOST using a classification model.
Create a user-friendly Streamlit application to facilitate easy predictions based on user inputs.

**Simple Approach**
Data Exploration and Cleaning:

Inspect Data: Load the dataset and examine basic statistics and distributions.
Handle Skewness and Outliers: Identify and address skewness using transformations (e.g., log transformation) and detect outliers using statistical methods.
Preprocess Data: Handle missing values, normalize features, and encode categorical variables.
Build ML Models:

Regression Model: Train a regression model (e.g., Decision Tree Regressor) to predict Selling_Price using the cleaned data.
Classification Model: Filter data to include only WON and LOST statuses, then train a classification model (e.g., Decision Tree Classifier) to predict the lead status.
Create Streamlit Application:

Develop Interface: Set up a Streamlit app where users can input feature values.
Integrate Models: Load the trained models into the app and use them to provide predictions based on user input.

**Conclusion**
By following this approach, we aim to:

Improve Prediction Accuracy: Machine learning models will provide precise predictions for copper selling prices and lead statuses, surpassing manual methods.
Streamline Lead Management: The classification model will help prioritize leads, enhancing sales efficiency and strategy.
Enhance Usability: The Streamlit application will make it easy for users to get predictions, improving accessibility and decision-making.
