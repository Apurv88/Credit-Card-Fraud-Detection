Title: Credit Card Fraud Detection

1. Introduction:
Credit card fraud is a significant concern in the digital economy, leading to billions in losses annually. Detecting fraudulent transactions quickly and accurately is crucial to ensure financial security and build trust in online payment systems. With the power of machine learning, it is now possible to identify fraudulent behavior patterns in real time.

2. Abstract:
This project aims to identify fraudulent transactions using anomaly detection techniques. It uses a publicly available dataset and apply machine learning model LightGBM classifier is trained with specific hyperparameters like learning rate, number of leaves, and tree depth. The goal is to detect unusual transaction patterns that may indicate fraud. The model’s performance is evaluated using ROC curves and confusion matrices, and a user-friendly interface is created to interact with the predictions.

LGBM:-Light Gradient Boosting Machine is a high performance gradient boosting framework that is optimized for Speed and efficiency unlike traditional decision trees like GBM uses Leaf wise growth which means it builds the most significant splits first making it much faster and memory efficient.

3. Tools Used:
- Python
- Scikit-Learn
- LightGBM
- Pandas
- NumPy
- Seaborn
- Mathplotlib
- Jupyter Notebook
- Streamlit  for UI

4. Steps Involved in Building the Project:
- Workflow: Includes data import, preprocessing, analysis, model training, evaluation, and deployment with a user interface.
- Libraries Used: Key libraries include pandas, NumPy, seaborn, matplotlib, scikit-learn, and LightGBM.
- Data Preprocessing: Extracts time features (hour, day, month) and removes irrelevant columns.
- Categorical Encoding: Label encoding is applied to categorical features like gender and merchant category.
- Distance Feature: A new feature is created by calculating the distance between transaction and merchant locations.
- Feature Selection: Selected features include merchant category, amount, credit card number, time-based features, and gender.
- Imbalanced Dataset: The original dataset is highly imbalanced with very few fraudulent transactions.
- Balancing the Data: SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic fraud cases and balance the dataset.
- Data Splitting: The dataset is split into training and testing sets.
- Model Training: The LGBM classifier is trained with specific hyperparameters like learning rate, number of leaves, and tree depth.
- Evaluation Metrics: Includes ROC-AUC score and classification report for model performance.
- Feature Importance: The top 10 important features for classification are visualized.
- ROC Curve: ROC-AUC curve is plotted to show model performance; a score of 0.9+ indicates good separation.
- Model Saving: The trained model and encoders are saved using joblib.
- User Interface: A front-end is built using Streamlit for real-time user interaction.
- Prediction Output: The model outputs whether the transaction is legitimate or fraudulent.


5. Conclusion:
The project successfully demonstrates how machine learning can be employed to detect credit card fraud effectively. Anomaly detection methods combined with classification model like LightGBM can accurately identify suspicious transactions. A deployable dashboard allows real-time prediction, making it feasible for integration into real-world financial systems.

Deliverables:
- Jupyter Notebook with model training and evaluation
- Streamlit UI for user input and prediction
- Confusion matrix to evaluate model performance

Dataset link:-https://drive.google.com/file/d/1118Jwzj51KpXd0T5jiebn9ykCygwbkhn/view?usp=sharing
