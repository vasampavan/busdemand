# Predicting Hourly Bus Boarding Demand Using Deep Learning


This project predicts bus passenger boarding demand using smart-card data. The dataset is highly imbalanced, which negatively impacts prediction accuracy. We employ Deep Generative Adversarial Networks (Deep-GAN) to create synthetic data for balancing the dataset. A Deep Neural Network (DNN) is then used to predict boarding instances, improving accuracy compared to traditional methods.


Features:

1. Handles imbalanced datasets using Deep-GAN.

2. Improves predictions for boarding vs. non-boarding instances.

3. Applicable for urban transport planning.

System Requirements:

OS: Windows 7 or later

Coding Language: Python

Front-End: Python

Back-End: Django-ORM

Designing: HTML, CSS, JavaScript

Database: MySQL (XAMPP Server)

Processor: Core i5

RAM: 8 GB (minimum)

Hard Disk: 512 GB

Algorithms Used:

1. Decision Trees

A non-parametric supervised learning algorithm, ideal for classification tasks. It splits the dataset into branches, helping to decide boarding vs. non-boarding instances based on feature values.

2. Random Forest

An ensemble of decision trees where each tree votes on the prediction. It improves accuracy and handles data variance effectively by reducing overfitting.

3. Logistic Regression

A linear model used for binary classification, predicting the probability of bus boarding by mapping inputs to outputs using a logistic function.

4. K-Nearest Neighbors (KNN)

A simple instance-based algorithm that classifies new data points based on proximity to existing instances. The distance metric helps identify whether a passenger will board.

5. Support Vector Machine (SVM)

A supervised machine learning model that finds the optimal hyperplane to separate boarding and non-boarding passengers in a high-dimensional space.

6. Gradient Boosting

An ensemble method that builds weak learners sequentially, minimizing errors at each step. It enhances the accuracy of predictions by combining multiple models.

7. Naive Bayes

A probabilistic algorithm based on Bayes’ Theorem. It calculates the likelihood of a passenger boarding by assuming that features are independent of each other.

Installation:

Install Python and Django.

Set up XAMPP server and configure MySQL.

Clone the repository and install dependencies.

Run the Django server and access the web interface.

Preview:

![image](https://github.com/user-attachments/assets/59b96ee3-51bf-4da6-adc4-70cb211f66a4)

![image](https://github.com/user-attachments/assets/acaee9c4-7c21-4ca2-a751-9a096d10580e)



![image](https://github.com/user-attachments/assets/22f03139-088b-41c0-b7da-aadd7263dbb4)


Conclusion:

This project demonstrates how Deep-GAN can address imbalanced data issues, leading to improved prediction accuracy for public transport demand forecasting. The findings 
offer valuable insights for transportation planning.
