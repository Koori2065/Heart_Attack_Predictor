HeartAttackPredictor

Overview

This project predicts the likelihood of a heart attack using the Heart Attack Chance.csv dataset. It employs Logistic Regression to classify patients based on various health-related features. The model is designed for educational purposes, demonstrating data preprocessing, model training, and evaluation using Python and scikit-learn.

Dataset

Source: Heart Attack Chance.csv


Features: All columns except output (e.g., age, cholesterol, blood pressure, etc.).



Target: output (binary: 0 for no heart attack, 1 for heart attack risk).



Model: Logistic Regression with C=1 and solver='liblinear' for binary classification.


Train-test split: 80% training, 20% testing (randomized with np.random.rand).


Evaluation:

Metrics: Confusion Matrix and Classification Report (precision, recall, F1-score).



Clone the repository:

git clone https://github.com/Koori2065/HeartAttackPredictor.git



License

MIT License

Copyright (c) 2025 [Kourosh Asadi]

Permission is hereby granted, free of charge, to any person obtaining a copy
