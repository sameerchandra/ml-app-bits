# Online Shopper Purchase Classification

This project compares five machine-learning classifiers for predicting whether
an online shopping session ends in a purchase. Models are trained offline and
the Streamlit app evaluates uploaded, labeled test data without retraining.

## a. Problem statement

Given the attributes of an online shopping session, predict whether the session
results in a purchase (`Revenue`). Because purchases are the minority class,
the comparison reports accuracy, AUC, precision, recall, F1, and Matthews
correlation coefficient (MCC), rather than relying on accuracy alone.

## b. Dataset description

The [Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
is from the UCI Machine Learning Repository. It contains 12,330 sessions with
17 input features and the binary `Revenue` target. The target has 10,422
non-purchase sessions and 1,908 purchase sessions; the UCI record reports no
missing values.

The dataset includes session activity and duration, web analytics measures,
month, visitor type, and other session attributes. The CSV used by the training
script is `online_shoppers_intention.csv`. The held-out `test_data.csv` contains
2,466 rows, including 382 purchases and 2,084 non-purchases.

Citation: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention
Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.
The UCI dataset is licensed under CC BY 4.0.

## c. GitHub Repository Link

[sameerchandra/ml-app-bits](https://github.com/sameerchandra/ml-app-bits)

## d. Models used

The dataset is split once into 80% training and 20% test data using stratified
sampling (`random_state=42`). Preprocessing is fitted as part of each model
pipeline on the training data only. The table reports results on the same
held-out test rows. AUC uses the predicted probability for the positive class,
`Revenue=True`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.881 | 0.888 | 0.743 | 0.356 | 0.481 | 0.460 |
| Decision Tree | 0.856 | 0.725 | 0.534 | 0.537 | 0.535 | 0.450 |
| K-Nearest Neighbors | 0.873 | 0.782 | 0.646 | 0.401 | 0.494 | 0.442 |
| Gaussian Naive Bayes | 0.674 | 0.793 | 0.294 | 0.788 | 0.428 | 0.323 |
| Support Vector Machine | 0.887 | 0.874 | 0.707 | 0.461 | 0.558 | 0.511 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.888) and precision (0.743), but recall is 0.356. Its default decision threshold misses many purchase sessions. |
| Decision Tree | Recall (0.537) and precision (0.534) are more balanced than the logistic model, but its AUC (0.725) is lowest, indicating weaker ranking performance on this split. |
| K-Nearest Neighbors | Reaches 0.873 accuracy and 0.646 precision, with 0.401 recall. Its MCC (0.442) is lower than those of Logistic Regression and the Decision Tree. |
| Gaussian Naive Bayes | Highest recall (0.788), but precision is only 0.294 and accuracy is 0.674. It finds more purchases at the cost of more false purchase predictions. |
| Support Vector Machine | Highest accuracy (0.887), F1 (0.558), and MCC (0.511). It has a more balanced precision/recall trade-off than the other models with strong overall results on this split. |
| Overall, Winner for your dataset? | **Support Vector Machine**, when selecting by the strongest combined F1 and MCC on this held-out split. If maximizing purchase recall is the priority, Gaussian Naive Bayes is the alternative. The winner depends on the costs of missed purchases versus false alarms. |

## Run locally

Install the pinned dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a labeled CSV containing the same 17 input columns and `Revenue` target.
The app loads the saved pipelines from `model/`, evaluates all five models on
the uploaded rows, and displays the metric comparison and selected model's
confusion matrix. Uploaded rows are not used to train or update a model.

To recreate the saved models and held-out CSV from the source dataset, place
`online_shoppers_intention.csv` in the repository root and run:

```bash
python train_models.py
```
