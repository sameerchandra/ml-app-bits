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
sampling (`random_state=2026`). Preprocessing is fitted as part of each model
pipeline on the training data only. The table reports results on the same
held-out test rows. AUC uses the predicted probability for the positive class,
`Revenue=True`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.878 | 0.880 | 0.716 | 0.356 | 0.476 | 0.448 |
| Decision Tree | 0.874 | 0.763 | 0.591 | 0.602 | 0.597 | 0.522 |
| K-Nearest Neighbors | 0.871 | 0.789 | 0.647 | 0.374 | 0.474 | 0.427 |
| Gaussian Naive Bayes | 0.684 | 0.797 | 0.298 | 0.770 | 0.430 | 0.324 |
| Support Vector Machine | 0.886 | 0.872 | 0.710 | 0.448 | 0.549 | 0.505 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.880) and strong precision (0.716), but recall is 0.356. Its default decision threshold misses many purchase sessions. |
| Decision Tree | Highest recall (0.602), F1 (0.597), and MCC (0.522), with 0.591 precision. These results provide the most balanced positive-class performance on this split. |
| K-Nearest Neighbors | Reaches 0.871 accuracy and 0.647 precision, with 0.374 recall. Its F1 (0.474) and MCC (0.427) are lower than the Decision Tree and SVM. |
| Gaussian Naive Bayes | Highest recall after the Decision Tree (0.770), but precision is 0.298 and accuracy is 0.684. It finds more purchases at the cost of more false purchase predictions. |
| Support Vector Machine | Highest accuracy (0.886) and precision (0.710), with 0.448 recall and 0.549 F1. It performs strongly overall but has lower F1 and MCC than the Decision Tree on this split. |
| Overall, Winner for your dataset? | **Decision Tree**, when selecting by F1 and MCC on this held-out split. If ranking quality is the priority, Logistic Regression has the highest AUC; if accuracy and precision matter most, SVM is strongest. The winner depends on the costs of missed purchases versus false alarms. |

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
