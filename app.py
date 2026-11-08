from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from modeling import calculate_metrics


MODEL_PATHS = {
    "Logistic Regression": Path("model/logistic_regression.joblib"),
    "Decision Tree": Path("model/decision_tree.joblib"),
    "K-Nearest Neighbors": Path("model/k_nearest_neighbors.joblib"),
    "Gaussian Naive Bayes": Path("model/gaussian_naive_bayes.joblib"),
    "Support Vector Machine": Path("model/support_vector_machine.joblib"),
}
TARGET_COLUMN = "Revenue"

st.set_page_config(page_title="Online Shopper Classifier")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


st.title("Online Shopper Purchase Classifier")
st.write(
    "Compare five classifiers on labeled Online Shoppers test data. "
    "Models are trained offline; uploaded rows are used only for evaluation."
)

model_name = st.selectbox("Inspect a classifier", list(MODEL_PATHS))
uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV with the 17 input features and `Revenue` target.")
else:
    try:
        test_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
    else:
        if test_data.empty:
            st.error("The uploaded CSV has no data rows.")
        elif TARGET_COLUMN not in test_data.columns:
            st.error(f"The CSV must include the `{TARGET_COLUMN}` target column.")
        elif test_data[TARGET_COLUMN].isna().any():
            st.error(f"The `{TARGET_COLUMN}` target column cannot contain empty values.")
        elif test_data[TARGET_COLUMN].nunique() != 2:
            st.error(f"The `{TARGET_COLUMN}` column must contain both classes.")
        else:
            metrics_by_model = {}
            model_by_name = {}
            expected_features = None

            for name, path in MODEL_PATHS.items():
                if not path.is_file():
                    st.error(f"Saved model is missing: `{path}`.")
                    st.stop()

                model = load_model(str(path))
                model_features = list(model.feature_names_in_)
                if expected_features is None:
                    expected_features = model_features
                elif model_features != expected_features:
                    st.error("Saved models do not use the same input feature schema.")
                    st.stop()

                model_by_name[name] = model

            actual_features = set(test_data.columns) - {TARGET_COLUMN}
            expected_feature_set = set(expected_features)
            missing_features = sorted(expected_feature_set - actual_features)
            unexpected_features = sorted(actual_features - expected_feature_set)
            if missing_features or unexpected_features:
                details = []
                if missing_features:
                    details.append(f"Missing features: {', '.join(missing_features)}.")
                if unexpected_features:
                    details.append(f"Unexpected columns: {', '.join(unexpected_features)}.")
                st.error(" ".join(details))
                st.stop()

            x_test = test_data[expected_features]
            y_test = test_data[TARGET_COLUMN]
            for name, model in model_by_name.items():
                classes = list(model.classes_)
                if len(classes) != 2 or True not in classes:
                    st.error(f"Saved model `{name}` does not have the expected binary classes.")
                    st.stop()

                positive_index = classes.index(True)
                metrics_by_model[name] = calculate_metrics(
                    y_test,
                    model.predict(x_test),
                    model.predict_proba(x_test)[:, positive_index],
                    positive_label=True,
                )

            st.subheader("Model comparison")
            st.caption(f"Evaluation on {len(test_data):,} uploaded labeled rows")
            st.dataframe(
                pd.DataFrame.from_dict(metrics_by_model, orient="index")
                .rename_axis("Model")
                .reset_index()
                .set_index("Model")
                .style.format("{:.3f}"),
                use_container_width=True,
            )

            selected_model = model_by_name[model_name]
            predictions = selected_model.predict(x_test)
            st.subheader(f"{model_name} details")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(y_test, predictions, labels=[False, True]),
                    index=["Actual: no purchase", "Actual: purchase"],
                    columns=["Predicted: no purchase", "Predicted: purchase"],
                ),
                use_container_width=True,
            )
            st.caption("Confusion matrix counts are ordered as no-purchase, purchase.")
