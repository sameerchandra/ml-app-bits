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
            loaded_models = {}
            expected_features = None

            for name, model_path in MODEL_PATHS.items():
                if not model_path.is_file():
                    st.error(f"Saved model is missing: `{model_path}`.")
                    st.stop()

                model = load_model(str(model_path))
                model_features = list(model.feature_names_in_)
                if expected_features is None:
                    expected_features = model_features
                elif model_features != expected_features:
                    st.error("Saved models do not use the same input feature schema.")
                    st.stop()

                loaded_models[name] = model

            uploaded_features = set(test_data.columns) - {TARGET_COLUMN}
            required_features = set(expected_features)
            missing_features = sorted(required_features - uploaded_features)
            unexpected_features = sorted(uploaded_features - required_features)
            if missing_features or unexpected_features:
                problems = []
                if missing_features:
                    problems.append(f"Missing features: {', '.join(missing_features)}.")
                if unexpected_features:
                    problems.append(f"Unexpected columns: {', '.join(unexpected_features)}.")
                st.error(" ".join(problems))
                st.stop()

            test_features = test_data[expected_features]
            test_labels = test_data[TARGET_COLUMN]
            for name, model in loaded_models.items():
                classes = list(model.classes_)
                if len(classes) != 2 or True not in classes:
                    st.error(f"Saved model `{name}` does not have the expected binary classes.")
                    st.stop()

                purchase_column = classes.index(True)
                metrics_by_model[name] = calculate_metrics(
                    test_labels,
                    model.predict(test_features),
                    model.predict_proba(test_features)[:, purchase_column],
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

            selected_model = loaded_models[model_name]
            predictions = selected_model.predict(test_features)
            st.subheader(f"{model_name} details")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(test_labels, predictions, labels=[False, True]),
                    index=["Actual: no purchase", "Actual: purchase"],
                    columns=["Predicted: no purchase", "Predicted: purchase"],
                ),
                use_container_width=True,
            )
            st.caption("Confusion matrix counts are ordered as no-purchase, purchase.")
