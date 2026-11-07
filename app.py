import pandas as pd
import streamlit as st


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
]
TARGET_COLUMN = "Revenue"
MIN_FEATURE_COUNT = 12


st.set_page_config(page_title="Online Shopper Classifier")
st.title("Online Shopper Purchase Classifier")
st.write(
    "Explore the Online Shoppers Purchasing Intention dataset and compare "
    "classification models on labeled test data."
)

model_name = st.selectbox("Choose a classifier", MODEL_NAMES)
uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")

if uploaded_file is None:
    st.info("Upload a test CSV that includes the input features and `Revenue` target.")
else:
    try:
        test_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
    else:
        if TARGET_COLUMN not in test_data.columns:
            st.error(f"The CSV must include the `{TARGET_COLUMN}` target column.")
        elif len(test_data.columns) - 1 < MIN_FEATURE_COUNT:
            st.error(f"The CSV must include at least {MIN_FEATURE_COUNT} input features.")
        elif test_data[TARGET_COLUMN].isna().any():
            st.error(f"The `{TARGET_COLUMN}` target column cannot contain empty values.")
        else:
            st.subheader("Uploaded test data")
            st.caption(f"Selected model: {model_name}")
            st.write(
                f"{len(test_data):,} rows · "
                f"{len(test_data.columns) - 1} input features"
            )
            st.dataframe(test_data.head(), use_container_width=True)
            st.info(
                "Prediction and evaluation will be enabled after the saved model "
                "artifacts are added."
            )
