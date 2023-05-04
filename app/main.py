import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest


def get_clean_data():
    data_path = "./data/data.csv"
    data = pd.read_csv(data_path)

    data["gender"] = data["gender"].map({"Male": 0, "Female": 1})

    smoker_converted = pd.get_dummies(data["smoking_history"], drop_first=True)
    data = pd.concat([data, smoker_converted], axis=1)
    data.drop("smoking_history", axis=1, inplace=True)

    data["never"] = data.apply(lambda row: row["never"] + row["ever"], axis=1)
    data.drop(["ever"], axis=1, inplace=True)
    
    data = data.rename(columns={'not current': 'not_current'})
    data["former"] = data["not_current"] + data["former"]
    data.drop(['not_current'], axis=1, inplace=True)

    data = data.dropna()

    contam = 11 / 99982
    outliers = False

    # Removing outliers with isolation forest. Contamination set to .01%
    if outliers:
        clf = IsolationForest(contamination=contam, random_state=42)
        clf.fit(data)
        is_outlier = clf.predict(data)
        data = data[is_outlier == 1]

    return data


def add_sidebar():
    st.sidebar.header("Patient Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Gender (0 - Female 1 - Male)", "gender"),
        ("Age", "age"),
        ("Hypertension", "hypertension"),
        ("Heart Disease", "heart_disease"),
        ("BMI", "bmi"),
        ("HbA1c Level", "HbA1c_level"),
        ("Blood Glucose Level", "blood_glucose_level"),
        ("Current Smoker", "current"),
        ("Former Smoker", "former"),
        ("Never a Smoker", "never"),
    ]

    # slider_style = {"background": "#ECECEC", "handle_color": "#FCA503"}

    input_dict = {}
    binary_keys = [
        "gender",
        "hypertension",
        "heart_disease",
        "current",
        "former",
        "never",
    ]

    for label, key in slider_labels:
        if key in binary_keys:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                # value=float(data[key].mean()),
                value=0.0,
                step=1.0,
            )
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                value=float(data[key].mean()),
            )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(["diabetes"], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = [
        "Gender",
        "Age",
        "Hypertension",
        "Heart Disease",
        "BMI",
        "HbA1c Level",
        "Blood Glucose Level",
        "Current Smoker",
        "Former Smoker",
        "Never a Smoker",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["gender"],
                input_data["age"],
                input_data["hypertension"],
                input_data["heart_disease"],
                input_data["bmi"],
                input_data["HbA1c_level"],
                input_data["blood_glucose_level"],
                input_data["current"],
                input_data["former"],
                input_data["never"],
            ],
            theta=categories,
            fill="toself",
            name="Input Data",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Patient prediction")
    st.write("The Patient is:")

    if prediction[0] == 0:
        st.write(
            "<span class='diabetes negative'>Negative</span>", unsafe_allow_html=True
        )
    else:
        st.write(
            "<span class='diabetes positive'>Positive</span>", unsafe_allow_html=True
        )

    st.write(
        "Probability of being negative: ", model.predict_proba(input_array_scaled)[0][0]
    )
    st.write(
        "Probability of being positive: ",
        model.predict_proba(input_array_scaled)[0][1],
    )

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis."
    )


def main():
    st.set_page_config(
        page_title="Patient Diabetes Predictor",
        page_icon="assets/icons8-combo-chart-16.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Patient Diabetes Predictor")
        st.write(
            "The diabetes prediction tool employs a gradient boosting classifier to determine the probability of a patient having diabetes based on various input parameters. It is important to note that while the tool can assist in making a diagnosis, it should not be considered a substitute for a professional medical diagnosis. Patients should always consult with a healthcare professional for a comprehensive diagnosis and personalized treatment plan."
        )

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

    st.caption('Checkout the data [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)!')


if __name__ == "__main__":
    main()
