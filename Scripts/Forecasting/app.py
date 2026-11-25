import streamlit as st
import pickle
import pandas as pd

# Load model + transformer
with open("predict_delay.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
transformer = package["transformer"]
features = package["feature_names"]

st.title("Train Delay Prediction")

# حقول الإدخال
departure = st.text_input("Departure Station")
arrival = st.text_input("Arrival Destination")
month = st.number_input("Month", min_value=1, max_value=12, step=1)
journey_day = st.text_input("Journey Day")
departure_hour = st.number_input("Departure Hour", min_value=0, max_value=23, step=1)
day_number = st.number_input("Day Number", min_value=1, max_value=31, step=1)

# زر التنبؤ
if st.button("Predict Delay"):
    input_df = pd.DataFrame([[
        departure, arrival, month, journey_day, departure_hour, day_number
    ]], columns=features)

    X = transformer.transform(input_df)
    prob = model.predict_proba(X)[0][1]
    prediction = 1 if prob > 0.35 else 0
############################################# 
    # # عرض النتائج بعد الضغط على الزر
    # st.write(f"Probability of Delay: {prob:.3f}")
    # st.write("Prediction:", "Delayed" if prediction == 1 else "No Delay")

    st.subheader("Prediction Result")

    # Probability as progress bar
    st.write("Probability of Delay:")
    st.progress(prob)  # يظهر شريط تقدم من 0 إلى 1

    # Prediction with color
    if prediction == 1:
        st.error("Prediction: Delayed")  # أحمر
    else:
        st.success("Prediction: No Delay")  # أخضر

    # عرض مدخلات المستخدم في جدول
    st.subheader("Input Data")
    st.table(input_df)
