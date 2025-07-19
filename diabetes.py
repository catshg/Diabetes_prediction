import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("diabetes 1.csv")

# Feature and target separation
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Model training
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))


# App start
def main():
    st.title("Diabetes Prediction ")

    # Load and display image
    image = Image.open("istock.jpg")

    st.image(image, caption='Diabetes Awareness', width=500)

    # Sidebar input
    st.sidebar.title("Enter Patient Details")
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, step=1)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99,23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1,32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078,2.42,0.001)
    age = st.sidebar.slider('Age', 10, 100, 30)

    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("The person is likely to have diabetes.")
        else:
            st.success("The person is not likely to have diabetes.")


# Display Accuracy
    st.markdown("---")
    st.info(f"📊 **Model Accuracy**")
    st.write(f"✅ Train Accuracy: **{train_accuracy:.2f}**")
    st.write(f"✅ Test Accuracy: **{test_accuracy:.2f}**")


if __name__ == '__main__':
    main()
