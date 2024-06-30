import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Titanic-Dataset.csv')


df = load_data()

# Preprocess the data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Select features and target
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']


# Train a Random Forest model
@st.cache_resource
def train_model():
    model = RandomForestClassifier()
    return model.fit(X, y)


model = train_model()

# Streamlit app
st.header('Titanic Survival Prediction 🚤')
st.write("Fill in the details to predict survival:")

pclass = st.selectbox('Pclass', (1, 2, 3))
sex = st.selectbox('Sex', ('male', 'female'))
age = st.slider('Age', 0, 80, 25)

# Convert user input to model input
sex = 0 if sex == 'male' else 1
input_data = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age]})

# Button for making prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_text = 'Survived' if prediction[0] == 1 else 'Not Survived'

    # Display the result
    st.write(f'Prediction: {prediction_text}')