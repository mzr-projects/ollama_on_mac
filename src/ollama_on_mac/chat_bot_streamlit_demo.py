import streamlit as st

st.set_page_config(page_title="Streamlit component demo", page_icon="Tools",layout="wide")

st.title("Hi! This is a streamlit chat bot")
input = st.text_input("Enter your name")

# if the input is not empty, write a welcome message
if input:
    st.write(f"Hello {input} wellcome to my chat")

agree = st.number_input("Your age:",min_value=0,max_value=100,value=25)
select_box = st.selectbox("Choose a model",["gpt-4o","gpt-4o-mini","gpt-4.1"])
temperature = st.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
agree = st.checkbox("I agree to the terms and conditions")
choice = st.radio("Choose a color",["red","green","blue"])

button = st.button("Submit")
if button:
    st.write(f"Generating response...")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Choose a model",["gpt-4o","gpt-4o-mini","gpt-4.1"])
    temperature = st.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)

col1, col2 = st.columns(2)
with col1:
    st.write("This is the first column")
    st.metric("Temperature",temperature)
with col2:
    st.write("This is the second column")
    st.metric("Wind Speed", "9 mph", delta="3 mph")