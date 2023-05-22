import streamlit as st
from PIL import Image



st.title("Machine Learning")

image = Image.open('images/Machine Learning.jpg')

st.image(image, caption='Classification of Common Algorithms for Machine Learning')

msg = '''
Machine learning is the science of getting computers to act without being explicitly programmed to do so. Over the past decade, machine learning has brought us self-driving cars, practical speech recognition, efficient web search, and vastly improved understanding of the human genome. Machine learning is so pervasive today, you probably use it dozens of times a day without knowing it. Many researchers also believe that this is the best way to make progress towards human-level artificial intelligence.
'''
st.markdown(msg)
