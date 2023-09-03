import streamlit as st
st.title("The Creators 👥")

st.write("____________________________________________________________________________________________________________________________________________")
# About Me Section

# Add developer image
from PIL import Image
image = Image.open('Resources/dev_photo.jpg')
new_image = image.resize((150, 140))
st.image(new_image)
st.write("<b style='font-size: 20px;'>David Oku</b>", unsafe_allow_html=True)
st.write('Student MSc. Data Science')
st.write('Leeds Beckett University,England, United Kingdom.')
st.markdown("""
About : I am a passionate about software engineering with keen interest in data science ,automation , machine learning and game development. I keen in seeing technology being used to gain insights and solve real-world problems.
Grab a cofee and connect with me on [LinkedIn](https://www.linkedin.com/in/masteroflogic/) for a tour around my globe🙃🙃.
""")
st.markdown("""[Personal Portfolio](https://masteroflogic1.github.io/myportfolio/)""")
st.markdown("""[Linked In](https://www.linkedin.com/in/masteroflogic/)""")
st.markdown("""[YouTube](https://www.youtube.com/@masteroflogic)""")




st.write("____________________________________________________________________________________________________________________________________________")
st.write("")
st.write("")
st.write("")
try:
    image = Image.open('Resources/supervisor_photo.jpg')
    new_image = image.resize((150, 140))
    st.image(new_image)
except FileNotFoundError:
    st.write("Image not found")

st.write("<b style='font-size: 20px;'>Shan-A-Khuda, Mohammad</b>", unsafe_allow_html=True)
st.write('Professor & Project Supervisor')
st.write('Leeds Beckett University,England, United Kingdom.')

