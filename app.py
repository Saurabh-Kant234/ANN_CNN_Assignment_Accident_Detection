import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title='Accident Detection System', page_icon='🚨', layout='centered'
)


@st.cache_resource
def load_model():
  return tf.keras.models.load_model('models/best_accident_model.keras')


model = load_model()

st.title('🚨 CCTV Traffic Accident Detection System')
st.write(
    'Upload a traffic frame image to check whether an accident has occurred.'
)

uploaded_file = st.file_uploader(
    'Choose an image...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded CCTV Frame', use_column_width=True)

  # Preprocess image to match model input dimensions
  image = image.resize((224, 224))
  img_array = np.array(image)

  if img_array.ndim == 2:
    img_array = np.stack((img_array,) * 3, axis=-1)
  elif img_array.shape[2] == 4:
    img_array = img_array[:, :, :3]

  img_array = np.expand_dims(img_array, axis=0)

  if st.button('Predict Accident Status'):
    with st.spinner('Analyzing traffic footage...'):
      score = float(model.predict(img_array)[0][0])

    st.subheader('Prediction Result:')
    # Alphabetical order: 'Accident' (0) vs 'Non accident' (1)
    if score < 0.5:
      st.error(
          f'⚠️ **Accident Detected!** (Confidence:'
          f' {((1 - score) * 100):.2f}%)'
      )
    else:
      st.success(
          f'✅ **Normal Traffic / No Accident** (Confidence:'
          f' {(score * 100):.2f}%)'
      )