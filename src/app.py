import streamlit as st
from PIL import Image
from main import *


# Configuración de la página
st.set_page_config(
    page_title="CNN"
    )


# Definir función para la segunda página
def main():
    st.title("Adjunta una foto")
    st.write("Sube un archivo:")
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        label = get_label(image)
        st.write(f"La imagen analizada es: {label}")
        st.image(image, caption='Imagen cargada', use_column_width=True)

if __name__ == "__main__":
    main()