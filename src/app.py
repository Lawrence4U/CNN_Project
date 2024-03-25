import streamlit as st
import pandas as pd


# Configuración de la página
st.set_page_config(
    page_title="CNN"
    )


# Definir función para la primera página
# def page1():
#     image_path = "./resources/vta.png"  # Reemplaza esto con la ruta real de tu imagen
#     # st.image(image_path, caption='VTA', width=100)
#     # st.title("Valora tu activo")

#     col1, col2 = st.columns([1, 2])
#     col1.image(image_path, width=100)
#     col2.title("Valora tu activo")
    
#     ref_catastral = st.text_input("Introduce la referencia catastral")
#     if ref_catastral:
#         # Obtener la valoración.
#         catastro = "9731115VK3893B0106HP"
#         info = start_test(catastro)

#         datos_csv, datos_texto = obtener_datos(ref_catastral)


#         # Mostrar texto sin ser editable
#         st.header("Datos extraidos para la valoración:")

#         col1, col2, col3, col4 = st.columns([1, 1, 1, 1])  # Dividir el espacio en dos columnas
#         # Descargar CSV
#         csv = datos_csv.to_csv(index=False)
#         col1.download_button(
#             label="Descargar CSV",
#             data=csv,
#             file_name='datos.csv',
#             mime='text/csv'
#         )
#         # Descargar PDF (solo un ejemplo vacío)
#         pdf = b''
#         col2.download_button(
#             label="Descargar PDF",
#             data=pdf,
#             file_name='datos.pdf',
#             mime='application/pdf'
#         )

#         informacion_activo(info)

# Definir función para la segunda página
def page2():
    st.title("Adjunta una foto")
    st.write("Sube un archivo:")
    uploaded_file = st.file_uploader("Seleccione un archivo")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Imagen:")
        st.write(data)

# Definir función para la tercera página
# def page3():
#     st.title("Calculadoras")
#     st.write("Realiza cálculos matemáticos básicos:")
#     num1 = st.number_input("Ingrese el primer número")
#     num2 = st.number_input("Ingrese el segundo número")
#     operation = st.selectbox("Seleccionar operación", ["Sumar", "Restar", "Multiplicar", "Dividir"])
#     if st.button("Calcular"):
#         if operation == "Sumar":
#             result = num1 + num2
#         elif operation == "Restar":
#             result = num1 - num2
#         elif operation == "Multiplicar":
#             result = num1 * num2
#         elif operation == "Dividir":
#             if num2 != 0:
#                 result = num1 / num2
#             else:
#                 result = "Error: División por cero"
#         st.write(f"El resultado es: {result}")

page2()