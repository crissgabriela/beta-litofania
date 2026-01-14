import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageOps
import tempfile

# Configuración de página
st.set_page_config(page_title="LithoMaker Beta", page_icon="💎", layout="centered")
st.title("💎 LithoMaker Beta")
st.write("Sube una foto y obtén tu modelo 3D listo para imprimir.")

# --- BARRA LATERAL ---
st.sidebar.header("Ajustes")
ancho = st.sidebar.slider("Ancho (mm):", 50, 150, 100)
min_grosor = st.sidebar.slider("Grosor mínimo (mm):", 0.6, 1.5, 0.8)
max_grosor = st.sidebar.slider("Grosor máximo (mm):", 2.0, 4.0, 3.0)
invertir = st.sidebar.checkbox("Invertir (Modo Litofanía)", value=True)

# --- FUNCIÓN MATEMÁTICA (Numpy) ---
def generar_mesh(image, width_mm, min_th, max_th, inverted):
    img = image.convert('L')
    if inverted: img = ImageOps.invert(img)
    
    # Redimensionar (10 px por mm para buena resolución web)
    ratio = img.height / img.width
    height_mm = width_mm * ratio
    pixels_w, pixels_h = int(width_mm * 10), int(height_mm * 10)
    img = img.resize((pixels_w, pixels_h))
    
    # Crear nube de puntos Z
    data = np.array(img)
    z = min_th + (data / 255.0) * (max_th - min_th)
    
    # Malla XY
    x, y = np.meshgrid(np.linspace(0, width_mm, pixels_w), 
                       np.linspace(0, height_mm, pixels_h))
    y = np.flipud(y) # Corregir orientación
    
    # Definir vértices para STL
    # (Simplificación para velocidad en la nube)
    vertices = np.zeros((pixels_h, pixels_w, 3))
    vertices[:,:,0], vertices[:,:,1], vertices[:,:,2] = x, y, z
    
    # Vectorización de caras (Triángulos)
    v00 = vertices[:-1, :-1]; v01 = vertices[:-1, 1:]
    v10 = vertices[1:, :-1]; v11 = vertices[1:, 1:]
    
    faces1 = np.concatenate([v00[...,None,:], v10[...,None,:], v11[...,None,:]], axis=-2)
    faces2 = np.concatenate([v00[...,None,:], v11[...,None,:], v01[...,None,:]], axis=-2)
    
    all_faces = np.concatenate([faces1, faces2], axis=0).reshape(-1, 3, 3)
    
    malla = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    malla.vectors = all_faces
    return malla

# (Opcional) Agregar después de generar el mesh
st.write("Vista previa de la geometría:")
# Streamlit no tiene visor STL nativo simple, 
# pero podemos mostrar la imagen de altura como referencia técnica
st.image(image, caption="Mapa de profundidad generado", clamp=True)

# --- INTERFAZ ---
archivo = st.file_uploader("Cargar imagen", type=['jpg', 'png', 'jpeg'])

if archivo:
    image = Image.open(archivo)
    st.image(image, caption="Vista previa", use_container_width=True)
    
    if st.button("🚀 Crear Modelo STL"):
        with st.spinner("Procesando en la nube..."):
            malla = generar_mesh(image, ancho, min_grosor, max_grosor, invertir)
            
            # Guardar en memoria temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                malla.save(tmp.name)
                
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="📥 DESCARGAR STL",
                        data=f,
                        file_name="litofania.stl",
                        mime="application/sla",
                        use_container_width=True
                    )

            st.success("¡Listo!")
