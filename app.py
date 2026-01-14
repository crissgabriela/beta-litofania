import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageOps
import tempfile

# Configuración de página profesional
st.set_page_config(page_title="LithoMaker Pro Beta", page_icon="💎", layout="centered")
st.title("💎 LithoMaker Pro")
st.write("Generador de litofanías sólidas con resolución técnica de 0.2mm.")

# --- BARRA LATERAL (Simplificada para Beta Comercial) ---
st.sidebar.header("Ajustes de Impresión")
ancho = st.sidebar.slider("Ancho total (mm):", 50, 150, 100)
min_grosor = st.sidebar.slider("Grosor mínimo (mm):", 0.6, 1.2, 0.6) 
max_grosor = st.sidebar.slider("Grosor máximo (mm):", 2.0, 5.0, 3.0) 
invertir = st.sidebar.checkbox("Invertir para Contraluz", value=True)

# RESOLUCIÓN FIJA (0.2 mm/pixel -> 5 px/mm)
RES_PX_MM = 5.0 

def generar_mesh_solido(image, width_mm, min_th, max_th, inverted):
    img = image.convert('L')
    if inverted: img = ImageOps.invert(img)
    
    ratio = img.height / img.width
    height_mm = width_mm * ratio
    
    # Cálculo de píxeles basado en la resolución fija de 0.2mm/px
    pixels_w = int(width_mm * RES_PX_MM)
    pixels_h = int(height_mm * RES_PX_MM)
    img = img.resize((pixels_w, pixels_h), Image.Resampling.LANCZOS)
    
    data = np.array(img)
    z_top = min_th + (data / 255.0) * (max_th - min_th)
    
    x_lin = np.linspace(0, width_mm, pixels_w)
    y_lin = np.linspace(0, height_mm, pixels_h)
    X, Y = np.meshgrid(x_lin, y_lin)
    Y = np.flipud(Y) 
    
    vertices_top = np.zeros((pixels_h, pixels_w, 3))
    vertices_top[:,:,0], vertices_top[:,:,1], vertices_top[:,:,2] = X, Y, z_top
    
    vertices_bottom = np.zeros((pixels_h, pixels_w, 3))
    vertices_bottom[:,:,0], vertices_bottom[:,:,1], vertices_bottom[:,:,2] = X, Y, 0

    faces = []
    # CARA SUPERIOR
    v00, v01 = vertices_top[:-1, :-1], vertices_top[:-1, 1:]
    v10, v11 = vertices_top[1:, :-1], vertices_top[1:, 1:]
    faces.append(np.concatenate([v00[...,None,:], v10[...,None,:], v11[...,None,:]], axis=-2).reshape(-1, 3, 3))
    faces.append(np.concatenate([v00[...,None,:], v11[...,None,:], v01[...,None,:]], axis=-2).reshape(-1, 3, 3))

    # CARA INFERIOR
    b00, b01 = vertices_bottom[:-1, :-1], vertices_bottom[:-1, 1:]
    b10, b11 = vertices_bottom[1:, :-1], vertices_bottom[1:, 1:]
    faces.append(np.concatenate([b00[...,None,:], b11[...,None,:], b10[...,None,:]], axis=-2).reshape(-1, 3, 3))
    faces.append(np.concatenate([b00[...,None,:], b01[...,None,:], b11[...,None,:]], axis=-2).reshape(-1, 3, 3))

    # PAREDES
    def crear_pared(b_t, b_b):
        t0, t1, b0, b1 = b_t[:-1], b_t[1:], b_b[:-1], b_b[1:]
        return np.concatenate([np.stack([t0, b0, t1], axis=1), np.stack([b0, b1, t1], axis=1)], axis=0)

    faces.append(crear_pared(vertices_top[0,:], vertices_bottom[0,:]))
    faces.append(crear_pared(vertices_bottom[-1,:], vertices_top[-1,:]))
    faces.append(crear_pared(vertices_bottom[:,0], vertices_top[:,0]))
    faces.append(crear_pared(vertices_top[:,-1], vertices_bottom[:,-1]))

    all_faces = np.concatenate(faces, axis=0)
    malla = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    malla.vectors = all_faces
    return malla, all_faces.shape[0]

# --- INTERFAZ ---
archivo = st.file_uploader("Subir Fotografía", type=['jpg', 'png', 'jpeg'])

if archivo:
    image = Image.open(archivo)
    st.image(image, caption="Imagen original cargada", width=300)
    
    if st.button("🚀 Generar STL Comercial"):
        with st.spinner("Procesando a 0.2mm/px..."):
            malla, n_tri = generar_mesh_solido(image, ancho, min_grosor, max_grosor, invertir)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                malla.save(tmp.name)
                st.success(f"Modelo sólido generado con {n_tri:,} triángulos.")
                
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="📥 DESCARGAR STL",
                        data=f,
                        file_name="litofania_pro.stl",
                        mime="application/sla",
                        use_container_width=True
                    )
