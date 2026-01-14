import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageOps
import tempfile
from streamlit_stl import stl_from_file

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro Commercial", layout="centered")
st.title("💎 LithoMaker Pro: Edición Die-Cut")

# --- PARÁMETROS TÉCNICOS FIJOS ---
RES_PX_MM = 4.0  # 0.25mm/px
LADO_MM = 90.0
PIXELS = int(LADO_MM * RES_PX_MM) 
MARCO_Z = 5.0      
LITHO_MIN_Z = 0.6  
LITHO_MAX_Z = 3.0  

# --- SIDEBAR: AJUSTES COMERCIALES ---
st.sidebar.header("1. Geometría del Producto")
forma = st.sidebar.selectbox("Forma final:", ["Corazón", "Círculo", "Cuadrado"])
ancho_marco = st.sidebar.slider("Ancho del Marco (mm):", 2.0, 5.0, 3.0)

st.sidebar.header("2. Encuadre de Imagen")
zoom = st.sidebar.slider("Zoom:", 0.5, 3.0, 1.2)
off_x = st.sidebar.slider("Mover X:", -60, 60, 0)
off_y = st.sidebar.slider("Mover Y:", -60, 60, 0)

# --- LÓGICA DE MÁSCARAS DE PRECISIÓN ---
def generar_mascaras_precisas(forma, size, border_mm):
    rango = 1.8 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    
    # Conversión de mm a unidades normalizadas (regla de 3 basada en rango)
    # Rango total es 3.6 para 90mm -> 1mm = 0.04 unidades
    offset = border_mm * 0.04

    if forma == "Círculo":
        radio_ext = 1.5
        mask_frame = x**2 + y**2 <= radio_ext**2
        mask_litho = x**2 + y**2 <= (radio_ext - offset)**2
    elif forma == "Cuadrado":
        lado_ext = 1.5
        mask_frame = (np.abs(x) <= lado_ext) & (np.abs(y) <= lado_ext)
        mask_litho = (np.abs(x) <= (lado_ext - offset)) & (np.abs(y) <= (lado_ext - offset))
    elif forma == "Corazón":
        def h(cx, cy, scale):
            return (cx/scale)**2 + ( (cy/scale) - 0.8 * np.sqrt(np.abs(cx/scale)) )**2
        mask_frame = h(x, y, 1.2) <= 1.0
        mask_litho = h(x, y, 1.2 - offset) <= 1.0
    
    return mask_litho, mask_frame

# --- PROCESAMIENTO ---
archivo = st.file_uploader("Subir Fotografía del Cliente", type=['jpg', 'png', 'jpeg'])

if archivo:
    img = Image.open(archivo).convert('L')
    img_res = img.resize((int(PIXELS*zoom), int((img.height/img.width)*PIXELS*zoom)), Image.Resampling.LANCZOS)
    
    canvas = Image.new('L', (PIXELS, PIXELS), color=255)
    pos_x = (PIXELS - img_res.width) // 2 + int(off_x * RES_PX_MM)
    pos_y = (PIXELS - img_res.height) // 2 + int(off_y * RES_PX_MM)
    canvas.paste(img_res, (pos_x, pos_y))
    
    m_litho, m_frame = generar_mascaras_precisas(forma, PIXELS, ancho_marco)
    img_array = np.array(canvas)
    
    # Previsualización con marco teñido
    preview = np.array(Image.fromarray(img_array).convert("RGB"))
    preview[~m_litho & m_frame] = [200, 50, 50] # Rojo para el marco
    preview[~m_frame] = [20, 20, 20]           # Negro para lo que se recorta
    st.image(preview, caption="Vista de producción (Rojo = Marco sólido)", width=350)

    if st.button(f"🚀 Generar STL Troquelado ({forma})"):
        with st.spinner("Calculando geometría y recortando bordes..."):
            z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
            z_final = np.where(m_litho, z_litho, MARCO_Z)
            
            x_lin = np.linspace(0, LADO_MM, PIXELS)
            y_lin = np.linspace(0, LADO_MM, PIXELS)
            X, Y = np.meshgrid(x_lin, y_lin)
            Y = np.flipud(Y)

            # --- TRIANGULACIÓN FILTRADA (DIE-CUT) ---
            faces = []
            for i in range(PIXELS - 1):
                for j in range(PIXELS - 1):
                    # CONDICIÓN COMERCIAL: Solo generamos si el cuadrado está dentro de la forma
                    if m_frame[i,j] or m_frame[i+1,j] or m_frame[i,j+1]:
                        # Vértices Top y Bottom
                        vt = np.array([[X[i,j], Y[i,j], z_final[i,j]], [X[i+1,j], Y[i+1,j], z_final[i+1,j]], 
                                       [X[i+1,j+1], Y[i+1,j+1], z_final[i+1,j+1]], [X[i,j+1], Y[i,j+1], z_final[i,j+1]]])
                        vb = np.array([[X[i,j], Y[i,j], 0], [X[i+1,j], Y[i+1,j], 0], 
                                       [X[i+1,j+1], Y[i+1,j+1], 0], [X[i,j+1], Y[i,j+1], 0]])
                        
                        faces.append([vt[0], vt[1], vt[2]]) # Top 1
                        faces.append([vt[0], vt[2], vt[3]]) # Top 2
                        faces.append([vb[0], vb[2], vb[1]]) # Btm 1
                        faces.append([vb[0], vb[3], vb[2]]) # Btm 2
            
            all_faces = np.array(faces)
            regalo_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
            regalo_mesh.vectors = all_faces
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                regalo_mesh.save(tmp.name)
                st.subheader("👀 Revisión 3D")
                stl_from_file(file_path=tmp.name, auto_rotate=True, height=300)
                
                with open(tmp.name, "rb") as f_stl:
                    st.download_button(
                        label=f"📥 DESCARGAR {forma.upper()} FINAL",
                        data=f_stl,
                        file_name=f"litho_comercial_{forma.lower()}.stl",
                        mime="application/sla",
                        width='stretch'
                    )
