import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont
import tempfile
from streamlit_stl import stl_from_file
import os

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro + Texto", layout="centered")
st.title("💎 LithoMaker Pro: Foto + Texto")

# --- PARÁMETROS DE INGENIERÍA ---
RES_PX_MM = 5.0  # Resolución (0.2 mm/px)
LADO_MM = 90.0   
PIXELS = int(LADO_MM * RES_PX_MM) 

# Espesores (Z)
MARCO_Z = 5.0      
LITHO_MIN_Z = 0.6  
LITHO_MAX_Z = 3.0  

# --- SIDEBAR ---
st.sidebar.header("1. Producto")
forma = st.sidebar.selectbox("Forma:", ["Corazón", "Círculo", "Cuadrado"])
ancho_marco = st.sidebar.slider("Ancho Marco (mm):", 2.0, 5.0, 3.0)

st.sidebar.header("2. Imagen")
zoom = st.sidebar.slider("Zoom:", 0.5, 3.0, 1.2)
off_x = st.sidebar.slider("Mover X:", -60, 60, 0)
off_y = st.sidebar.slider("Mover Y:", -60, 60, 0)

st.sidebar.divider()
st.sidebar.header("3. Base de Texto")
texto_usuario = st.sidebar.text_input("Escribir nombre o frase:", "AMOR")
font_size = st.sidebar.slider("Tamaño de letra:", 20, 100, 50)
espesor_texto = st.sidebar.slider("Espesor Texto (mm):", 2.0, 20.0, 10.0)

# --- FUNCIONES AUXILIARES ---

def cargar_fuente(size):
    # Intentamos cargar una fuente decente del sistema Linux (Streamlit Cloud)
    try:
        # Ruta común en servidores Debian/Ubuntu
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        try:
            # Ruta alternativa o local
            return ImageFont.truetype("arial.ttf", size)
        except:
            # Fallback básico (se ve feo pero funciona)
            return ImageFont.load_default()

def generar_mascara_texto(texto, size, f_size):
    # Crear lienzo negro
    img = Image.new('L', (size, size // 2), color=0) # Altura mitad del ancho para base
    draw = ImageDraw.Draw(img)
    font = cargar_fuente(int(f_size * 2)) # Multiplicador para calidad
    
    # Calcular tamaño del texto para centrarlo
    bbox = draw.textbbox((0, 0), texto, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Dibujar texto centrado en blanco
    x = (img.width - text_w) // 2
    y = (img.height - text_h) // 2
    draw.text((x, y), texto, font=font, fill=255)
    
    # Convertir a array booleano (True donde hay letra)
    return np.array(img) > 128

def obtener_mascaras(forma_tipo, size, border_mm):
    rango = 1.6 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    
    units_per_mm = (rango * 2) / LADO_MM
    offset = border_mm * units_per_mm

    if forma_tipo == "Círculo":
        R = 1.3
        mask_frame = (x**2 + y**2) <= R**2
        mask_litho = (x**2 + y**2) <= (R - offset)**2
    elif forma_tipo == "Cuadrado":
        L = 1.3
        mask_frame = (np.abs(x) <= L) & (np.abs(y) <= L)
        mask_litho = (np.abs(x) <= (L - offset)) & (np.abs(y) <= (L - offset))
    elif forma_tipo == "Corazón":
        def heart_eq(cx, cy):
            return (cx**2 + (cy - 0.6 * np.sqrt(np.abs(cx)))**2)
        R_heart = 1.6 
        mask_frame = heart_eq(x, y) <= R_heart
        mask_litho = heart_eq(x, y) <= (R_heart - offset*1.8) 

    return mask_litho, mask_frame

def generar_stl_manifold(z_grid, mask_total, lado_real_mm):
    """Genera geometría cerrada basada en máscara"""
    filas, cols = z_grid.shape
    
    # Ajustar dimensiones físicas según el tamaño del grid (para texto vs litho)
    x_lin = np.linspace(0, lado_real_mm, cols)
    y_lin = np.linspace(0, lado_real_mm * (filas/cols), filas) # Proporcional
    X, Y = np.meshgrid(x_lin, y_lin)
    Y = np.flipud(Y) 
    
    faces = []
    valid_pixels = np.argwhere(mask_total)
    
    for i, j in valid_pixels:
        if i >= filas-1 or j >= cols-1: continue
            
        z0, z1, z2, z3 = z_grid[i,j], z_grid[i,j+1], z_grid[i+1,j], z_grid[i+1,j+1]
        x0, y0 = X[i,j], Y[i,j]
        x1, y1 = X[i,j+1], Y[i,j+1]
        x2, y2 = X[i+1,j], Y[i+1,j]
        x3, y3 = X[i+1,j+1], Y[i+1,j+1]
        
        vt0=[x0,y0,z0]; vb0=[x0,y0,0]
        vt1=[x1,y1,z1]; vb1=[x1,y1,0]
        vt2=[x2,y2,z2]; vb2=[x2,y2,0]
        vt3=[x3,y3,z3]; vb3=[x3,y3,0]
        
        # Tapa y Base
        faces.append([vt0, vt2, vt3]); faces.append([vt0, vt3, vt1])
        faces.append([vb0, vb3, vb2]); faces.append([vb0, vb1, vb3])
        
        # Paredes (Detección de bordes)
        if i==0 or not mask_total[i-1, j]: # Norte
            faces.append([vt0, vt1, vb1]); faces.append([vt0, vb1, vb0])
        if (i+1)>=filas-1 or not mask_total[i+1, j]: # Sur
            faces.append([vt2, vb3, vt3]); faces.append([vt2, vb2, vb3])
        if j==0 or not mask_total[i, j-1]: # Oeste
            faces.append([vt0, vb2, vt2]); faces.append([vt0, vb0, vb2])
        if (j+1)>=cols-1 or not mask_total[i, j+1]: # Este
            faces.append([vt1, vt3, vb3]); faces.append([vt1, vb3, vb1])

    return np.array(faces)

# --- INTERFAZ PRINCIPAL ---

tab1, tab2 = st.tabs(["🖼️ 1. Litofanía", "🔤 2. Texto Base"])

with tab1:
    archivo = st.file_uploader("Subir Fotografía", type=['jpg', 'png', 'jpeg'])
    if archivo:
        img = Image.open(archivo).convert('L')
        img_res = img.resize((int(PIXELS*zoom), int((img.height/img.width)*PIXELS*zoom)), Image.Resampling.LANCZOS)
        
        canvas = Image.new('L', (PIXELS, PIXELS), color=255)
        pos_x = (PIXELS - img_res.width) // 2 + int(off_x * RES_PX_MM)
        pos_y = (PIXELS - img_res.height) // 2 + int(off_y * RES_PX_MM)
        canvas.paste(img_res, (pos_x, pos_y))
        
        m_litho, m_frame = obtener_mascaras(forma, PIXELS, ancho_marco)
        img_array = np.array(canvas)
        
        # Preview
        preview = np.array(Image.fromarray(img_array).convert("RGB"))
        preview[m_frame & ~m_litho] = [200, 50, 50] 
        preview[~m_frame] = [30, 30, 30]
        st.image(preview, caption="Vista Previa Litofanía", width=350)
        
        if st.button("🚀 Generar Litofanía (STL)"):
            with st.spinner("Procesando litofanía..."):
                z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
                z_final = np.where(m_litho, z_litho, MARCO_Z)
                
                faces = generar_stl_manifold(z_final, m_frame, LADO_MM)
                
                if len(faces) > 0:
                    regalo_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                    regalo_mesh.vectors = faces
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                        regalo_mesh.save(tmp.name)
                        st.subheader("Litofanía Lista")
                        stl_from_file(file_path=tmp.name, height=250)
                        with open(tmp.name, "rb") as f:
                            st.download_button("📥 Descargar Litofanía", f, f"litho_{forma}.stl")

with tab2:
    st.write(f"**Texto a generar:** {texto_usuario}")
    st.info("💡 Este texto será un objeto sólido separado. En tu Slicer (Cura/Bambu), coloca la litofanía ENCIMA o FUSIONADA con este texto.")
    
    if st.button("🔤 Generar Base de Texto (STL)"):
        with st.spinner("Creando geometría de texto..."):
            # 1. Generar máscara de texto
            mask_text = generar_mascara_texto(texto_usuario, PIXELS, font_size)
            
            # 2. Crear matriz Z constante (todo el texto tiene la misma altura)
            # El espesor lo define el usuario
            z_text = np.full(mask_text.shape, espesor_texto) 
            
            # 3. Generar Manifold (Usamos ancho 90mm como referencia, altura proporcional)
            faces_text = generar_stl_manifold(z_text, mask_text, LADO_MM)
            
            if len(faces_text) > 0:
                text_mesh = mesh.Mesh(np.zeros(faces_text.shape[0], dtype=mesh.Mesh.dtype))
                text_mesh.vectors = faces_text
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_t:
                    text_mesh.save(tmp_t.name)
                    st.success("Texto generado correctamente.")
                    st.subheader("Vista Previa Texto")
                    stl_from_file(file_path=tmp_t.name, material="material", auto_rotate=True, height=200)
                    
                    with open(tmp_t.name, "rb") as f_t:
                        st.download_button("📥 Descargar Texto Base", f_t, f"base_{texto_usuario}.stl")
            else:
                st.warning("El texto no generó geometría. Intenta aumentar el tamaño de letra.")
