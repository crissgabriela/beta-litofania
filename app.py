import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont
import tempfile
from streamlit_stl import stl_from_file
import math

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro 2026", layout="centered")
st.title("💎 LithoMaker Pro: Normalización de Caracteres")

# --- PARÁMETROS DE INGENIERÍA ---
RES_PX_MM = 5.0  # 0.2 mm/pixel

# Espesores Litofanía
MARCO_Z = 5.0      
LITHO_MIN_Z = 0.6  
LITHO_MAX_Z = 3.0  

# --- SIDEBAR ---
st.sidebar.header("1. Producto Litofanía")
forma = st.sidebar.selectbox("Forma:", ["Corazón", "Círculo", "Cuadrado"])
ancho_marco = st.sidebar.slider("Ancho Marco (mm):", 2.0, 5.0, 3.0)

st.sidebar.header("2. Imagen")
zoom = st.sidebar.slider("Zoom:", 0.5, 3.0, 1.2)
off_x = st.sidebar.slider("Mover X:", -60, 60, 0)
off_y = st.sidebar.slider("Mover Y:", -60, 60, 0)

st.sidebar.divider()
st.sidebar.header("3. Configuración Placa")
texto_usuario = st.sidebar.text_input("Escribir nombre:", "ALEJANDRA").upper()

st.sidebar.info("📏 Dimensiones Normalizadas:")
st.sidebar.markdown("""
* **Largo Fijo:** 180 mm
* **Altura Letra:** 30 mm (Independiente)
* **Profundidad:** 30 mm
* **Base Suelo:** 5 mm
""")

# --- FUNCIONES ---

def cargar_fuente(size_px):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size_px))
    except:
        return ImageFont.load_default()

def obtener_mascaras(forma_tipo, size, border_mm):
    LADO_ESTANDAR = 90.0
    rango = 2.0 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    units_per_mm = (rango * 2) / LADO_ESTANDAR
    offset = border_mm * units_per_mm
    if forma_tipo == "Círculo":
        R = 1.3
        m_frame = (x**2 + y**2) <= R**2
        m_litho = (x**2 + y**2) <= (R - offset)**2
    elif forma_tipo == "Cuadrado":
        L = 1.3
        m_frame = (np.abs(x) <= L) & (np.abs(y) <= L)
        m_litho = (np.abs(x) <= (L - offset)) & (np.abs(y) <= (L - offset))
    elif forma_tipo == "Corazón":
        def heart(cx, cy): return (cx**2 + (cy - 0.6 * np.sqrt(np.abs(cx)))**2)
        m_frame = heart(x, y) <= 1.8 
        m_litho = heart(x, y) <= (1.8 - offset*1.8) 
    return m_litho, m_frame

def generar_stl_manifold(z_grid, mask_total):
    filas, cols = z_grid.shape
    ancho_mm, alto_mm = cols / RES_PX_MM, filas / RES_PX_MM
    x_lin, y_lin = np.linspace(0, ancho_mm, cols), np.linspace(0, alto_mm, filas)
    X, Y = np.meshgrid(x_lin, y_lin)
    Y = np.flipud(Y) 
    faces = []
    v_px = np.argwhere(mask_total)
    for i, j in v_px:
        if i >= filas-1 or j >= cols-1: continue
        z0, z1, z2, z3 = z_grid[i,j], z_grid[i,j+1], z_grid[i+1,j], z_grid[i+1,j+1]
        vt0=[X[i,j],Y[i,j],z0]; vb0=[X[i,j],Y[i,j],0]
        vt1=[X[i,j+1],Y[i,j+1],z1]; vb1=[X[i,j+1],Y[i,j+1],0]
        vt2=[X[i+1,j],Y[i+1,j],z2]; vb2=[X[i+1,j],Y[i+1,j],0]
        vt3=[X[i+1,j+1],Y[i+1,j+1],z3]; vb3=[X[i+1,j+1],Y[i+1,j+1],0]
        faces.append([vt0, vt2, vt3]); faces.append([vt0, vt3, vt1])
        faces.append([vb0, vb3, vb2]); faces.append([vb0, vb1, vb3])
        if i==0 or not mask_total[i-1, j]: faces.append([vt0, vt1, vb1]); faces.append([vt0, vb1, vb0])
        if (i+1)>=filas-1 or not mask_total[i+1, j]: faces.append([vt2, vb3, vt3]); faces.append([vt2, vb2, vb3])
        if j==0 or not mask_total[i, j-1]: faces.append([vt0, vb2, vt2]); faces.append([vt0, vb0, vb2])
        if (j+1)>=cols-1 or not mask_total[i, j+1]: faces.append([vt1, vt3, vb3]); faces.append([vt1, vb3, vb1])
    return np.array(faces)

def procesar_texto_nameplate_normalizado(texto):
    ANCHO_BASE_MM = 180.0
    ALTO_TEXTO_MM = 30.0  
    ALTO_BASE_MM = 5.0    
    EXTRUSION_MM = 30.0   
    
    px_w = int(ANCHO_BASE_MM * RES_PX_MM)
    px_h_text = int(ALTO_TEXTO_MM * RES_PX_MM)
    px_h_base = int(ALTO_BASE_MM * RES_PX_MM)
    
    # Crear canvas final
    total_h = px_h_text + px_h_base
    final_canvas = Image.new('L', (px_w, total_h), 0)
    
    # Dividir ancho entre número de caracteres
    n_chars = len(texto)
    char_w_target = px_w // n_chars
    
    font = cargar_fuente(250)
    
    for i, letra in enumerate(texto):
        # Renderizar cada letra por separado en alta resolución
        char_img = Image.new('L', (500, 500), 0)
        draw = ImageDraw.Draw(char_img)
        draw.text((50, 50), letra, font=font, fill=255)
        
        # Obtener el cuadro de contorno exacto de la letra
        bbox = char_img.getbbox()
        if bbox:
            letra_recortada = char_img.crop(bbox)
            # Escalar letra para que ocupe char_w_target x px_h_text
            # Esto deforma la letra para que todas midan lo mismo
            letra_final = letra_recortada.resize((char_w_target, px_h_text), Image.Resampling.LANCZOS)
            final_canvas.paste(letra_final, (i * char_w_target, 0))
    
    mask_text = np.array(final_canvas) > 128
    mask_base = np.zeros((total_h, px_w), dtype=bool)
    mask_base[px_h_text:, :] = True 
    
    mask_total = mask_text | mask_base
    z_map = np.zeros((total_h, px_w))
    z_map[mask_text] = EXTRUSION_MM
    z_map[mask_base] = EXTRUSION_MM + 1.0 
    
    return z_map, mask_total

# --- INTERFAZ ---
tab1, tab2 = st.tabs(["🖼️ Litofanía", "🔤 Placa de Nombre"])

with tab1:
    archivo = st.file_uploader("Subir Fotografía", type=['jpg', 'png', 'jpeg'])
    if archivo:
        PIXELS_STD = int(90.0 * RES_PX_MM)
        img = Image.open(archivo).convert('L')
        img_res = img.resize((int(PIXELS_STD*zoom), int((img.height/img.width)*PIXELS_STD*zoom)), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (PIXELS_STD, PIXELS_STD), color=255)
        pos_x = (PIXELS_STD - img_res.width) // 2 + int(off_x * RES_PX_MM)
        pos_y = (PIXELS_STD - img_res.height) // 2 + int(off_y * RES_PX_MM)
        canvas.paste(img_res, (pos_x, pos_y))
        m_litho, m_frame = obtener_mascaras(forma, PIXELS_STD, ancho_marco)
        img_array = np.array(canvas)
        prev = np.array(Image.fromarray(img_array).convert("RGB"))
        prev[m_frame & ~m_litho] = [200, 50, 50] 
        prev[~m_frame] = [30, 30, 30]
        st.image(prev, caption="Vista de Encuadre", width='stretch')
        if st.button("🚀 Generar Litofanía"):
            z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
            z_final = np.where(m_litho, z_litho, MARCO_Z)
            faces = generar_stl_manifold(z_final, m_frame)
            reg_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            reg_mesh.vectors = faces
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                reg_mesh.save(tmp.name)
                stl_from_file(file_path=tmp.name, height=300)
                with open(tmp.name, "rb") as f:
                    st.download_button("📥 Descargar STL", f, f"litho_{forma}.stl", width='stretch')

with tab2:
    st.markdown("#### Nameplate 180x30mm (Normalización Forzada)")
    if st.button("🔤 Generar Placa de Nombre"):
        with st.spinner("Procesando caracteres individuales..."):
            z_map, mask_tot = procesar_texto_nameplate_normalizado(texto_usuario)
            faces_t = generar_stl_manifold(z_map, mask_tot)
            if len(faces_t) > 0:
                text_mesh = mesh.Mesh(np.zeros(faces_t.shape[0], dtype=mesh.Mesh.dtype))
                text_mesh.vectors = faces_t
                v_y, v_z = text_mesh.vectors[:,:,1].copy(), text_mesh.vectors[:,:,2].copy()
                text_mesh.vectors[:,:,1], text_mesh.vectors[:,:,2] = v_z, v_y
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_t:
                    text_mesh.save(tmp_t.name)
                    st.success("✅ Placa normalizada generada con éxito.")
                    stl_from_file(file_path=tmp_t.name, material="material", auto_rotate=True, height=250)
                    with open(tmp_t.name, "rb") as f_t:
                        st.download_button("📥 Descargar Placa", f_t, f"base_texto_{texto_usuario}.stl", width='stretch')
