import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont
import tempfile
from streamlit_stl import stl_from_file
import math

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro + Texto", layout="centered")
st.title("💎 LithoMaker Pro: Suite de Impresión")

# --- PARÁMETROS DE INGENIERÍA ---
RES_PX_MM = 5.0  # 5 píxeles por mm (Alta definición)

# Espesores (Z)
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
st.sidebar.header("3. Generador de Textos")
texto_usuario = st.sidebar.text_input("Escribir nombre:", "TE AMO")
tamano_fuente = st.sidebar.slider("Tamaño de letra (pt):", 30, 150, 80)
altura_texto = st.sidebar.slider("Relieve del Texto (mm):", 2.0, 15.0, 5.0)

# --- FUNCIONES AUXILIARES ---

def cargar_fuente(size):
    try:
        # Intentamos cargar fuente del sistema
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        # Fallback
        return ImageFont.load_default()

def obtener_mascaras(forma_tipo, size, border_mm):
    LADO_ESTANDAR = 90.0
    rango = 1.6 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    
    units_per_mm = (rango * 2) / LADO_ESTANDAR
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

# --- GENERADOR MANIFOLD DINÁMICO ---
def generar_stl_manifold(z_grid, mask_total):
    filas, cols = z_grid.shape
    
    # Calcular dimensiones físicas reales en mm
    ancho_real_mm = cols / RES_PX_MM
    alto_real_mm = filas / RES_PX_MM
    
    x_lin = np.linspace(0, ancho_real_mm, cols)
    y_lin = np.linspace(0, alto_real_mm, filas)
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
        
        # Paredes
        if i==0 or not mask_total[i-1, j]: # Norte
            faces.append([vt0, vt1, vb1]); faces.append([vt0, vb1, vb0])
        if (i+1)>=filas-1 or not mask_total[i+1, j]: # Sur
            faces.append([vt2, vb3, vt3]); faces.append([vt2, vb2, vb3])
        if j==0 or not mask_total[i, j-1]: # Oeste
            faces.append([vt0, vb2, vt2]); faces.append([vt0, vb0, vb2])
        if (j+1)>=cols-1 or not mask_total[i, j+1]: # Este
            faces.append([vt1, vt3, vb3]); faces.append([vt1, vb3, vb1])

    return np.array(faces)

# --- LÓGICA DE TEXTO ---
def procesar_texto_con_base(texto, font_size_pt, relieve_mm):
    font = cargar_fuente(font_size_pt)
    
    # 1. Dimensiones
    dummy_img = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), texto, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 2. Base
    margen_px = int(1.0 * RES_PX_MM) # 1mm margen
    base_w = text_w
    base_h = text_h + (margen_px * 2)
    
    img_text = Image.new('L', (base_w, base_h), 0)
    draw_text = ImageDraw.Draw(img_text)
    
    # Dibujar texto
    draw_text.text((-bbox[0], margen_px - bbox[1]), texto, font=font, fill=255)
    
    mask_letters = np.array(img_text) > 128
    mask_base = np.ones((base_h, base_w), dtype=bool)
    
    # 3. Z Map
    z_map = np.full((base_h, base_w), 5.0) # Base 5mm
    z_map[mask_letters] = 5.0 + relieve_mm # Letras suman altura
    
    return z_map, mask_base

# --- INTERFAZ ---

tab1, tab2 = st.tabs(["🖼️ Litofanía", "🔤 Base de Texto"])

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
        
        preview = np.array(Image.fromarray(img_array).convert("RGB"))
        preview[m_frame & ~m_litho] = [200, 50, 50] 
        preview[~m_frame] = [30, 30, 30]
        st.image(preview, caption="Vista Previa", width=350)
        
        if st.button("🚀 Generar Litofanía"):
            with st.spinner("Creando litofanía..."):
                z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
                z_final = np.where(m_litho, z_litho, MARCO_Z)
                
                faces = generar_stl_manifold(z_final, m_frame)
                
                regalo_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                regalo_mesh.vectors = faces
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                    regalo_mesh.save(tmp.name)
                    st.subheader("Litofanía Lista")
                    stl_from_file(file_path=tmp.name, height=250)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Descargar STL", f, f"litho_{forma}.stl")

with tab2:
    st.markdown("#### Crea una base sólida con nombre")
    
    font_preview = cargar_fuente(tamano_fuente)
    img_prev = Image.new('RGB', (400, 100), (240, 240, 240))
    d = ImageDraw.Draw(img_prev)
    d.text((10, 20), texto_usuario, font=font_preview, fill=(0,0,0))
    st.image(img_prev, caption="Estilo de letra", width=300)
    
    st.info(f"La base tendrá 5mm de altura. El texto sobresaldrá {altura_texto}mm más.")

    if st.button("🔤 Generar Placa de Nombre"):
        with st.spinner("Fusionando texto y base..."):
            
            z_text_map, mask_total = procesar_texto_con_base(texto_usuario, tamano_fuente, altura_texto)
            faces_text = generar_stl_manifold(z_text_map, mask_total)
            
            if len(faces_text) > 0:
                text_mesh = mesh.Mesh(np.zeros(faces_text.shape[0], dtype=mesh.Mesh.dtype))
                text_mesh.vectors = faces_text
                
                # ROTACIÓN 90 GRADOS (Para que quede de pie)
                text_mesh.rotate([1.0, 0.0, 0.0], math.radians(90))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_t:
                    text_mesh.save(tmp_t.name)
                    
                    # Cálculo seguro de dimensiones para evitar el error de sintaxis
                    ancho_final = mask_total.shape[1] / RES_PX_MM
                    alto_final = mask_total.shape[0] / RES_PX_MM
                    st.success(f"¡Placa generada! Dimensiones: {ancho_final:.1f} x {alto_final:.1f} mm")
                    
                    st.subheader("Vista Previa 3D")
                    stl_from_file(file_path=tmp_t.name, material="material", auto_rotate=True, height=250)
                    
                    with open(tmp_t.name, "rb") as f_t:
                        st.download_button(
                            label=f"📥 Descargar Placa '{texto_usuario}'", 
                            data=f_t, 
                            file_name=f"base_texto_{texto_usuario}.stl"
                        )
            else:
                st.error("Error al generar la geometría del texto.")
