import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont
import tempfile
from streamlit_stl import stl_from_file

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro Final", layout="centered")
st.title("💎 LithoMaker Pro: Suite de Impresión")

# --- PARÁMETROS DE INGENIERÍA ---
RES_PX_MM = 4.0  # 4 px/mm (Bajamos levemente para manejar bloques grandes de 50mm prof)

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
texto_usuario = st.sidebar.text_input("Escribir nombre:", "AMOR")

# Parámetros FIJOS solicitados (informativos en la UI o sliders restringidos)
st.sidebar.info("📏 Dimensiones Fijas:")
st.sidebar.markdown("""
* **Largo Base:** 180 mm
* **Altura Letra:** 50 mm (Fijo)
* **Extrusión:** 50 mm (Fijo)
""")

# --- FUNCIONES AUXILIARES ---

def cargar_fuente(size_px):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size_px))
    except:
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

# --- GENERADOR MANIFOLD ---
def generar_stl_manifold(z_grid, mask_total):
    filas, cols = z_grid.shape
    ancho_real_mm = cols / RES_PX_MM
    alto_real_mm = filas / RES_PX_MM
    
    x_lin = np.linspace(0, ancho_real_mm, cols)
    y_lin = np.linspace(0, alto_real_mm, filas)
    X, Y = np.meshgrid(x_lin, y_lin)
    Y = np.flipud(Y) 
    
    faces = []
    # Vectorización parcial para velocidad en bloques grandes
    valid_pixels = np.argwhere(mask_total)
    
    # Pre-cálculo de vértices para no repetir lógica
    for i, j in valid_pixels:
        if i >= filas-1 or j >= cols-1: continue
            
        z0, z1, z2, z3 = z_grid[i,j], z_grid[i,j+1], z_grid[i+1,j], z_grid[i+1,j+1]
        
        # Optimización: Si z es 0, no generamos nada (aire)
        # Pero aquí z_grid siempre tiene altura (5mm base o 50mm texto)
        
        vt0=[X[i,j],Y[i,j],z0]; vb0=[X[i,j],Y[i,j],0]
        vt1=[X[i,j+1],Y[i,j+1],z1]; vb1=[X[i,j+1],Y[i,j+1],0]
        vt2=[X[i+1,j],Y[i+1,j],z2]; vb2=[X[i+1,j],Y[i+1,j],0]
        vt3=[X[i+1,j+1],Y[i+1,j+1],z3]; vb3=[X[i+1,j+1],Y[i+1,j+1],0]
        
        faces.append([vt0, vt2, vt3]); faces.append([vt0, vt3, vt1])
        faces.append([vb0, vb3, vb2]); faces.append([vb0, vb1, vb3])
        
        if i==0 or not mask_total[i-1, j]: 
            faces.append([vt0, vt1, vb1]); faces.append([vt0, vb1, vb0])
        if (i+1)>=filas-1 or not mask_total[i+1, j]: 
            faces.append([vt2, vb3, vt3]); faces.append([vt2, vb2, vb3])
        if j==0 or not mask_total[i, j-1]: 
            faces.append([vt0, vb2, vt2]); faces.append([vt0, vb0, vb2])
        if (j+1)>=cols-1 or not mask_total[i, j+1]: 
            faces.append([vt1, vt3, vb3]); faces.append([vt1, vb3, vb1])

    return np.array(faces)

# --- LÓGICA DE TEXTO DEFORMABLE ---
def procesar_texto_stretch(texto):
    # DIMENSIONES FIJAS
    ANCHO_OBJETIVO_MM = 180.0
    ALTO_TEXTO_MM = 50.0  # Altura de la letra
    ALTO_BASE_MM = 5.0    # Altura del suelo
    EXTRUSION_MM = 50.0   # Profundidad (Z)
    
    # Conversión a píxeles
    target_w_px = int(ANCHO_OBJETIVO_MM * RES_PX_MM)
    text_h_px = int(ALTO_TEXTO_MM * RES_PX_MM)
    base_h_px = int(ALTO_BASE_MM * RES_PX_MM)
    
    # 1. Generar Texto en Alta Resolución (cuadrado inicial para no perder calidad)
    # Usamos una fuente grande para renderizar y luego redimensionamos
    font_size_init = text_h_px 
    font = cargar_fuente(font_size_init)
    
    # Medir texto sin deformar
    dummy = ImageDraw.Draw(Image.new('L', (1,1)))
    bbox = dummy.textbbox((0,0), texto, font=font)
    w_raw = bbox[2] - bbox[0]
    h_raw = bbox[3] - bbox[1]
    
    # Renderizar texto original
    img_temp = Image.new('L', (w_raw, h_raw), 0)
    draw = ImageDraw.Draw(img_temp)
    draw.text((-bbox[0], -bbox[1]), texto, font=font, fill=255)
    
    # 2. DEFORMACIÓN (Stretch)
    # Redimensionamos la imagen del texto a 180mm x 50mm
    # Usamos LANCZOS para suavizado
    img_stretched = img_temp.resize((target_w_px, text_h_px), Image.Resampling.LANCZOS)
    
    # 3. Composición Final (Texto + Base)
    total_h_px = text_h_px + base_h_px
    final_canvas = Image.new('L', (target_w_px, total_h_px), 0)
    
    # Pegar texto arriba (Y=0)
    final_canvas.paste(img_stretched, (0, 0))
    
    # 4. Generar Máscaras
    mask_text = np.array(final_canvas) > 128
    
    # Base rectangular abajo (últimos 5mm)
    mask_base = np.zeros((total_h_px, target_w_px), dtype=bool)
    mask_base[text_h_px:, :] = True 
    
    mask_total = mask_text | mask_base
    
    # 5. Z-MAP (Extrusión Fija 50mm)
    z_map = np.zeros((total_h_px, target_w_px))
    
    # El texto tiene 50mm de profundidad
    z_map[mask_text] = EXTRUSION_MM
    
    # La base tiene 50mm + 2mm (52mm) para asegurar estabilidad y fusión
    z_map[mask_base] = EXTRUSION_MM + 2.0
    
    return z_map, mask_total

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
    st.markdown("#### Base de Nombre (Ajuste Automático a 180mm)")
    
    st.info("El texto se estirará o comprimirá para ocupar exactamente 180mm de ancho y 50mm de alto.")

    if st.button("🔤 Generar Placa"):
        with st.spinner("Deformando texto y generando bloque sólido..."):
            
            z_text_map, mask_total = procesar_texto_stretch(texto_usuario)
            faces_text = generar_stl_manifold(z_text_map, mask_total)
            
            if len(faces_text) > 0:
                text_mesh = mesh.Mesh(np.zeros(faces_text.shape[0], dtype=mesh.Mesh.dtype))
                text_mesh.vectors = faces_text
                
                # ROTACIÓN MANUAL (De pie)
                vec_y = text_mesh.vectors[:, :, 1].copy()
                vec_z = text_mesh.vectors[:, :, 2].copy()
                text_mesh.vectors[:, :, 1] = vec_z 
                text_mesh.vectors[:, :, 2] = vec_y 
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_t:
                    text_mesh.save(tmp_t.name)
                    
                    st.success(f"¡Placa Generada! 180mm x 50mm x 50mm")
                    
                    st.subheader("Vista Previa")
                    stl_from_file(file_path=tmp_t.name, material="material", auto_rotate=True, height=250)
                    
                    with open(tmp_t.name, "rb") as f_t:
                        st.download_button(
                            label=f"📥 Descargar Placa '{texto_usuario}'", 
                            data=f_t, 
                            file_name=f"base_180mm_{texto_usuario}.stl"
                        )
            else:
                st.error("Error: Geometría vacía.")
