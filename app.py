import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw, ImageFont
import tempfile
from streamlit_stl import stl_from_file
import math

# Configuración de página
st.set_page_config(page_title="LithoMaker Pro 2026", layout="centered")
st.title("💎 LithoMaker Pro: Suite de Diseño")

# --- PARÁMETROS DE INGENIERÍA ---
RES_PX_MM = 5.0 
MARCO_Z = 5.0      
LITHO_MIN_Z = 0.6  
LITHO_MAX_Z = 3.0  

# --- PERSISTENCIA DE DATOS ---
if 'litho_mesh' not in st.session_state:
    st.session_state['litho_mesh'] = None
if 'text_mesh' not in st.session_state:
    st.session_state['text_mesh'] = None

# --- SIDEBAR ---
st.sidebar.header("1. Configuración de Foto")
forma = st.sidebar.selectbox("Forma:", ["Corazón", "Círculo", "Cuadrado"])
ancho_marco = st.sidebar.slider("Ancho Marco (mm):", 2.0, 5.0, 3.0)

st.sidebar.header("2. Encuadre")
zoom = st.sidebar.slider("Zoom:", 0.5, 3.0, 1.2)
off_x = st.sidebar.slider("Mover X:", -60, 60, 0)
off_y = st.sidebar.slider("Mover Y:", -60, 60, 0)

st.sidebar.divider()
st.sidebar.header("3. Configuración Placa")
texto_usuario = st.sidebar.text_input("Nombre para la base:", "ALEJANDRA").upper()

# --- FUNCIONES ---

def cargar_fuente(size_px):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size_px))
    except: return ImageFont.load_default()

def obtener_mascaras(forma_tipo, size, border_mm):
    rango = 2.0 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    units_per_mm = (rango * 2) / 90.0
    offset = border_mm * units_per_mm
    if forma_tipo == "Círculo":
        R = 1.3
        return (x**2 + y**2) <= (R - offset)**2, (x**2 + y**2) <= R**2
    elif forma_tipo == "Cuadrado":
        L = 1.3
        return (np.abs(x) <= (L - offset)) & (np.abs(y) <= (L - offset)), (np.abs(x) <= L) & (np.abs(y) <= L)
    elif forma_tipo == "Corazón":
        def heart(cx, cy): return (cx**2 + (cy - 0.6 * np.sqrt(np.abs(cx)))**2)
        return heart(x, y) <= (1.6 - offset*1.8), heart(x, y) <= 1.6
    return np.ones((size,size)), np.ones((size,size))

def generar_stl_manifold(z_grid, mask_total):
    filas, cols = z_grid.shape
    ancho_mm, alto_mm = cols / RES_PX_MM, filas / RES_PX_MM
    x_lin, y_lin = np.linspace(0, ancho_mm, cols), np.linspace(0, alto_mm, filas)
    X, Y = np.meshgrid(x_lin, y_lin); Y = np.flipud(Y) 
    faces = []
    v_px = np.argwhere(mask_total)
    for i, j in v_px:
        if i >= filas-1 or j >= cols-1: continue
        z0,z1,z2,z3 = z_grid[i,j], z_grid[i,j+1], z_grid[i+1,j], z_grid[i+1,j+1]
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
    px_w, px_ht, px_hb = int(180*RES_PX_MM), int(30*RES_PX_MM), int(5*RES_PX_MM)
    canvas = Image.new('L', (px_w, px_ht + px_hb), 0)
    n_chars = len(texto); cw = px_w // n_chars
    font = cargar_fuente(250)
    for i, letra in enumerate(texto):
        cimg = Image.new('L', (500, 500), 0); d = ImageDraw.Draw(cimg); d.text((50, 50), letra, font=font, fill=255)
        bbox = cimg.getbbox()
        if bbox:
            l_rec = cimg.crop(bbox).resize((cw, px_ht), Image.Resampling.LANCZOS)
            canvas.paste(l_rec, (i * cw, 0))
    m_t = np.array(canvas) > 128
    m_b = np.zeros((px_ht + px_hb, px_w), dtype=bool); m_b[px_ht:, :] = True 
    z = np.zeros((px_ht + px_hb, px_w))
    z[m_t], z[m_b] = 30.0, 31.0
    return z, m_t | m_b

# --- INTERFAZ TABS ---
t1, t2, t3 = st.tabs(["🖼️ 1. Foto / Litofanía", "🔤 2. Base / Placa", "🧩 3. Escenario 3D"])

with t1:
    archivo = st.file_uploader("Cargar imagen", type=['jpg', 'png', 'jpeg'])
    if archivo:
        PX_S = int(90.0 * RES_PX_MM)
        img_orig = Image.open(archivo).convert('L')
        img_r = img_orig.resize((int(PX_S*zoom), int((img_orig.height/img_orig.width)*PX_S*zoom)), Image.Resampling.LANCZOS)
        canv = Image.new('L', (PX_S, PX_S), color=255)
        canv.paste(img_r, ((PX_S - img_r.width)//2 + int(off_x*RES_PX_MM), (PX_S - img_r.height)//2 + int(off_y*RES_PX_MM)))
        ml, mf = obtener_mascaras(forma, PX_S, ancho_marco)
        img_a = np.array(canv)
        prev = np.array(Image.fromarray(img_a).convert("RGB"))
        prev[mf & ~ml] = [200, 50, 50]; prev[~mf] = [30, 30, 30]
        st.image(prev, caption="Encuadre", width='stretch') # [cite: 7]
        if st.button("🚀 Procesar Foto", width='stretch'):
            z_f = np.where(ml, 3.0 - (img_a/255.0)*2.4, 5.0)
            faces = generar_stl_manifold(z_f, mf)
            m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)); m.vectors = faces
            m.rotate([1, 0, 0], math.radians(90))
            st.session_state['litho_mesh'] = m
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                m.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("📥 Descargar STL Foto", f, f"litofania_{forma}.stl", width='stretch')

with t2:
    st.write(f"Texto actual: **{texto_usuario}**")
    if st.button("🔤 Procesar Placa", width='stretch'):
        with st.spinner("Modelando base..."):
            z_m, m_t = procesar_texto_nameplate_normalizado(texto_usuario)
            faces = generar_stl_manifold(z_m, m_t)
            m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)); m.vectors = faces
            m.rotate([1, 0, 0], math.radians(90))
            st.session_state['text_mesh'] = m
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_t:
                m.save(tmp_t.name)
                with open(tmp_t.name, "rb") as ft:
                    st.download_button("📥 Descargar STL Placa", ft, f"placa_{texto_usuario}.stl", width='stretch')

with t3:
    st.header("Visualización de Ensamble")
    if st.session_state['litho_mesh'] is not None and st.session_state['text_mesh'] is not None:
        px_off = st.slider("Ajuste Lateral (mm):", 0, 90, 45)
        pz_off = st.slider("Ajuste Altura (mm):", -5, 5, 2)
        
        if st.button("🧩 Ver Ensamble", width='stretch'):
            m_litho = mesh.Mesh(st.session_state['litho_mesh'].data.copy())
            m_text = mesh.Mesh(st.session_state['text_mesh'].data.copy())
            m_litho.x += px_off 
            m_litho.z += pz_off
            combined = mesh.Mesh(np.concatenate([m_litho.data, m_text.data]))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_v:
                combined.save(tmp_v.name)
                stl_from_file(file_path=tmp_v.name, height=400, auto_rotate=True)
                st.info("💡 Recuerda: Descarga las piezas por separado en las pestañas 1 y 2 para laminarlas a tu gusto.")
    else:
        st.warning("⚠️ Genera la foto y la placa en sus respectivas pestañas para ver el ensamble.")
