import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageOps
import tempfile
from streamlit_stl import stl_from_file

# Configuración
st.set_page_config(page_title="LithoMaker Pro Commercial", layout="centered")
st.title("💎 LithoMaker Pro: Geometría Sólida")

# --- PARÁMETROS ---
# Bajamos un poco la resolución por defecto para asegurar que cierre bien los bordes sin saturar
RES_PX_MM = 3.0  
LADO_MM = 90.0
PIXELS = int(LADO_MM * RES_PX_MM) 
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

# --- MÁSCARAS DE PRECISIÓN ---
def obtener_mascaras(forma, size, border_mm):
    rango = 1.6 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin)
    
    # Factor de conversión para el borde en mm
    offset = border_mm * (rango * 2 / LADO_MM) # Aprox conversion

    if forma == "Círculo":
        R = 1.3
        mask_frame = x**2 + y**2 <= R**2
        mask_litho = x**2 + y**2 <= (R - offset)**2
    elif forma == "Cuadrado":
        L = 1.3
        mask_frame = (np.abs(x) <= L) & (np.abs(y) <= L)
        mask_litho = (np.abs(x) <= (L - offset)) & (np.abs(y) <= (L - offset))
    elif forma == "Corazón":
        # Fórmula ajustada para buen cierre
        def heart(cx, cy):
            return (cx**2 + (cy - 0.8 * np.sqrt(np.abs(cx)))**2)
        # Ajustamos umbrales
        mask_frame = heart(x, y) <= 1.4
        mask_litho = heart(x, y) <= (1.4 - offset*1.5) 
    
    return mask_litho, mask_frame

# --- GENERADOR DE MALLA SÓLIDA (WATERTIGHT) ---
def generar_stl_solido(Z_map, mask_valid):
    """
    Genera un sólido cerrado usando la máscara para definir bordes.
    Z_map: Alturas superiores.
    mask_valid: Booleano indicando qué pixeles son parte del objeto.
    """
    filas, cols = Z_map.shape
    
    # Generamos coordenadas físicas
    x_lin = np.linspace(0, LADO_MM, cols)
    y_lin = np.linspace(0, LADO_MM, filas)
    X, Y = np.meshgrid(x_lin, y_lin)
    Y = np.flipud(Y) # Orientación imagen
    
    faces = []
    
    # Arrays auxiliares para acceso rápido
    # Vértices Top: (x, y, z)
    # Vértices Bottom: (x, y, 0)
    
    # Iteramos solo dentro de los límites seguros (evitando bordes de matriz)
    # Usamos índices para velocidad
    idxs = np.argwhere(mask_valid[:-1, :-1]) # Pixeles válidos to-left
    
    for i, j in idxs:
        # Verificamos que el cuadro completo de 4 pixeles sea válido para crear la tapa/base
        if mask_valid[i+1, j] and mask_valid[i, j+1] and mask_valid[i+1, j+1]:
            
            # --- 1. TAPAS Y BASES ---
            # Coordenadas de los 4 puntos del pixel
            # Top (Relieve)
            vt00 = [X[i,j], Y[i,j], Z_map[i,j]]
            vt10 = [X[i+1,j], Y[i+1,j], Z_map[i+1,j]]
            vt01 = [X[i,j+1], Y[i,j+1], Z_map[i,j+1]]
            vt11 = [X[i+1,j+1], Y[i+1,j+1], Z_map[i+1,j+1]]
            
            # Bottom (Plano Z=0)
            vb00 = [X[i,j], Y[i,j], 0]
            vb10 = [X[i+1,j], Y[i+1,j], 0]
            vb01 = [X[i,j+1], Y[i,j+1], 0]
            vb11 = [X[i+1,j+1], Y[i+1,j+1], 0]
            
            # Cara Superior (2 triángulos)
            faces.append([vt00, vt10, vt11])
            faces.append([vt00, vt11, vt01])
            
            # Cara Inferior (2 triángulos, orden invertido para normales hacia abajo)
            faces.append([vb00, vb11, vb10])
            faces.append([vb00, vb01, vb11])
            
            # --- 2. PAREDES LATERALES (DETECCIÓN DE BORDES) ---
            # Verificamos vecinos para ver si estamos en el borde del objeto
            # Si un vecino es FALSO (fuera de mascara), creamos pared
            
            # Pared Norte (Arriba, i-1)
            if i == 0 or not mask_valid[i-1, j]: 
                 # Conectar Top y Bottom en el borde norte del pixel
                 faces.append([vt00, vt01, vb01]) # Tri 1
                 faces.append([vt00, vb01, vb00]) # Tri 2
                 
            # Pared Sur (Abajo, i+2 que corresponde al sig loop, aqui chequeamos i+1 local)
            # Para simplificar: miramos si el vecino de abajo (i+1) será invalido en su propio ciclo?
            # Mejor lógica: Construimos pared "Sur" del pixel actual
            # La pared sur conecta (i+1,j) con (i+1,j+1)
            if (i+1) == (filas-1) or not mask_valid[i+2, j]: # Aprox check
                 pass # Lógica compleja, simplificamos:
            
            # LÓGICA DE PARED SIMPLIFICADA Y ROBUSTA:
            # En lugar de predecir, verificamos explícitamente los 4 lados del QUAD actual
            
            # Lado OESTE (Izquierda) -> Conecta (i,j) con (i+1,j)
            if j == 0 or not mask_valid[i, j-1]:
                faces.append([vt00, vb00, vb10])
                faces.append([vt00, vb10, vt10])

            # Lado ESTE (Derecha) -> Conecta (i,j+1) con (i+1,j+1)
            if (j+1) == (cols-1) or not mask_valid[i, j+2]: # j+2 pq j es indice, mask es j+1
                 pass # Check neighbors logic is tricky with indices.
    
    # --- RE-BARRIDO SOLO PARA PAREDES (MÁS SEGURO) ---
    # Iteramos buscando transiciones True -> False
    
    # Paredes Verticales (Transición horizontal)
    for i in range(filas-1):
        for j in range(cols-1):
            curr = mask_valid[i,j]
            right = mask_valid[i,j+1]
            down = mask_valid[i+1,j]
            
            # Pared derecha (Si yo soy valido y el de la derecha no)
            if curr and not right:
                # Vértices del borde derecho: (i, j+1) a (i+1, j+1) ?? No, es en j
                # El borde está en J. 
                # Puntos: Top(i,j+1) -> Top(i+1,j+1) ... no, usamos los vertices del pixel actual
                
                # Vertices derechos del pixel actual:
                tr = [X[i,j+1], Y[i,j+1], Z_map[i,j+1]] # Top Right
                br = [X[i+1,j+1], Y[i+1,j+1], Z_map[i+1,j+1]] # Btm Right (Top index + 1 row)
                
                # Corrección: Una pared conecta el Top con el Bottom en la misma coordenada
                pt_top = [X[i,j+1], Y[i,j+1], Z_map[i,j+1]]
                pt_btm = [X[i,j+1], Y[i,j+1], 0]
                pt_top_next = [X[i+1,j+1], Y[i+1,j+1], Z_map[i+1,j+1]]
                pt_btm_next = [X[i+1,j+1], Y[i+1,j+1], 0]
                
                # Quad mirando a la derecha
                faces.append([pt_top, pt_btm, pt_btm_next])
                faces.append([pt_top, pt_btm_next, pt_top_next])

            # Pared izquierda (Si yo soy valido y el de la izquierda no)
            if curr and (j==0 or not mask_valid[i,j-1]):
                pt_top = [X[i,j], Y[i,j], Z_map[i,j]]
                pt_btm = [X[i,j], Y[i,j], 0]
                pt_top_next = [X[i+1,j], Y[i+1,j], Z_map[i+1,j]]
                pt_btm_next = [X[i+1,j], Y[i+1,j], 0]
                
                # Quad mirando a la izquierda
                faces.append([pt_top, pt_top_next, pt_btm_next])
                faces.append([pt_top, pt_btm_next, pt_btm])
            
            # Pared Abajo (Si yo soy valido y el de abajo no)
            if curr and not down:
                pt_top = [X[i+1,j], Y[i+1,j], Z_map[i+1,j]]
                pt_btm = [X[i+1,j], Y[i+1,j], 0]
                pt_top_next = [X[i+1,j+1], Y[i+1,j+1], Z_map[i+1,j+1]]
                pt_btm_next = [X[i+1,j+1], Y[i+1,j+1], 0]
                
                faces.append([pt_top, pt_top_next, pt_btm_next])
                faces.append([pt_top, pt_btm_next, pt_btm])
                
            # Pared Arriba (Si yo soy valido y el de arriba no)
            if curr and (i==0 or not mask_valid[i-1,j]):
                pt_top = [X[i,j], Y[i,j], Z_map[i,j]]
                pt_btm = [X[i,j], Y[i,j], 0]
                pt_top_next = [X[i,j+1], Y[i,j+1], Z_map[i,j+1]]
                pt_btm_next = [X[i,j+1], Y[i,j+1], 0]
                
                faces.append([pt_top, pt_btm, pt_btm_next])
                faces.append([pt_top, pt_btm_next, pt_top_next])

    return np.array(faces)

# --- PROCESAMIENTO ---
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
    preview[~m_litho & m_frame] = [200, 50, 50] 
    preview[~m_frame] = [20, 20, 20]           
    st.image(preview, caption="Rojo=Marco, Negro=Recorte", width=350)

    if st.button(f"🚀 Generar {forma} Sólido"):
        with st.spinner("Creando geometría cerrada (Manifold)..."):
            # Mapa de Alturas
            z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
            z_final = np.where(m_litho, z_litho, MARCO_Z)
            
            # Generación Sólida
            # Pasamos 'm_frame' como la máscara válida. Todo lo que esté fuera de m_frame se borra.
            all_faces = generar_stl_solido(z_final, m_frame)
            
            regalo_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
            regalo_mesh.vectors = all_faces
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                regalo_mesh.save(tmp.name)
                st.subheader("👀 Revisión 3D")
                stl_from_file(file_path=tmp.name, auto_rotate=True, height=300)
                
                with open(tmp.name, "rb") as f_stl:
                    st.download_button(
                        label="📥 DESCARGAR STL CERRADO",
                        data=f_stl,
                        file_name=f"litho_solida_{forma.lower()}.stl",
                        mime="application/sla",
                        width='stretch' # Corregido para nuevas versiones
                    )
