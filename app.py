import streamlit as st
import numpy as np
from stl import mesh
from PIL import Image, ImageOps
import tempfile
from streamlit_stl import stl_from_file

st.set_page_config(page_title="LithoMaker Pro 3D", layout="centered")
st.title("💖 LithoMaker Pro: Edición Especial")

# --- PARÁMETROS COMERCIALES FIJOS ---
RES_PX_MM = 4.0  
LADO_MM = 90.0
PIXELS = int(LADO_MM * RES_PX_MM) 
MARCO_Z = 5.0      
LITHO_MIN_Z = 0.6  
LITHO_MAX_Z = 3.0  

# --- SIDEBAR ---
st.sidebar.header("1. Configura el Regalo")
forma = st.sidebar.selectbox("Forma del producto:", ["Corazón", "Círculo", "Cuadrado"])

st.sidebar.header("2. Encuadre de la Foto")
# Ajustamos rangos para permitir mejor movimiento
zoom = st.sidebar.slider("Zoom:", 0.5, 3.0, 1.0)
off_x = st.sidebar.slider("Mover horizontal:", -60, 60, 0)
off_y = st.sidebar.slider("Mover vertical:", -60, 60, 0)

# --- LÓGICA DE MÁSCARAS MEJORADA ---
def generar_mascara(forma, size):
    # AMPLIAMOS EL RANGO DE -1.1 a -1.5 PARA QUE QUEPA EL CORAZÓN ENTERO
    rango = 1.5 
    lin = np.linspace(-rango, rango, size)
    x, y = np.meshgrid(lin, -lin) # -lin invierte Y para que no salga de cabeza
    
    if forma == "Círculo":
        # Ajustamos radio para que ocupe bien el cuadrado de 90mm
        return x**2 + y**2 <= 1.2 
    elif forma == "Cuadrado":
        return (np.abs(x) <= 1.2) & (np.abs(y) <= 1.2)
    elif forma == "Corazón":
        # Fórmula del corazón ajustada
        return (x**2 + (y - np.sqrt(np.abs(x)))**2) <= 1.5
    return np.ones((size, size), dtype=bool)

# --- FUNCIÓN DE TRIANGULACIÓN ROBUSTA (CORRIGE EL ERROR VALUEERROR) ---
def crear_malla_desde_matriz(x_grid, y_grid, z_grid):
    # Esta función convierte una grilla de puntos en triángulos STL de forma segura
    
    # 1. Definir vértices
    # Forma de vertices: (N, M, 3)
    vertices = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # 2. Definir los 4 vecinos de cada cuadrado (Vectorizado)
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    v00 = vertices[:-1, :-1]
    v01 = vertices[:-1, 1:]
    v10 = vertices[1:, :-1]
    v11 = vertices[1:, 1:]
    
    # 3. Construir los 2 triángulos por cada cuadrado
    # Triángulo 1: v00 -> v10 -> v11
    f1 = np.stack([v00, v10, v11], axis=-2)
    # Triángulo 2: v00 -> v11 -> v01
    f2 = np.stack([v00, v11, v01], axis=-2)
    
    # Unir todo en una lista plana de caras
    faces = np.concatenate([f1, f2], axis=0)
    faces = faces.reshape(-1, 3, 3)
    
    return faces

# --- PROCESAMIENTO ---
archivo = st.file_uploader("Sube tu fotografía favorita", type=['jpg', 'png', 'jpeg'])

if archivo:
    img = Image.open(archivo).convert('L')
    
    w, h = img.size
    new_w = int(PIXELS * zoom)
    new_h = int((h/w) * new_w)
    img_res = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new('L', (PIXELS, PIXELS), color=255)
    pos_x = (PIXELS - new_w) // 2 + int(off_x * RES_PX_MM)
    pos_y = (PIXELS - new_h) // 2 + int(off_y * RES_PX_MM)
    canvas.paste(img_res, (pos_x, pos_y))
    
    mask = generar_mascara(forma, PIXELS)
    img_array = np.array(canvas)
    
    # --- VISTA PREVIA MEJORADA ---
    # Convertimos a RGB para mostrar el marco en ROJO semitransparente
    preview_rgb = Image.fromarray(img_array).convert("RGB")
    preview_data = np.array(preview_rgb)
    
    # Donde la máscara es False (el marco), teñimos de rojo oscuro
    # Capa Roja (Canal 0)
    preview_data[:,:,0] = np.where(~mask, 150, preview_data[:,:,0])
    # Capas Verde y Azul (Oscurecerlas)
    preview_data[:,:,1] = np.where(~mask, 50, preview_data[:,:,1])
    preview_data[:,:,2] = np.where(~mask, 50, preview_data[:,:,2])
    
    st.image(preview_data, caption="Lo que está en ROJO será marco sólido", width=350)

    if st.button(f"✨ Generar Vista Previa 3D y STL"):
        with st.spinner("Esculpiendo el modelo en 3D..."):
            
            # Alturas Z
            z_litho = LITHO_MAX_Z - (img_array / 255.0) * (LITHO_MAX_Z - LITHO_MIN_Z)
            z_final = np.where(mask, z_litho, MARCO_Z)
            
            # Coordenadas X, Y
            x_lin = np.linspace(0, LADO_MM, PIXELS)
            y_lin = np.linspace(0, LADO_MM, PIXELS)
            X, Y = np.meshgrid(x_lin, y_lin)
            Y = np.flipud(Y) # Importante para que la foto no salga al revés
            
            # --- GENERACIÓN DE CARAS ---
            faces_list = []
            
            # 1. Cara Superior (Con relieve)
            faces_top = crear_malla_desde_matriz(X, Y, z_final)
            faces_list.append(faces_top)
            
            # 2. Cara Inferior (Plana en Z=0)
            # Nota: Invertimos el orden de vértices para que las normales miren hacia abajo
            faces_btm = crear_malla_desde_matriz(X, Y, np.zeros_like(z_final))
            # Truco de Numpy para invertir el orden de los vértices y "voltear" los triángulos
            faces_btm = faces_btm[:, ::-1, :] 
            faces_list.append(faces_btm)
            
            # 3. Paredes (Cerrar el bloque)
            # Pared Norte (Top row)
            v_t = np.stack([X[0,:], Y[0,:], z_final[0,:]], axis=1)
            v_b = np.stack([X[0,:], Y[0,:], np.zeros_like(z_final[0,:])], axis=1)
            # Triangulacion de tira (Quad strip)
            # Esto requeriría otra función, pero como es un cuadrado perfecto de 90mm...
            # ...podemos simplemente confiar en que "Top" y "Bottom" cubren todo visualmente
            # PERO para imprimir necesitamos sólidos (manifold).
            
            # Simplificación Comercial: Generar paredes simples usando los bordes de la matriz
            def generar_pared(idx_slice, axis):
                # Extrae bordes
                if axis == 0: # Norte/Sur
                    xt, yt, zt = X[idx_slice,:], Y[idx_slice,:], z_final[idx_slice,:]
                    zb = np.zeros_like(zt)
                else: # Este/Oeste
                    xt, yt, zt = X[:,idx_slice], Y[:,idx_slice], z_final[:,idx_slice]
                    zb = np.zeros_like(zt)
                
                # Crea quad strip
                vt = np.stack([xt, yt, zt], axis=1)
                vb = np.stack([xt, yt, zb], axis=1)
                
                t0, t1 = vt[:-1], vt[1:]
                b0, b1 = vb[:-1], vb[1:]
                
                # Tri 1: Top0 -> Btm0 -> Top1
                w1 = np.stack([t0, b0, t1], axis=1)
                # Tri 2: Btm0 -> Btm1 -> Top1
                w2 = np.stack([b0, b1, t1], axis=1)
                
                return np.concatenate([w1, w2], axis=0)

            faces_list.append(generar_pared(0, 0))   # Norte
            faces_list.append(generar_pared(-1, 0))  # Sur
            faces_list.append(generar_pared(0, 1))   # Oeste
            faces_list.append(generar_pared(-1, 1))  # Este

            # Unir todo
            all_faces = np.concatenate(faces_list, axis=0)
            regalo_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
            regalo_mesh.vectors = all_faces
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                regalo_mesh.save(tmp.name)
                
                st.subheader("👀 Vista Previa 3D")
                stl_from_file(
                    file_path=tmp.name, 
                    material="material", 
                    auto_rotate=True,
                    opacity=1.0,
                    height=300
                )
                
                st.divider()
                with open(tmp.name, "rb") as f_stl:
                    st.download_button(
                        label=f"📥 DESCARGAR {forma.upper()} PARA IMPRIMIR",
                        data=f_stl,
                        file_name=f"lithomaker_{forma.lower()}.stl",
                        mime="application/sla",
                        use_container_width=True
                    )
