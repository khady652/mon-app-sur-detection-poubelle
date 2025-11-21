import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(page_title="Détection de Poubelles", layout="wide")
# =================================================================
# 3. Interface Streamlit
# =================================================================

st.title(" Bienvenu dans votre Application de Détection des Poubelles 🗑️")
st.markdown("---")

# charement de notre model
MODEL_FILE_NAME = "best (1).pt"

@st.cache_resource
def load_yolo_model():
    # REMPLACER le code existant par ceci :
    try:
        # Tenter de charger directement le modèle. 
        # Si le fichier est présent dans le repo, YOLO (ultralytics) le trouvera.
        model = YOLO(MODEL_FILE_NAME)
        st.success("✅ Modèle YOLO chargé avec succès.")
        return model
    except Exception as e:
        # Si le chargement échoue pour une raison (chemin, structure du fichier, etc.), 
        # afficher l'erreur pour le débogage.
        st.error(f"❌ Erreur critique lors du chargement du modèle YOLO '{MODEL_FILE_NAME}'. Vérifiez les logs : {e}")
        return None

# Ne modifiez pas le reste du code, il est correct.

model = load_yolo_model()

def predict_and_draw(image):
    
    if model is None:
        return image, "Le traitement est impossible car le modèle n'a pas pu être chargé."
        
    # --- 1. Préparation de l'image pour YOLO ---
    np_image = np.array(image)
    
    # --- 2. Exécution de l'Inférence ---
    results = model(np_image, verbose=False) 
    
    # --- 3. Extraction du Message de Prédiction ---
    detections = results[0].boxes.cpu().numpy()
    
    if len(detections) > 0:
        best_detection = detections[0]
        class_id = int(best_detection.cls[0])
        predicted_class = model.names[class_id] 
        confidence = best_detection.conf[0]
        
        prediction_message = (
            f"Le statut détecté est : **{predicted_class.upper()}** "
            f"avec une confiance de **{confidence:.2f}**."
        )
    else:
        prediction_message = "Aucune poubelle n'a été détectée dans cette image."
    # -----------------------------------------------------
    
    # --- 4. Tracé Automatique par YOLO ---
    plotted_image_array = results[0].plot(
        labels=True, 
        conf=True, 
        line_width=3
    )
    
    # 5. Conversion et Retour
    processed_image = Image.fromarray(plotted_image_array)
    
    return processed_image, prediction_message



# --- BOUTON DE TÉLÉCHARGEMENT DU MODÈLE (Côté) ---
st.sidebar.header("Fichier Modèle")
try:
    with open(MODEL_FILE_NAME, "rb") as file:
        st.sidebar.download_button(
            label="⬇️ Télécharger le modèle",
            data=file,
            file_name=MODEL_FILE_NAME,
            mime="application/octet-stream",
        )
except FileNotFoundError:
    pass 

st.sidebar.header("Instructions")
st.sidebar.info("1. Chargez une image de poubelle.\n2. Le modèle détecte la poubelle et son statut (Plein/Vide).")

st.markdown("## ⬇️ Télécharger une Image pour l'Analyse")
uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Traitement de l'image
    try:
        # 1. Lire l'image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Image Originale", use_column_width=True)
            
        # 2. Exécuter l'inférence
        with st.spinner("Analyse en cours : Détection et Classification par YOLO..."):
            #  RÉCUPÉRATION DES DEUX VALEURS 
            processed_image, prediction_message = predict_and_draw(image)
        
        # 3. Afficher les résultats
        with col2:
            st.image(processed_image, caption="poubelle detecté", use_column_width=True)
            
        #  LIGNE CORRIGÉE : AFFICHAGE DU MESSAGE DE PRÉDICTION 
        st.success(f"Analyse terminée ! La détection et le classement sont affichés ci-dessus. {prediction_message}")

    except Exception as e:
        st.error(f"❌ Une erreur s'est produite lors du traitement de l'image : {e}")