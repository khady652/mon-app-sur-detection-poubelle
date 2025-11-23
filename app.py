import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(page_title="Détection de Poubelles", layout="wide")

# =================================================================
# 1. Configuration et Chargement du Modèle (Corrigé)
# =================================================================

st.title(" Bienvenu dans votre Application de Détection des Poubelles 🗑️")
st.markdown("---")

# Nom du fichier modèle
MODEL_FILE_NAME = "best.pt"

@st.cache_resource
def load_yolo_model():
    """
    Tente de charger le modèle YOLO.
    (La vérification os.path.exists est retirée pour éviter les problèmes de chemin Streamlit Cloud)
    """
    try:
        # Tenter de charger directement le modèle. YOLO trouvera le fichier s'il est dans le dépôt.
        model = YOLO(MODEL_FILE_NAME) 
        st.success("")
        return model
    except FileNotFoundError:
        # Affiche un message d'erreur spécifique si le fichier .pt est introuvable
        st.error(f"❌ Fichier modèle '{MODEL_FILE_NAME}' non trouvé dans le dépôt. Le traitement est impossible.")
        return None
    except Exception as e:
        # Affiche toute autre erreur lors de l'initialisation (problème de dépendance, etc.)
        st.error(f"❌ Erreur  lors du chargement du modèle : {e}")
        return None

# Le modèle est chargé au démarrage de l'application
model = load_yolo_model()

# =================================================================
# 2. Fonction de Prédiction
# =================================================================

def predict_and_draw(image):
    
    if model is None:
        return image, "Le traitement est impossible car le modèle n'a pas pu être chargé."
        
    np_image = np.array(image)
    results = model(np_image, verbose=False, conf=0.5)
    
    detections = results[0].boxes.cpu().numpy()
    
    if len(detections) > 0:
        messages = []
        for det in detections:
            class_id = int(det.cls[0])
            predicted_class = model.names.get(class_id, "CLASSE INCONNUE")
            confidence = det.conf[0]
            messages.append(f"{predicted_class.upper()} ")
        prediction_message = "j'ai detecter une  : " + ", ".join(messages)
    else:
        prediction_message = " desole! mais je n' ai detecte aucune poubelle  dans cette image."
        
    plotted_image_array = results[0].plot(labels=True, conf=True, line_width=3)
    processed_image = Image.fromarray(plotted_image_array)
    
    return processed_image, prediction_message


# =================================================================
# 3. Interface Streamlit
# =================================================================

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
    # Affiche le bouton uniquement si le fichier est trouvé
    pass 

st.sidebar.header("Instructions")
st.sidebar.info("1. Chargez une image de poubelle.\n2. Le modèle détecte la poubelle et son statut (Plein/Vide).")

st.markdown("## ⬇️ Télécharger une Image pour l'Analyse")
uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    # --- Début du traitement de l'image ---
    
    # Vérification essentielle : Si le modèle n'a pas chargé au démarrage, on arrête
    if model is None:
        st.error("⚠️ Le traitement est impossible car le modèle de détection n'est pas disponible.")
        # Utiliser st.stop() pour arrêter l'exécution du reste du script
        st.stop() 

    try:
        # 1. Lire et convertir l'image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Image Originale", use_column_width=True)
            
        # 2. Exécuter l'inférence
        with st.spinner("Analyse en cours : Détection et Classification par YOLO..."):
            processed_image, prediction_message = predict_and_draw(image)
        
        # 3. Afficher les résultats
        with col2:
             st.image(processed_image, caption="Résultat de la détection", use_column_width=True)
             st.info(prediction_message)

    except Exception as e:
        # Affiche toute erreur survenant pendant la lecture ou le traitement de l'image
        st.error(f"❌ Une erreur s'est produite lors du traitement de l'image : {e}")
        st.stop() # 💡 CORRECTION : st.stop() pour arrêter l'exécution après une erreur.