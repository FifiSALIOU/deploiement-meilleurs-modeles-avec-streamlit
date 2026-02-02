import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Détection de Véhicules",
    page_icon="🚗",
    layout="wide"
)

st.title("Détection de Véhicules avec YOLO 🚗")
st.write("Téléversez une image pour détecter les véhicules.")

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

# --- IMPORTANT ---
# REMPLACEZ CI-DESSOUS PAR LE CHEMIN VERS VOTRE MEILLEUR MODÈLE
# Par exemple : 'models/best.pt'
PATH_TO_BEST_MODEL = "models/best.pt" 
# -----------------

@st.cache_resource
def load_model(model_path):
    """
    Charge le modèle YOLO depuis le chemin spécifié.
    Utilise st.cache_resource pour ne charger le modèle qu'une seule fois.
    """
    if not os.path.exists(model_path):
        st.error(f"ERREUR : Le fichier du modèle n'a pas été trouvé à l'emplacement : {model_path}")
        st.error("Veuillez vérifier le chemin dans la variable 'PATH_TO_BEST_MODEL' du script.")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model(PATH_TO_BEST_MODEL)

if model is not None:
    st.success(f"Modèle chargé avec succès depuis : {PATH_TO_BEST_MODEL}")

    # ============================================================================
    # INTERFACE UTILISATEUR
    # ============================================================================

    uploaded_file = st.file_uploader(
        "Choisissez une image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Ouvrir l'image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image Originale")
            st.image(image, caption="Image téléversée.", use_container_width=True)

        # Exécuter la prédiction
        if st.button("Lancer la détection"):
            with st.spinner("Détection en cours..."):
                # Le modèle retourne une liste de résultats
                results = model(image)

                # `results[0].plot()` retourne une image (array numpy BGR) avec les détections dessinées
                result_image_np = results[0].plot()
                
                # Conversion de BGR (OpenCV) à RGB (PIL)
                result_image_pil = Image.fromarray(result_image_np[..., ::-1])

            with col2:
                st.subheader("Image avec Détections")
                st.image(result_image_pil, caption="Résultat de la détection.", use_container_width=True)
                
            # Afficher les détails des détections (optionnel)
            st.subheader("Détails des objets détectés")
            names = results[0].names
            for box in results[0].boxes:
                st.write(f"- **{names[int(box.cls)]}** (Confiance: {box.conf.item():.2f})")

else:
    st.warning("Le modèle n'a pas pu être chargé. L'application ne peut pas fonctionner.")
    