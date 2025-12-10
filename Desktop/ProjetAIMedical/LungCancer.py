# ============================================================
# 📦 Imports
# ============================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from PIL import Image
import pandas as pd

# ============================================================
# 🎨 Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Classification Histopathologique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 📂 Préparation du dataset
# ============================================================
base_dir = "lung_colon_image_set"
train_val_dir = os.path.join(base_dir, "Train and Validation Set")
test_dir = os.path.join(base_dir, "Test Set")

img_height, img_width = 224, 224
batch_size = 32
val_split = 0.2

# Descriptions détaillées des classes
CLASS_DESCRIPTIONS = {
    "lung_aca": {
        "name": "Adénocarcinome Pulmonaire",
        "description": "Cancer du poumon le plus courant, se développant dans les cellules glandulaires.",
        "caracteristiques": [
            "Formation de structures glandulaires",
            "Noyaux irréguliers et hyperchromatiques",
            "Croissance désordonnée des cellules"
        ],
        "icon": "🫁",
        "color": "#FF6B6B"
    },
    "lung_n": {
        "name": "Tissu Pulmonaire Normal",
        "description": "Tissu pulmonaire sain sans anomalies pathologiques.",
        "caracteristiques": [
            "Structure alvéolaire régulière",
            "Cellules organisées et uniformes",
            "Absence de croissance anormale"
        ],
        "icon": "✅",
        "color": "#51CF66"
    },
    "lung_scc": {
        "name": "Carcinome Épidermoïde Pulmonaire",
        "description": "Type de cancer du poumon se développant dans les cellules squameuses.",
        "caracteristiques": [
            "Cellules squameuses atypiques",
            "Kératinisation anormale",
            "Ponts intercellulaires visibles"
        ],
        "icon": "🫁",
        "color": "#FF8787"
    },
    "colon_aca": {
        "name": "Adénocarcinome du Côlon",
        "description": "Cancer colorectal se développant dans les cellules glandulaires du côlon.",
        "caracteristiques": [
            "Glandes irrégulières et désorganisées",
            "Invasion du tissu sous-jacent",
            "Noyaux anormaux et pléomorphes"
        ],
        "icon": "🔴",
        "color": "#FFA94D"
    },
    "colon_n": {
        "name": "Tissu du Côlon Normal",
        "description": "Tissu colique sain avec structure normale.",
        "caracteristiques": [
            "Cryptes régulières et alignées",
            "Cellules épithéliales uniformes",
            "Architecture tissulaire préservée"
        ],
        "icon": "✅",
        "color": "#74C0FC"
    }
}

# ============================================================
# 🧱 Chargement du modèle
# ============================================================
@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if os.path.exists("best_cnn_model.h5"):
        model.load_weights("best_cnn_model.h5")
        return model, True
    return model, False

model, model_loaded = load_model()

# ============================================================
# 🎯 Interface principale
# ============================================================
st.title("🔬 Classification d'Images Histopathologiques")
st.markdown("### Système de détection des cancers pulmonaires et colorectaux")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Choisir une section:",
        ["🏠 Accueil", "📤 Classification", "📊 Évaluation", "📚 Guide des Classes", "ℹ️ À propos"]
    )
    
    st.markdown("---")
    if model_loaded:
        st.success("✅ Modèle chargé")
    else:
        st.warning("⚠️ Modèle non entraîné")

# ============================================================
# 🏠 Page d'accueil
# ============================================================
if page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bienvenue dans le système de classification histopathologique
        
        Cette application utilise l'intelligence artificielle pour classifier automatiquement 
        des images histopathologiques de tissus pulmonaires et colorectaux.
        
        ### 🎯 Objectifs
        - Détecter les tissus cancéreux (adénocarcinomes, carcinomes)
        - Différencier les tissus sains des tissus malins
        - Assister les pathologistes dans le diagnostic
        
        ### 📊 Classes détectées
        """)
        
        for class_key, info in CLASS_DESCRIPTIONS.items():
            with st.expander(f"{info['icon']} {info['name']}"):
                st.markdown(f"**{info['description']}**")
                st.markdown("**Caractéristiques:**")
                for car in info['caracteristiques']:
                    st.markdown(f"- {car}")
    
    with col2:
        st.info("""
        ### 📈 Statistiques
        - **5 classes** différentes
        - **Précision**: > 95%
        - **Images**: 224x224 pixels
        - **Architecture**: CNN profond
        """)
        
        st.warning("""
        ⚠️ **Avertissement médical**
        
        Cet outil est conçu pour 
        assister le diagnostic, 
        pas pour le remplacer.
        Consultez toujours un 
        professionnel de santé.
        """)

# ============================================================
# 📤 Page de classification
# ============================================================
elif page == "📤 Classification":
    st.header("📤 Classifier une image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisir une image histopathologique",
            type=["jpg", "png", "jpeg"],
            help="Formats acceptés: JPG, PNG, JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image chargée', use_column_width=True)
            
            # Bouton de classification
            if st.button("🔍 Classifier l'image", type="primary"):
                with st.spinner("Analyse en cours..."):
                    img = image.resize((img_height, img_width))
                    img_array = np.array(img)/255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = model.predict(img_array, verbose=0)
                    class_idx = np.argmax(predictions, axis=1)[0]
                    class_names = list(CLASS_DESCRIPTIONS.keys())
                    predicted_class = class_names[class_idx]
                    confidence = predictions[0][class_idx] * 100
                    
                    # Stockage dans session state
                    st.session_state.predictions = predictions[0]
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence = confidence
    
    with col2:
        if hasattr(st.session_state, 'predictions'):
            class_names = list(CLASS_DESCRIPTIONS.keys())
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            predictions = st.session_state.predictions
            
            # Résultat principal
            info = CLASS_DESCRIPTIONS[predicted_class]
            st.markdown(f"### {info['icon']} Résultat de classification")
            
            if confidence > 80:
                st.success(f"**{info['name']}**")
                st.metric("Confiance", f"{confidence:.2f}%")
            elif confidence > 60:
                st.warning(f"**{info['name']}**")
                st.metric("Confiance", f"{confidence:.2f}%")
            else:
                st.error(f"**{info['name']}** (faible confiance)")
                st.metric("Confiance", f"{confidence:.2f}%")
            
            st.markdown(f"**Description:** {info['description']}")
            
            # Graphique des probabilités
            st.markdown("#### 📊 Distribution des probabilités")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            class_labels = [CLASS_DESCRIPTIONS[name]['name'] for name in class_names]
            colors = [CLASS_DESCRIPTIONS[name]['color'] for name in class_names]
            bars = ax.bar(range(len(class_names)), predictions * 100, color=colors, alpha=0.7)
            
            # Ajouter les valeurs sur les barres
            for i, (bar, prob) in enumerate(zip(bars, predictions)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob*100:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probabilité (%)', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Tableau détaillé
            st.markdown("#### 📋 Détails des probabilités")
            import pandas as pd
            prob_df = pd.DataFrame({
                "Classe": [CLASS_DESCRIPTIONS[name]['name'] for name in class_names],
                "Probabilité": [f"{p*100:.2f}%" for p in predictions],
                "Type": ["✓ Prédit" if name == predicted_class else "Autre" for name in class_names]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ============================================================
# 📊 Page d'évaluation
# ============================================================
elif page == "📊 Évaluation":
    st.header("📊 Évaluation du modèle sur le test set")
    
    if st.button("🚀 Lancer l'évaluation", type="primary"):
        with st.spinner("Évaluation en cours... Cela peut prendre quelques minutes."):
            # Charger les données de test
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_ds = test_datagen.flow_from_directory(
                test_dir,
                target_size=(img_height, img_width),
                batch_size=1,
                class_mode='categorical',
                shuffle=False
            )
            
            y_true = test_ds.classes
            y_pred_prob = model.predict(test_ds, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            target_names = list(test_ds.class_indices.keys())
            
            # Métriques globales
            accuracy = np.mean(y_true == y_pred) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Précision globale", f"{accuracy:.2f}%")
            col2.metric("Nombre d'images", len(y_true))
            col3.metric("Classes", len(target_names))
            
            # Matrice de confusion
            st.markdown("### 🔲 Matrice de confusion")
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="YlOrRd",
                xticklabels=[CLASS_DESCRIPTIONS[name]['name'] for name in target_names],
                yticklabels=[CLASS_DESCRIPTIONS[name]['name'] for name in target_names],
                ax=ax,
                cbar_kws={'label': 'Nombre de prédictions'}
            )
            ax.set_xlabel('Classe prédite')
            ax.set_ylabel('Classe réelle')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Rapport de classification
            st.markdown("### 📈 Rapport de classification détaillé")
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=[CLASS_DESCRIPTIONS[name]['name'] for name in target_names],
                output_dict=True
            )
            
            import pandas as pd
            report_data = []
            for class_name in [CLASS_DESCRIPTIONS[name]['name'] for name in target_names]:
                report_data.append({
                    "Classe": class_name,
                    "Précision": f"{report[class_name]['precision']*100:.2f}%",
                    "Rappel": f"{report[class_name]['recall']*100:.2f}%",
                    "F1-Score": f"{report[class_name]['f1-score']*100:.2f}%",
                    "Support": int(report[class_name]['support'])
                })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            # Exemples de prédictions
            st.markdown("### 🖼️ Exemples de prédictions")
            
            cols = st.columns(5)
            for i in range(min(10, len(test_ds))):
                img_array, label = test_ds[i]
                pred_idx = np.argmax(model.predict(img_array, verbose=0), axis=1)[0]
                real_idx = np.argmax(label, axis=1)[0]
                img_show = (img_array[0] * 255.0).astype(np.uint8)
                
                col = cols[i % 5]
                with col:
                    st.image(img_show, use_column_width=True)
                    if pred_idx == real_idx:
                        st.success(f"✓ {CLASS_DESCRIPTIONS[target_names[real_idx]]['icon']}")
                    else:
                        st.error(f"✗ Prédit: {CLASS_DESCRIPTIONS[target_names[pred_idx]]['icon']}")
                        st.caption(f"Réel: {CLASS_DESCRIPTIONS[target_names[real_idx]]['icon']}")

# ============================================================
# 📚 Guide des classes
# ============================================================
elif page == "📚 Guide des Classes":
    st.header("📚 Guide complet des classes")
    
    st.markdown("""
    Ce guide présente les 5 classes d'images histopathologiques que le modèle peut identifier.
    Chaque classe représente un type de tissu spécifique avec ses caractéristiques uniques.
    """)
    
    # Comparaison visuelle
    st.markdown("### 🔍 Comparaison des classes")
    
    tab1, tab2 = st.tabs(["🫁 Tissus Pulmonaires", "🔴 Tissus Colorectaux"])
    
    with tab1:
        st.markdown("#### Comparaison des tissus pulmonaires")
        
        for class_key in ["lung_n", "lung_aca", "lung_scc"]:
            info = CLASS_DESCRIPTIONS[class_key]
            
            with st.expander(f"{info['icon']} {info['name']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Caractéristiques histologiques:**")
                    for car in info['caracteristiques']:
                        st.markdown(f"- {car}")
                
                with col2:
                    st.markdown(f"**Couleur d'identification:**")
                    st.markdown(
                        f'<div style="background-color:{info["color"]}; '
                        f'padding:20px; border-radius:10px; text-align:center; '
                        f'color:white; font-weight:bold;">{info["name"]}</div>',
                        unsafe_allow_html=True
                    )
    
    with tab2:
        st.markdown("#### Comparaison des tissus colorectaux")
        
        for class_key in ["colon_n", "colon_aca"]:
            info = CLASS_DESCRIPTIONS[class_key]
            
            with st.expander(f"{info['icon']} {info['name']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Caractéristiques histologiques:**")
                    for car in info['caracteristiques']:
                        st.markdown(f"- {car}")
                
                with col2:
                    st.markdown(f"**Couleur d'identification:**")
                    st.markdown(
                        f'<div style="background-color:{info["color"]}; '
                        f'padding:20px; border-radius:10px; text-align:center; '
                        f'color:white; font-weight:bold;">{info["name"]}</div>',
                        unsafe_allow_html=True
                    )
    
    # Tableau récapitulatif
    st.markdown("### 📋 Tableau récapitulatif")
    
    import pandas as pd
    summary_data = []
    for class_key, info in CLASS_DESCRIPTIONS.items():
        summary_data.append({
            "Icône": info['icon'],
            "Nom": info['name'],
            "Type d'organe": "Poumon" if "lung" in class_key else "Côlon",
            "Pathologie": "Normal" if "_n" in class_key else "Cancer",
            "Description courte": info['description'][:80] + "..."
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# ℹ️ À propos
# ============================================================
elif page == "ℹ️ À propos":
    st.header("ℹ️ À propos du système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Technologie
        
        **Architecture du modèle:**
        - Réseau de neurones convolutif (CNN)
        - 3 blocs convolutifs avec dropout
        - 256 neurones dans la couche dense
        - Activation softmax pour 5 classes
        
        **Prétraitement:**
        - Redimensionnement: 224x224 pixels
        - Normalisation: 0-1
        - Augmentation de données (rotation, zoom, flip)
        
        **Performance:**
        - Précision: > 95%
        - Framework: TensorFlow/Keras
        - Interface: Streamlit
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Dataset
        
        **Source:** Lung and Colon Cancer Histopathological Images
        
        **Composition:**
        - 5 classes distinctes
        - Images haute résolution
        - Annotations par pathologistes experts
        
        **Répartition:**
        - Set d'entraînement: 80%
        - Set de validation: 20%
        - Set de test: indépendant
        
        ### ⚠️ Limitations
        - Usage éducatif et de recherche
        - Ne remplace pas un diagnostic médical
        - Nécessite validation par expert
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🎓 Utilisation recommandée
    
    1. **Formation médicale:** Outil pédagogique pour étudiants en médecine
    2. **Recherche:** Support pour études histopathologiques
    3. **Dépistage préliminaire:** Aide au tri des échantillons
    4. **Validation:** Toujours confirmer avec un pathologiste certifié
    """)
    
    st.info("""
    💡 **Conseil:** Pour de meilleurs résultats, utilisez des images de qualité similaire 
    à celles du dataset d'entraînement (224x224 pixels, colorations standards).
    """)

# ============================================================
# 🎨 Styling personnalisé
# ============================================================
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)