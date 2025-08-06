import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import load_and_prepare_data, predict_category, get_cluster_definitions

# ------------------ Configuración de la página principal ------------------
# This must be the first Streamlit command called.
st.set_page_config(
    page_title="Clasificador de Alimentos",
    page_icon=":hamburger:",
    layout="wide"
)

st.title("Clasificador de Alimentos de Burger King 🍔")
st.markdown("""
---
Esta aplicación utiliza un modelo de K-Means para clasificar alimentos en 3 categorías 
en función de su contenido de calorías y colesterol.
""")

# We use st.cache_data so this function runs only once
# and doesn't reload every time the user interacts with the app.
@st.cache_data
def get_data_and_model():
    """
    Loads and prepares the data and the K-Means model.
    """
    return load_and_prepare_data()

# Load data and the model
df_with_clusters, scaler, kmeans_model = get_data_and_model()

# If the data could not be loaded correctly, show an error message and exit.
if df_with_clusters is None:
    st.error("Error: No se pudo cargar el archivo 'p_items/burger_simplify.csv'. Por favor, verifica la ruta.")
else:
    # Define custom colors for the clusters
    # You can customize these colors as you wish
    cluster_colors = {
        "Alimento de consumo frecuente": 'khaki',
        "Alimento de consumo moderado":  'orange',
        "Alimento de consumo ocasional": 'red'
    }

    # ------------------ Sidebar (Side panel) ------------------
    st.sidebar.header("Parámetros del Alimento")
    st.sidebar.image("p_items/burguer_restaurant.png", caption="")

    # Sliders for user input values
    calories_input = st.sidebar.slider(
        "Calorías (kcal)",
        min_value=int(df_with_clusters['Calories'].min()),
        max_value=int(df_with_clusters['Calories'].max()),
        value=500
    )
    cholesterol_input = st.sidebar.slider(
        "Colesterol (mg)",
        min_value=int(df_with_clusters['Cholesterol (mg)'].min()),
        max_value=int(df_with_clusters['Cholesterol (mg)'].max()),
        value=50
    )

    # ------------------ Main application content ------------------

    # Button to predict the food category
    if st.sidebar.button("Clasificar Alimento"):
        # Predict the category of the entered food
        predicted_category = predict_category(calories_input, cholesterol_input, scaler, kmeans_model)

        # ------------------ Cluster visualization ------------------
        st.subheader("Gráfico de Clusters")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x='Calories', 
            y='Cholesterol (mg)', 
            hue='Etiqueta_Cluster', 
            data=df_with_clusters, 
            palette=cluster_colors, 
            ax=ax,
            s=80
        )
        ax.set_title("Clusters de Productos de Burger King")
        ax.set_xlabel("Calorías (kcal)")
        ax.set_ylabel("Colesterol (mg)")

        # Get the color of the predicted cluster from the custom palette
        predicted_color = cluster_colors.get(predicted_category, 'black')

        # Mark the user's point with an 'X'
        ax.scatter(
            calories_input, 
            cholesterol_input, 
            color=predicted_color, 
            marker='X', 
            s=300, 
            label=f"Tu alimento: '{predicted_category}'"
        )
        ax.legend(title="Categorías")
        
        st.pyplot(fig)

        # ------------------ Results and definitions ------------------
        st.subheader("Resultados de la Clasificación")

        # Prediction message
        st.markdown(f"### **Dado el contenido de calorías ({calories_input} kcal) y colesterol ({cholesterol_input} mg), el alimento entra en la categoría de:**")
        st.markdown(f"## **<span style='color:{predicted_color}'>{predicted_category}</span>**", unsafe_allow_html=True)
        st.markdown("---")

        # Cluster definitions
        st.subheader("Definiciones de las Categorías")
        cluster_definitions = get_cluster_definitions()
        for label, definition in cluster_definitions.items():
            st.markdown(f"**{label}**")
            st.write(definition)
            st.write("") # Blank space to improve readability
