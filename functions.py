import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Definimos las etiquetas de los clusters
CLUSTER_LABELS = {
    0: "Alimento de consumo frecuente",
    1: "Alimento de consumo moderado",
    2: "Alimento de consumo ocasional"
}

def load_and_prepare_data():
    """
    Carga el dataframe desde la ruta predefinida, lo prepara y entrena el modelo K-Means.
    
    Returns:
        tuple: Una tupla con el dataframe original con los clusters, el escalador y el modelo.
    """
    try:
        # Cargar el dataframe desde la nueva ruta
        df = pd.read_csv("p_items/burger_simplify.csv")
    except FileNotFoundError:
        return None, None, None

    # Seleccionar solo las columnas de interés para el clustering
    features = ['Calories', 'Cholesterol (mg)']
    df_features = df[features]

    # Normalizar los datos
    # Creamos una instancia del escalador
    scaler = StandardScaler()
    # Ajustamos y transformamos los datos
    df_scaled = scaler.fit_transform(df_features)

    # Entrenar el modelo K-Means con 3 clusters (valor óptimo encontrado previamente)
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_model.fit(df_scaled)

    # Asignar los clusters al dataframe original
    df['Cluster'] = kmeans_model.labels_

    # Mapeamos los clusters a etiquetas más descriptivas
    # Verificamos los centroides para asignar las etiquetas correctamente
    centroids = kmeans_model.cluster_centers_
    # Creamos un dataframe temporal para ordenar los centroides por calorías
    centroid_df = pd.DataFrame(centroids, columns=features)
    centroid_df['Cluster_Index'] = range(len(centroid_df))
    # Ordenamos por las calorías (menor a mayor) para asignar las etiquetas
    centroid_df_sorted = centroid_df.sort_values(by='Calories')
    
    # Creamos un mapeo de los índices de cluster a las etiquetas
    cluster_mapping = {
        centroid_df_sorted.iloc[0]['Cluster_Index']: CLUSTER_LABELS[0],
        centroid_df_sorted.iloc[1]['Cluster_Index']: CLUSTER_LABELS[1],
        centroid_df_sorted.iloc[2]['Cluster_Index']: CLUSTER_LABELS[2]
    }
    
    df['Etiqueta_Cluster'] = df['Cluster'].map(cluster_mapping)

    return df, scaler, kmeans_model

def get_cluster_definitions():
    """
    Retorna un diccionario con las definiciones de cada etiqueta de cluster.
    """
    return {
        "Alimento de consumo frecuente": "Productos con bajo contenido calórico y de colesterol. Pueden consumirse con mayor regularidad.",
        "Alimento de consumo moderado": "Productos con niveles intermedios de calorías y colesterol. Deben consumirse con moderación como parte de una dieta equilibrada.",
        "Alimento de consumo ocasional": "Productos con alto contenido calórico y de colesterol. Su consumo debe ser esporádico o en ocasiones especiales."
    }

def predict_category(calories, cholesterol, scaler, kmeans_model):
    """
    Predice la categoría de un nuevo alimento dado sus calorías y colesterol.
    
    Args:
        calories (int): El número de calorías del alimento.
        cholesterol (int): La cantidad de colesterol en mg del alimento.
        scaler (StandardScaler): El escalador ajustado con el que se normalizaron los datos.
        kmeans_model (KMeans): El modelo K-Means entrenado.
        
    Returns:
        str: La etiqueta de la categoría a la que pertenece el alimento.
    """
    # Creamos un dataframe con los nuevos datos para que el escalador pueda procesarlos
    new_data = pd.DataFrame([[calories, cholesterol]], columns=['Calories', 'Cholesterol (mg)'])
    
    # Escalar los nuevos datos usando el mismo escalador
    new_data_scaled = scaler.transform(new_data)
    
    # Predecir el cluster
    predicted_cluster = kmeans_model.predict(new_data_scaled)[0]
    
    # Retornar la etiqueta del cluster
    # Asignamos las etiquetas basadas en el mapeo que hicimos al inicio
    # Verificamos los centroides para asignar las etiquetas correctamente
    centroids = kmeans_model.cluster_centers_
    centroid_df = pd.DataFrame(centroids, columns=['Calories', 'Cholesterol (mg)'])
    centroid_df['Cluster_Index'] = range(len(centroid_df))
    centroid_df_sorted = centroid_df.sort_values(by='Calories')
    
    cluster_mapping = {
        centroid_df_sorted.iloc[0]['Cluster_Index']: CLUSTER_LABELS[0],
        centroid_df_sorted.iloc[1]['Cluster_Index']: CLUSTER_LABELS[1],
        centroid_df_sorted.iloc[2]['Cluster_Index']: CLUSTER_LABELS[2]
    }

    return cluster_mapping[predicted_cluster]

# Guardar el escalador y el modelo para su uso en la app.py
def save_model_and_scaler(scaler, model, model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
    """
    Guarda el modelo K-Means y el escalador a archivos.
    
    Args:
        scaler (StandardScaler): El escalador a guardar.
        model (KMeans): El modelo a guardar.
        model_path (str): La ruta del archivo para guardar el modelo.
        scaler_path (str): La ruta del archivo para guardar el escalador.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model_and_scaler(model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
    """
    Carga el modelo K-Means y el escalador desde archivos.
    
    Args:
        model_path (str): La ruta del archivo del modelo.
        scaler_path (str): La ruta del archivo del escalador.
        
    Returns:
        tuple: Una tupla con el modelo cargado y el escalador.
    """
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None
