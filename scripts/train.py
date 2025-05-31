import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import mlflow
import mlflow.sklearn
import sklearn
from mlflow.tracking import MlflowClient
import ssl
import joblib
import json
import os

# Evitar SSL errors en fetch_california_housing
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuración de MLflow ---
# Define dónde MLflow guardará los metadatos y artefactos de los experimentos.
# "file:///mlruns" indica que los guardará localmente en el directorio 'mlruns'.
mlflow.set_tracking_uri("file:///mlruns")
# Asigna un nombre al experimento. Todas las "runs" de este script se agruparán bajo este experimento.
mlflow.set_experiment("california_housing_experiment")

def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    return X, y

def get_preprocessor(numerical_cols):
    return ColumnTransformer([
        ('num', StandardScaler(), numerical_cols)
    ])

def main():
    # 1. Carga y partición de los datos
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Obtener los nombres de las columnas directamente de los datos de entrenamiento
    numerical_cols = X.columns.tolist() 
    preprocessor = get_preprocessor(numerical_cols)

    # 2. Grid Search
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # 3. Inicio de un MLflow Run
    # `mlflow.start_run()` crea una nueva "run" (ejecución de experimento) en MLflow.
    # El contexto `with` asegura que la run se cierre correctamente al finalizar.
    # `run_name` le da un nombre legible a esta ejecución en la UI de MLflow.
    with mlflow.start_run(run_name="GradientBoostingRegressor") as run:
        run_id = run.info.run_id

        # 3.1 Log del grid y métricas
        # Registra los mejores parámetros del modelo encontrados por Grid Search.
        mlflow.log_params(best_params)
        # Registra parámetros individuales.
        mlflow.log_param("random_state", 42)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("grid_param_keys", list(param_grid.keys()))
        # Registra métricas.
        mlflow.log_metric("best_cv_score", best_cv_score)

        # 3.2 Entrenamiento final y evaluación
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', best_model)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"MSE final: {mse:.4f}")
        mlflow.log_metric("mse", mse)  # Registra la métrica final MSE.

        # 3.3 Log del modelo como artefacto
        artifact_path = "gbr_model" # Define la subcarpeta dentro de la run para guardar el modelo.
        # `mlflow.sklearn.log_model` guarda el modelo entrenado como un artefacto.
        mlflow.sklearn.log_model(
            sk_model=pipeline, # El modelo (o pipeline) a guardar.
            artifact_path=artifact_path, # Ruta dentro de la run donde se guardará.
            input_example=X_test.iloc[:1] # Ejemplo de entrada para facilitar la inferencia
        )

        # 3.4 Registro en Model Registry
        registry_name = "GradientBoostingModel"
        # URI del modelo guardado como artefacto en la run actual.
        # Es el enlace directo al modelo que acaba de ser logged.
        model_uri = f"runs:/{run_id}/{artifact_path}"
        # Crea una instancia del cliente de MLflow para interactuar con el Tracking Server y el Model Registry.
        client = MlflowClient()
        
        try: # Intenta crear un modelo registrado. Si ya existe, simplemente continúa.
            client.create_registered_model(registry_name)
        except mlflow.exceptions.MlflowException: 
            # Ya estaba registrado
            pass
        
        # Crea una nueva versión del modelo bajo el nombre registrado.
        # Esto vincula el modelo guardado en esta run con el Model Registry,
        # permitiendo gestionar versiones y transiciones de etapas (Staging, Production).
        client.create_model_version(
            name=registry_name,
            source=model_uri,
            run_id=run_id
        )    
        
        # --- Guardado del modelo y su configuración para MLServer ---
        mlserver_model_dir = "models/sklearn-model"
        os.makedirs(mlserver_model_dir, exist_ok=True)

        # Guarda el pipeline como archivo joblib
        joblib.dump(pipeline, os.path.join(mlserver_model_dir, "model.joblib"))

        # Crea el archivo model-settings.json para MLServer, incluyendo los nombres de las columnas
        model_settings = {
            "name": "sklearn-model",
            # Especifica la implementación del runtime personalizado para manejar DataFrames
            "implementation": "models.sklearn-model.custom_sklearn_runtime.CustomSKLearnModel", 
            "parameters": {
                "version": "v1.0.0"
            },
            "inputs": [
                {
                    "name": "input_data",
                    "datatype": "FP64",
                    "shape": [-1, len(numerical_cols)], # La forma debe coincidir con el número de columnas
                    "parameters": {
                        "columns": numerical_cols, # ¡Los nombres de las columnas son clave para el ColumnTransformer!
                        "use_dataframe": True     # Indica a MLServer que espere y trabaje con DataFrames
                    }
                }
            ]
        }
        
        with open(os.path.join(mlserver_model_dir, "model-settings.json"), "w") as f:
            json.dump(model_settings, f, indent=4)
        
        print(f"Modelo y configuración de MLServer guardados en {mlserver_model_dir}")

if __name__ == "__main__":
    main()