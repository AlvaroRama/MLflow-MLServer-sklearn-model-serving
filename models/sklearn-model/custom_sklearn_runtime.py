import pandas as pd
from mlserver_sklearn import SKLearnModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec

class CustomSKLearnModel(SKLearnModel):
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # mlserver_sklearn.SKLearnModel.predict ya decodifica la request
        # y maneja la conversión a DataFrame si se especifica en model-settings.json.
        # Sin embargo, si el ColumnTransformer sigue quejándose,
        # lo haremos explícitamente aquí.

        # Decodificar el request. Esto debería darnos un array NumPy 2D.
        # Asumimos una única entrada como en tu script de inferencia.
        input_data_list = payload.inputs[0].data
        input_shape = payload.inputs[0].shape

        # Reconstruir el array NumPy
        import numpy as np
        # Ajustar la forma para np.array, ya que 'data' es una lista aplanada o lista de listas
        # Si input_data_list es una lista de listas, ya es 2D. Si es aplanada, hay que reshape.
        # Tu script de inferencia envía: "data": [[values]], lo cual es una lista de listas.
        if len(input_shape) == 2 and isinstance(input_data_list, list) and isinstance(input_data_list[0], list):
            # Ya es lista de listas, ideal para DataFrame
            numpy_input = np.array(input_data_list)
        else:
            # En caso de que se aplane o venga de otra forma inesperada
            numpy_input = np.array(input_data_list).reshape(input_shape)


        # Obtener los nombres de las columnas del model-settings.json
        # Accedemos a los parámetros de la primera entrada (inputs[0])
        column_names = self.settings.inputs[0].parameters.columns

        if column_names is None:
            raise ValueError(
                "Missing 'columns' parameter in model-settings.json input. "
                "A custom runtime requires column names for DataFrame conversion."
            )

        # Convertir a Pandas DataFrame explícitamente
        df_input = pd.DataFrame(numpy_input, columns=column_names)

        # Realizar la predicción usando el modelo cargado (que es tu pipeline)
        # El modelo es self.model en la clase base SKLearnModel
        predictions = self._model.predict(df_input)

        # Convertir las predicciones a un formato de salida de MLServer
        # Asumimos que las predicciones son un array NumPy 1D
        output_data = predictions.tolist()
        output_shape = [-1, 1] # Asumimos una única salida para cada entrada

        # Crear el objeto ResponseOutput
        output = ResponseOutput(
            name="prediction", # Nombre arbitrario para la salida
            datatype="FP64",
            shape=output_shape,
            data=output_data
        )

        return InferenceResponse(outputs=[output], model_name=self.settings.name)