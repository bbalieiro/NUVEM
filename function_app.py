import azure.functions as func
import logging
import os
from io import BytesIO

app = func.FunctionApp()

# ================================================
#   BLOB TRIGGER — KNN (Regressão) + EXPORT PARA OUTRO BLOB
# ================================================
@app.function_name(name="finance_knn_blob")
@app.blob_trigger(
    arg_name="myblob",
    path="dadosbrutos/{name}",
    connection="AzureWebJobsStorage"
)
def finance_knn_blob(myblob: func.InputStream):

    logging.info(f"Recebido arquivo: {myblob.name}")

    try:
        import pandas as pd
        import numpy as np

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        from azure.storage.blob import BlobServiceClient

    except Exception as e:
        logging.error(f"Erro ao importar bibliotecas: {e}")
        return

    try:
        # --------------------------------------------------------
        # 1. Ler CSV do Blob
        # --------------------------------------------------------
        df = pd.read_csv(BytesIO(myblob.read()))
        logging.info(f"Lidas {df.shape[0]} linhas e {df.shape[1]} colunas.")

        target = "loan_amount"
        if target not in df.columns:
            logging.error(f"A coluna alvo '{target}' não existe.")
            return

        # --------------------------------------------------------
        # 2. Separar X e y
        # --------------------------------------------------------
        X = df.drop(columns=[target])
        y = df[target]

        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        logging.info(f"Numéricas: {num_cols}")
        logging.info(f"Categóricas: {cat_cols}")

        # --------------------------------------------------------
        # 3. Pipeline de preparação + KNN Regressor
        # --------------------------------------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ]
        )

        modelo = Pipeline(steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsRegressor(n_neighbors=5))
        ])

        # --------------------------------------------------------
        # 4. Train/Test e treinamento
        # --------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        modelo.fit(X_train, y_train)

        # --------------------------------------------------------
        # 5. Gerar previsões para o dataset completo
        # --------------------------------------------------------
        df["loan_predicted"] = modelo.predict(X)

        logging.info("Previsões geradas com sucesso.")

        # --------------------------------------------------------
        # 6. Salvar CSV em buffer
        # --------------------------------------------------------
        output_csv = df.to_csv(index=False).encode("utf-8")

        # --------------------------------------------------------
        # 7. Enviar para outro container
        # --------------------------------------------------------
        conn_str = os.environ["AzureWebJobsStorage"]
        blob_service = BlobServiceClient.from_connection_string(conn_str)

        output_container = "dadosprocessados"
        output_name = os.path.basename(myblob.name)

        blob_client = blob_service.get_blob_client(container=output_container, blob=output_name)

        blob_client.upload_blob(output_csv, overwrite=True)

        logging.info(
            f"Arquivo processado salvo em: {output_container}/{output_name}"
        )

        logging.info("Processamento finalizado com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        raise
