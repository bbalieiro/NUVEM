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

        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        # 3. Pipeline com imputação + dummy + zscore + KNN
        # --------------------------------------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]), num_cols),

                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols)
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
        # 5. Previsões + métricas
        # --------------------------------------------------------
        y_pred_all = modelo.predict(X)
        y_pred_test = modelo.predict(X_test)

        df["loan_predicted"] = y_pred_all

        # --------- MÉTRICAS ----------
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        logging.info(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        # --------------------------------------------------------
        # 6. Adicionar colunas de métricas no CSV
        # --------------------------------------------------------
        df["metric_mae"] = mae
        df["metric_mse"] = mse
        df["metric_rmse"] = rmse
        df["metric_r2"] = r2

        # --------------------------------------------------------
        # 7. Gerar CSV e enviar para outro container
        # --------------------------------------------------------
        output_csv = df.to_csv(index=False).encode("utf-8")

        conn_str = os.environ["AzureWebJobsStorage"]
        blob_service = BlobServiceClient.from_connection_string(conn_str)

        output_container = "dadosprocessados"
        output_name = os.path.basename(myblob.name)

        blob_client = blob_service.get_blob_client(
            container=output_container,
            blob=output_name
        )

        blob_client.upload_blob(output_csv, overwrite=True)

        logging.info(
            f"Arquivo processado salvo em: {output_container}/{output_name}"
        )

        logging.info("Processamento finalizado com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        raise
