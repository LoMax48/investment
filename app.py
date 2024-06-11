from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import os

app = Flask(__name__)

# Подключение к базе данных PostgreSQL
DATABASE_URI = "postgresql://user:password@db/nir"
engine = create_engine(DATABASE_URI)

# Путь для сохранения модели
MLP_PATH = "models/mlp.pkl"
DT_PATH = "models/dt.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_data():
    df = pd.read_sql_table("region", engine)
    return df


def train_models(df):
    years = np.array([2014, 2015, 2016, 2017, 2018, 2019]).reshape(-1, 1)
    vrp_columns = ["vrp_2014", "vrp_2015", "vrp_2016", "vrp_2017", "vrp_2018", "vrp_2019"]

    # Создадим новый столбец для прогноза
    df["vrp_2023"] = np.nan

    # Пройдемся по каждому региону и сделаем прогноз
    for index, row in df.iterrows():
        # Значения ВРП для данного региона
        vrp_values = row[vrp_columns].values.reshape(-1, 1)

        # Обучение модели линейной регрессии
        regression_model = LinearRegression()
        regression_model.fit(years, vrp_values)

        # Прогноз ВРП на 2023 год
        vrp_2023 = regression_model.predict(np.array([[2023]]))

        # Запишем прогнозное значение в DataFrame
        df.at[index, "vrp_2023"] = vrp_2023[0, 0] / df.at[index, "population"]

    x = df.drop(columns=["vrp_2014", "vrp_2015", "vrp_2016", "vrp_2017", "vrp_2018", "vrp_2019", "population",
                         "id", "name", "investment_msu", "result", "reason"], axis=1)

    y = df["investment_msu"].apply(lambda x: 7 if x > 7 else x)

    x = x.rename(str, axis="columns")

    # Разделение данных на обучающие и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=48)
    mlp.fit(x_train_scaled, y_train)
    y_pred_mlp = mlp.predict(x_test_scaled)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"Accuracy for mlp: {accuracy_mlp:.2f}")

    dt = DecisionTreeClassifier(random_state=48)
    dt.fit(x_train_scaled, y_train)
    y_pred_dt = dt.predict(x_test_scaled)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Accuracy for decision tree: {accuracy_dt:.2f}")

    joblib.dump(mlp, MLP_PATH)
    joblib.dump(scaler, SCALER_PATH)

    df["result"] = mlp.predict(scaler.transform(x))
    df = df.drop("vrp_2023", axis=1)
    df.to_sql("region", engine, if_exists="replace", index=False)

    return mlp, scaler


# Проверка, существует ли сохранённая модель
if not os.path.exists(MLP_PATH) or not os.path.exists(SCALER_PATH):
    data = load_data()
    mlp, scaler = train_models(data)
else:
    mlp = joblib.load(MLP_PATH)
    scaler = joblib.load(SCALER_PATH)


@app.route("/")
def index():
    return "hello world"
    return render_template("index.html", data=json.dumps(regions_data))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
