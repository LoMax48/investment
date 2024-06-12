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
DATABASE_URI = 'postgresql://user:password@db/nir'
engine = create_engine(DATABASE_URI)

# Путь для сохранения модели
MLP_PATH = 'models/mlp.pkl'
DT_PATH = 'models/dt.pkl'
SCALER_PATH = 'models/scaler.pkl'


def load_data():
    df = pd.read_sql_table('region', engine)
    return df


def train_models(df):
    years = np.array([2014, 2015, 2016, 2017, 2018, 2019]).reshape(-1, 1)
    vrp_columns = ['vrp_2014', 'vrp_2015', 'vrp_2016', 'vrp_2017', 'vrp_2018', 'vrp_2019']

    # Создадим новый столбец для прогноза
    df['vrp_2023'] = np.nan

    # Пройдёмся по каждому региону и сделаем прогноз
    for index, row in df.iterrows():
        # Значения ВРП для данного региона
        vrp_values = row[vrp_columns].values.reshape(-1, 1)

        # Обучение модели линейной регрессии
        regression_model = LinearRegression()
        regression_model.fit(years, vrp_values)

        # Прогноз ВРП на 2023 год
        vrp_2023 = regression_model.predict(np.array([[2023]]))

        # Запишем прогнозное значение в DataFrame
        df.at[index, 'vrp_2023'] = vrp_2023[0, 0] / df.at[index, 'population']

    x = df.drop(columns=['vrp_2014', 'vrp_2015', 'vrp_2016', 'vrp_2017', 'vrp_2018', 'vrp_2019', 'population',
                         'id', 'name', 'investment_msu', 'result', 'reason'], axis=1)

    y = df['investment_msu'].apply(lambda x: 7 if x > 7 else x)

    x = x.rename(str, axis='columns')

    # Разделение данных на обучающие и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)

    standard_scaler = StandardScaler()
    x_train_scaled = standard_scaler.fit_transform(x_train)
    x_test_scaled = standard_scaler.transform(x_test)

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=48)
    mlp_classifier.fit(x_train_scaled, y_train)
    y_pred_mlp = mlp_classifier.predict(x_test_scaled)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    df['result'] = mlp_classifier.predict(standard_scaler.transform(x))
    print(f"Accuracy for mlp: {accuracy_mlp:.2f}")

    decision_tree = DecisionTreeClassifier(random_state=48)
    decision_tree.fit(x_train_scaled, y_train)
    y_pred_dt = decision_tree.predict(x_test_scaled)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    df['result_dt'] = decision_tree.predict(standard_scaler.transform(x))
    print(f"Accuracy for decision tree: {accuracy_dt:.2f}")

    joblib.dump(mlp_classifier, MLP_PATH)
    joblib.dump(standard_scaler, SCALER_PATH)
    joblib.dump(decision_tree, DT_PATH)

    # Рассчитываем инвестиционную привлекательность для всех регионов
    # и объяснение результата классификации с помощью дерева решений
    df['reason'] = df.apply(lambda elem: explain_prediction(elem[x.columns], elem['result_dt'], decision_tree,
                                                            standard_scaler, x.columns), axis=1)

    df = df.drop(columns=['vrp_2023', 'result_dt'], axis=1)
    df.to_sql('region', engine, if_exists='replace', index=False)

    return standard_scaler, mlp_classifier, decision_tree


def explain_prediction(features, current_class, decision_tree, standard_scaler, feature_names):
    features_scaled = standard_scaler.transform([features])
    path = decision_tree.decision_path(features_scaled)
    feature = decision_tree.tree_.feature
    threshold = decision_tree.tree_.threshold

    explanation = []

    for node_id in path.indices:
        if feature[node_id] != -2:
            feature_name = feature_names[feature[node_id]]
            threshold_value = threshold[node_id]
            feature_value = features[feature[node_id]]

            if feature_value <= threshold_value:
                if decision_tree.classes_[np.argmax(decision_tree.tree_.value[node_id])] > current_class - 1:
                    explanation.append(f"{feature_name} should be > {threshold_value}")
            else:
                if decision_tree.classes_[np.argmax(decision_tree.tree_.value[node_id])] > current_class - 1:
                    explanation.append(f"{feature_name} should be <= {threshold_value}")

    return ", ".join(explanation)


def format_reason(reason):
    return reason


# Проверка, существует ли сохранённая модель
if not os.path.exists(MLP_PATH) or not os.path.exists(SCALER_PATH):
    data = load_data()
    scaler, mlp, dt = train_models(data)
else:
    scaler = joblib.load(SCALER_PATH)
    mlp = joblib.load(MLP_PATH)
    dt = joblib.load(DT_PATH)


@app.route('/')
@app.route('/home')
def index():
    return 'hello world'
    return render_template('index.html', data=json.dumps(regions_data))


@app.route('/predict', methods=['POST'])
def predict():
    return 'hello world'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
