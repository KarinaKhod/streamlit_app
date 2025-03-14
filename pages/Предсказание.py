import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Настройка scikit-learn для работы с DataFrame (чтобы трансформеры возвращали DataFrame)
sklearn.set_config(transform_output="pandas")

# Настройка страницы: заголовок окна и макет
st.set_page_config(page_title="Прогноз цен на дома - Предсказание",page_icon='🏠', layout="wide")

# Функция для загрузки обученной модели (используем кэширование для ускорения)
def load_model():
    try:
        with open("ml_pipelstreamlit run main.pyine.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

# Заголовок страницы предсказания
st.title("Предсказание цены дома")
st.markdown("Загрузите CSV-файл с данными для предсказания цены.")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Выберите CSV-файл", type="csv")

if uploaded_file is not None:
    try:
        # Чтение CSV-файла в DataFrame
        data = pd.read_csv(uploaded_file)
        st.subheader("Просмотр загруженных данных")
        st.dataframe(data.head())
        
        # Если в данных присутствует столбец 'SalePrice', удаляем его
        if "SalePrice" in data.columns:
            data = data.drop("SalePrice", axis=1)
        
        # Загрузка обученной модели
        model = load_model()
        if model is None:
            st.error("Модель не загружена!")
        else:
            # Получение предсказаний
            predictions_log = model.predict(data)
            # Преобразуем логарифмы обратно в реальные значения
            predictions = np.exp(predictions_log)
            
            st.subheader("Предсказанные цены")
            results = pd.DataFrame({
                "Номер записи": range(1, len(predictions) + 1),
                "Предсказанная цена": predictions
            })
            st.dataframe(results)
            st.success("Предсказание успешно выполнено!")
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
else:
    st.info("Пожалуйста, загрузите CSV-файл для продолжения.")