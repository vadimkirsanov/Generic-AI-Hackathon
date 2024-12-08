import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import pydicom
import tensorflow as tf
from PIL import Image

# Загрузка обученной модели 
model_path = '/Users/vadimkirsanov/Desktop/MIPT_DS/Python_coding_data/chest_xray_hac/my_model_4dense.keras'  # Укажите правильный путь
model = tf.keras.models.load_model(model_path)

IMG_SIZE = 224  # Размер изображения, используемый при обучении модели

def preprocess_image(file, img_size):
    """
    Предобработка изображения: чтение, изменение размера, нормализация.
    :param file: входной файл изображения
    :param img_size: конечный размер изображения
    :return: предобработанное изображение
    """
    # Определение типа файла и его чтение 
    if file.type == 'application/dicom':
        dicom = pydicom.dcmread(file)
        img = dicom.pixel_array
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Используем PIL для открытия любых изображений 🖼
        image = Image.open(file)
        img = np.array(image.convert('RGB'))

    # Изменение размера и нормализация 
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(file):
    """
    Прогнозирует класс изображения.
    :param file: входной файл изображения
    :return: вероятности классов
    """
    preprocessed_image = preprocess_image(file, IMG_SIZE)
    prediction = model.predict(preprocessed_image)
    return prediction[0]

def main():
    st.title("Image Prediction")

    # Загрузка файлов изображений 
    uploaded_files = st.file_uploader(
        "Upload image files (DICOM, JPG, PNG)",
        accept_multiple_files=True
    )

    # Обработка и отображение результатов
    if uploaded_files:
        results = []
        for idx, uploaded_file in enumerate(uploaded_files):
            prediction = predict_image(uploaded_file)
            foreign_bodies_prob = int(round(prediction[0] * 100))
            clavicle_fracture_prob = int(round(prediction[1] * 100))

            results.append({
                'study_instance_anon': uploaded_file.name,
                'result_fracture': clavicle_fracture_prob,
                'result_medimp': foreign_bodies_prob
            })

        results_df = pd.DataFrame(results)

        # Отображение DataFrame в Streamlit 
        st.dataframe(results_df)

        # Создание Excel файла в памяти 
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False)
        excel_bytes.seek(0)  # Перемещаем курсор в начало файла после записи

        # Кнопка для скачивания Excel 
        st.download_button(
            label="Download Results as Excel",
            data=excel_bytes,
            file_name='predictions.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
