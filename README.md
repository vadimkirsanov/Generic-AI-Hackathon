# Generic-AI-Hackathon
A machine learning model capable of automatically detecting clavicle fractures and foreign bodies in chest X-rays.


# Инструкция по использованию проекта

## Описание

Данный проект позволяет обрабатывать DICOM-файлы и сопоставлять информацию из них с данными, содержащимися в Excel-документах. Основные шаги включают разархивацию данных, настройку путей, обработку и структурирование данных с помощью Pandas DataFrame.

## Шаги для запуска и использования модели

### Шаг 1: Установка необходимых зависимостей

Для работы проекта необходимы следующие зависимости:
- Python 3.x
- Модуль для работы с файловой системой: `os` (устанавливается вместе с Python)
- Pandas для работы с таблицами (DataFrame)

Установка Pandas:

```bash
pip install pandas
```

### Шаг 2: Подготовка данных

1. **Разархивирование DICOM-файлов**: 
   - Убедитесь, что все архивы с DICOM-файлами разархивированы.

2. **Настройка путей**:
   - Укажите путь до папки `block_0000_anon` в переменной `medimp_directory`. 
   ```python
   medimp_directory = 'путь/до/папки/medimp_block_0000_anon'
   ```
   - Укажите путь до папки `block_0000_anon` в переменной `clav_fracture_directory`. 
   ```python
   clav_fracture_directory = 'путь/до/папки/clav_fracture_block_0000_anon'
   ```
   - Укажите путь до xls документов в `medimp_excel_path` и `clav_fracture_excel_path`.
   ```python
   medimp_excel_path = 'путь/до/medimp_excel_file.xlsx'
   clav_fracture_excel_path = 'путь/до/clav_fracture_excel_file.xlsx'
   ```

### Шаг 3: Запуск обработки

Запустите код для выполнения следующих этапов:

- **Обработка DICOM-файлов**: Используйте функцию `dataframe_dcm` для сбора информации о файлах в указанных папках. Она создаёт `DataFrame` с данными о путях к DICOM-файлам.
   ```python
   df_medimp = dataframe_dcm(medimp_directory).sort_values(by='path_name')
   df_clav_fracture = dataframe_dcm(clav_fracture_directory).sort_values(by='path_name')
   ```

- **Обработка данных Excel**: Загружаем данные из Excel и сортируем по столбцу `study_instance_anon`.
   ```python
   medimp_data = pd.read_excel(medimp_excel_path).sort_values(by='study_instance_anon')
   clav_fracture_data = pd.read_excel(clav_fracture_excel_path).sort_values(by='study_instance_anon')
   ```

- **Слияние данных**: Объединяем DataFrame из DICOM-файлов с данными из Excel для получения структурированной информации.
   ```python
   df_medimp = df_medimp.merge(medimp_data[['study_instance_anon', 'pathology']], how='left', left_on='path_name', right_on='study_instance_anon')
   df_clav_fracture = df_clav_fracture.merge(clav_fracture_data[['study_instance_anon', 'pathology']], how='left', left_on='path_name', right_on='study_instance_anon')
   ```

- **Удаление временных столбцов**: Удаляется временный столбец после слияния, чтобы очистить данные.
   ```python
   df_medimp.drop(columns=['study_instance_anon', 'index'], inplace=True)
   df_clav_fracture.drop(columns=['study_instance_anon', 'index'], inplace=True)
   ```

- **Проверка данных**: Выводим первые пять строк для проверки.
   ```python
   print(df_medimp.head())
   print(df_clav_fracture.head())
   ```
