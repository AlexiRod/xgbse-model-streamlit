# xgbse model streamlit
## Веб-приложение на streamlit для демонстрации работы XGBoost модели

### Даты разработки: 03.2023-04.2023

### Ключевые технологии: Python, XGBoost, xgbse, streamlit

### Описание

МОДЕЛИРОВАНИЕ ВЕРОЯТНОСТИ ДОЖИТИЯ В КРЕДИТНОМ ПОРТФЕЛЕ С ПОМОЩЬЮ XGBOOST

Программное моделирование вероятности дожития в кредитном портфеле, решающее задачу кредитного скоринга на доступных данных о клиенте (кредитная история, социально-демографические характеристики, информация о кредите и его составляющих и т.д.) С помощью модификации градиентного бустинга XGBoost и пакета xgbse строится модель, которая на основании вероятностей выхода в дефолт (Lifetime probability of default) строит кривые выживаемости для произвольного срока кредита каждого клиента портфеля из открытого датасета Lending club. Результирующая модель помогает разрешить вопрос о NPV сделки, визуализировать и обоснавать потенциальные риски для банковского клиента по его данным.
Для демонстрации работы модели реализовано небольшое веб-приложение с использованием библиотеки streamlit, позволяющее строить кривые выживания/выхода в дефолт в разных пользовательских режимах.

Конечная модель находится в файле estimator.pkl в формате бинарного файла pickle. 

Подготовленный и обработанный датасет находится в файле lending_club_dataset_for_xgbse.csv

Ноутбуки, в которых велась разработка модели находятся в папке collab и разделены на две части.

Файлы из папки pages и файл About.py содержат исходный код приложения с использованием библиотеки streamlit.