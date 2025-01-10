import argparse
import pandas as pd

from modules.utils import download_data, txt_to_df
from modules.data_transformation import validate_raw_data, add_new_features
from modules.db_manager import (
    connect_db,
    load_df_to_db,
    table_to_dataframe,
    close_db,
    create_clean_table,
    insert_clean_data
)

def run_pipeline(url: str, local_filename: str, db_name: str):
    # Скачиваем файл
    print(f"Скачивание данных с {url}...")
    download_data(url, local_filename)
    print("Скачивание завершено.\n")

    # Читаем в DataFrame
    print("Чтение данных в DataFrame...")
    df = txt_to_df(local_filename)
    print("Первые 5 строк датасета:")
    print(df.head(), "\n")

    # Подключаемся к БД
    print(f"Подключение к базе данных {db_name}...")
    conn = connect_db(db_name)

    # Загружаем DataFrame в таблицу raw_data
    print("Загрузка данных в таблицу 'raw_data'...")
    load_df_to_db(df, conn, table_name="raw_data")

    # Закрываем соединение
    close_db(conn)
    print("Данные загружены и соединение закрыто.\n")

    # Извлекаем данные из таблицы raw_data
    print("Извлечение данных из таблицы 'raw_data'...")
    df = table_to_dataframe(db_name, table_name="raw_data")

    # Валидируем данные
    print("Валидация данных...")
    df = validate_raw_data(df)

    # Добавляем новые признаки
    print("Добавление новых признаков...")
    df = add_new_features(df)

    # Создаём таблицу clean_data
    print("Создание таблицы 'clean_data'...")
    create_clean_table(db_name)

    # Вставляем очищенные данные
    print("Вставка очищенных данных в 'clean_data'...")
    insert_clean_data(df, db_name)

    # Проверяем данные в clean_data
    print("Проверка данных в 'clean_data'...")
    conn = connect_db(db_name)
    check_df = pd.read_sql("SELECT * FROM clean_data LIMIT 5", conn)
    close_db(conn)

    print("Первые 5 строк из 'clean_data':")
    print(check_df.head())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Запуск пайплайна обработки данных.")
    parser.add_argument(
        "--url",
        type=str,
        default="https://raw.githubusercontent.com/ew314/sgRNA_seq2seq/main/relative_activity_predictor/data/Table_S8_machine_learning_input.txt",
        help="Ссылка на исходный файл данных (по умолчанию: URL из задания)."
    )
    parser.add_argument(
        "--local_filename",
        type=str,
        default="Table_S8_machine_learning_input.txt",
        help="Локальное имя файла для сохранения данных."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="crispr_sgRNA.db",
        help="Имя базы данных SQLite."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args.url, args.local_filename, args.db_name)