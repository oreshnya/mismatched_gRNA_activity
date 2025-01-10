import pandas as pd
import sqlite3
import sys

def connect_db(db_name: str) -> sqlite3.Connection:
    """
    Подключается к локальному файлу БД (создаёт его, если не существует).
    Возвращает объект соединения sqlite3.
    """
    conn = sqlite3.connect(db_name)
    return conn


def get_db_name():
    """
    Извлекает имя базы данных из аргументов командной строки.
    Если аргумент не указан, возвращает значение по умолчанию.
    """
    default_db = "crispr_sgRNA.db"
    for arg in sys.argv:
        if arg.startswith("--db_name="):
            return arg.split("=")[1]
    return default_db


def load_df_to_db(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str) -> None:
    """
    Загружает DataFrame в таблицу table_name в базе, используя .to_sql().
    Устанавливаем if_exists='replace' для перезаписи таблицы при повторном запуске.
    """
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Данные успешно загружены в таблицу '{table_name}'.")


def table_to_dataframe(db_name: str, table_name: str) -> pd.DataFrame:
    """
    Подключается к базе SQLite и выгружает данные из указанной таблицы.
    Возвращает DataFrame с данными.
    """
    conn = connect_db(db_name)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    close_db(conn)
    return df
    

def close_db(conn: sqlite3.Connection) -> None:
    """
    Закрывает соединение с базой данных.
    """
    conn.close()
    print("Соединение с БД закрыто.")


def create_clean_table(db_name: str) -> None:
    """
    Подключается к БД db_name и создаёт таблицу clean_data
    c необходимыми полями и CHECK-ограничениями.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS clean_data (
        key TEXT PRIMARY KEY,
        perfect_match_sgRNA TEXT NOT NULL,
        gene TEXT,
        sgRNA_sequence TEXT,
        mismatch_position INTEGER NOT NULL CHECK(mismatch_position < 0),
        new_pairing TEXT,
        K562 INTEGER NOT NULL CHECK(K562 IN (0,1)),
        Jurkat INTEGER NOT NULL CHECK(Jurkat IN (0,1)),
        mean_relative_gamma REAL NOT NULL,
        genome_input TEXT,
        sgRNA_input TEXT,
        encoded_or TEXT,
        encoded_stacked TEXT,
        encoded_7channels TEXT,
        gc_content REAL,
        pam TEXT
    );
    """

    conn = connect_db(db_name)
    cur = conn.cursor()
    cur.execute(create_table_sql)
    conn.commit()
    close_db(conn)
    print("Таблица 'clean_data' успешно создана (или уже существует).")


def insert_clean_data(df: pd.DataFrame, db_name: str) -> None:
    """
    Вставляет строки из df в таблицу clean_data.
    Если на какой-то строке возникает IntegrityError (UNIQUE, CHECK, etc.),
    мы просто пропускаем (skip) эту строку и продолжаем дальше.
    """

    conn = connect_db(db_name)
    cur = conn.cursor()

    insert_sql = """
    INSERT INTO clean_data(
        key,
        perfect_match_sgRNA,
        gene,
        sgRNA_sequence,
        mismatch_position,
        new_pairing,
        K562,
        Jurkat,
        mean_relative_gamma,
        genome_input,
        sgRNA_input,
        encoded_or,
        encoded_stacked,
        encoded_7channels,
        gc_content,
        pam      
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    skipped_count = 0
    inserted_count = 0

    for i, row in enumerate(df.itertuples(index=False, name="DataRow"), start=1):
        try:
            cur.execute(insert_sql, (
                row.key,
                row.perfect_match_sgRNA,
                row.gene,
                row.sgRNA_sequence,
                int(row.mismatch_position),
                row.new_pairing,
                int(row.K562),
                int(row.Jurkat),
                float(row.mean_relative_gamma),
                row.genome_input,
                row.sgRNA_input,
                str(row.encoded_or),
                str(row.encoded_stacked),
                str(row.encoded_7channels),
                float(row.gc_content),
                row.pam
            ))
            inserted_count += 1
        except sqlite3.IntegrityError as e:
            skipped_count += 1
            # Логируем, что строчка пропущена
            print(f"[WARNING] Строка #{i} (key={row.key}) пропущена: {e}")
            continue

    conn.commit()
    close_db(conn)
    print(f"[SKIP-INSERT] Успешно вставлено {inserted_count} строк, пропущено {skipped_count} из {len(df)}.")