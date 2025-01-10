import os
import requests
import pandas as pd

'==================== ЗАГРУЗКА СЫРЫХ ДАННЫХ ИЗ СЕТИ ===================='

def download_data(url: str, local_filename: str) -> None:
    """
    Скачивает файл по ссылке URL и сохраняет под именем local_filename.
    Если файл уже существует, пропускает скачивание.
    """
    if not os.path.exists(local_filename):
        print(f"Скачиваем файл из {url}...")
        response = requests.get(url)
        response.raise_for_status()  # проверяем, что статус 200 ОК
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print("Файл успешно скачан!")
    else:
        print(f"Файл '{local_filename}' уже существует, пропускаем скачивание.")


def txt_to_df(local_filename: str) -> pd.DataFrame:
    """
    Считывает данные из локального txt-файла в DataFrame.
    Предполагается табуляция (\t).
    Переименовывает первый безымянный столбец в 'key'.
    """
    df = pd.read_csv(local_filename, sep="\t")
    # Переименуем безымянный столбец (если он действительно без названия)
    df.rename(columns={df.columns[0]: "key"}, inplace=True)

    df.rename(columns={
        "perfect match sgRNA": "perfect_match_sgRNA",
        "sgRNA sequence": "sgRNA_sequence",
        "mismatch position": "mismatch_position",
        "new pairing": "new_pairing",
        "mean relative gamma": "mean_relative_gamma",
        "genome input": "genome_input",
        "sgRNA input": "sgRNA_input"
    }, inplace=True)
    
    return df