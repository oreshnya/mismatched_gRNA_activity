import streamlit as st
import pandas as pd
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from modules.db_manager import get_db_name, table_to_dataframe

def main():

    st.title("CRISPR sgRNA Dashboard")

    # Получаем имя базы данных из аргументов командной строки
    db_name = get_db_name()

    # Проверка существования файла базы данных
    if not os.path.exists(db_name):
        st.error(f"Файл базы данных '{db_name}' не найден.")
        st.stop()

    # 1) Загрузка данных
    st.subheader("Загрузка данных из базы")
    df = table_to_dataframe(db_name, table_name='clean_data')
    st.write(f"Всего строк: {len(df)}")

    # 2) Фильтр по pam
    st.subheader("Фильтр по PAM")
    unique_pams = df["pam"].dropna().unique().tolist()
    unique_pams.sort()  # чтобы список шел по алфавиту
    pam_filter = st.multiselect("Выберите значение PAM", options=unique_pams, default=unique_pams)
    
    # Применяем фильтр
    if pam_filter:
        df_filtered = df[df["pam"].isin(pam_filter)]
    else:
        df_filtered = df.copy()
    
    st.write(f"Отфильтровано строк: {len(df_filtered)}")
    
    # 3) График распределения mean_relative_gamma (с цветовой группировкой по pam)
    st.subheader("Распределение mean_relative_gamma")
    fig_hist = px.histogram(
        df_filtered, 
        x="mean_relative_gamma", 
        color="pam", 
        hover_data=df_filtered.columns,
        width=1000, 
        height=450,
        nbins=50
    )
    fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_hist)
    
    # 4) Четыре гистограммы на одной Figure (pam, mismatch_position, gc_content, new_pairing)
    st.subheader("Распределение ключевых признаков")

    # Список столбцов, которые хотим визуализировать
    columns_to_plot = ["pam", "mismatch_position", "gc_content", "new_pairing"]

    # Создаём сетку 2x2
    fig_sub = make_subplots(
        rows=2, cols=2,
        subplot_titles=("PAM", "Mismatch Position", "GC Content", "New Pairing"),
        vertical_spacing=0.2
    )

    # Индексы для subplot
    subplot_positions = [(1,1), (1,2), (2,1), (2,2)]

    for (col_name, (row, col)) in zip(columns_to_plot, subplot_positions):
        hist_fig = px.histogram(df_filtered, x=col_name)
        # Настраиваем отображение
        hist_fig.update_traces(marker_line_width=0.5, marker_line_color="black")
        
        # Добавляем трейсы в нашу общую Figure
        for trace in hist_fig.data:
            fig_sub.add_trace(trace, row=row, col=col)

    fig_sub.update_layout(
        height=700,
        width=1000,
        title_text="Распределение по различным признакам",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_sub)
    
    # 5) Дополнительная статистика: описательные данные
    st.subheader("Основные статистики по числовым признакам")
    numeric_cols = ["mismatch_position", "mean_relative_gamma", "gc_content"]
    st.write(df_filtered[numeric_cols].describe())

    # 6) (Опционально) Корреляционная матрица
    if st.checkbox("Показать корреляционную матрицу (числовые колонки)"):
        corr = df_filtered[numeric_cols].corr()
        st.write(corr)
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(corr_fig)

    st.info("Это учебный дашборд для визуального анализа данных о sgRNA в CRISPR.")

if __name__ == "__main__":
    main()