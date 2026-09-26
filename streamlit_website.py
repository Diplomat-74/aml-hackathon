"""
AML Signal Escalation — Team 264FC1A2
EDA / результаты сайта. Запуск: streamlit run streamlit_website.py

Честная версия: никаких захардкоженных метрик. Каждое число либо посчитано
из реально загруженных файлов, либо страница явно говорит, что данных нет.

Ожидаемые файлы рядом со скриптом (можно докладывать по мере готовности):
  train_signals.csv   — signal_id, signal_sanasi, eskalatsiya
  test_signals.csv    — signal_id, signal_sanasi
  team_264FC1A2.csv   — signal_id, ehtimollik (реальный submission, когда он появится)
  metrics.json        — необязательно: {"cv_auc_mean": .., "cv_auc_std": .., "holdout_auc": ..}
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AML Signal Escalation — 264FC1A2", layout="wide")
st.title("🛡️ AML Signal Escalation — Team 264FC1A2")

HERE = Path(__file__).resolve().parent


def load_csv(name: str):
    p = HERE / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Не смог прочитать {name}: {e}")
        return None


train = load_csv("train_signals.csv")
test = load_csv("test_signals.csv")
sub = load_csv("team_264FC1A2.csv")

metrics_path = HERE / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# --- Метрики модели -------------------------------------------------------
st.header("Метрики модели")
if metrics:
    c1, c2 = st.columns(2)
    if "cv_auc_mean" in metrics:
        c1.metric(
            "CV AUC",
            f"{metrics['cv_auc_mean']:.3f} ± {metrics.get('cv_auc_std', 0):.3f}",
        )
    if "holdout_auc" in metrics:
        c2.metric("Holdout / hidden-test AUC", f"{metrics['holdout_auc']:.3f}")
else:
    st.info(
        "metrics.json не найден — метрики появятся здесь после реального прогона "
        "пайплайна (см. SUBMISSION.ipynb). Ничего не выдумываем заранее."
    )

# --- EDA по сигналам -------------------------------------------------------
st.header("Данные")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Train signals")
    if train is not None:
        st.metric("Всего сигналов", len(train))
        if "eskalatsiya" in train.columns:
            rate = train["eskalatsiya"].mean()
            st.metric("Доля эскалаций", f"{rate:.2%}")
            st.bar_chart(train["eskalatsiya"].value_counts().sort_index())
        if "signal_sanasi" in train.columns:
            st.caption(
                f"Период: {train['signal_sanasi'].min()} — {train['signal_sanasi'].max()}"
            )
    else:
        st.info("train_signals.csv не загружен в репозиторий.")

with col2:
    st.subheader("Test signals")
    if test is not None:
        st.metric("Всего сигналов", len(test))
        if "signal_sanasi" in test.columns:
            st.caption(
                f"Период: {test['signal_sanasi'].min()} — {test['signal_sanasi'].max()}"
            )
    else:
        st.info("test_signals.csv не загружен в репозиторий.")

# --- Submission -------------------------------------------------------------
st.header("Submission")
if sub is not None:
    st.metric("Строк в submission", len(sub))
    expected_cols = ["signal_id", "ehtimollik"]
    if list(sub.columns) == expected_cols:
        st.success("Колонки соответствуют требуемому формату (signal_id, ehtimollik).")
    else:
        st.error(f"Ожидались колонки {expected_cols}, а в файле {list(sub.columns)}.")

    if test is not None:
        test_ids = set(test["signal_id"].astype(str))
        sub_ids = set(sub["signal_id"].astype(str))
        checks = {
            "ровно одна строка на каждый test signal_id": len(sub) == len(test_ids),
            "нет дублей id": sub["signal_id"].astype(str).is_unique,
            "нет посторонних id": sub_ids <= test_ids,
            "нет пропущенных id": test_ids <= sub_ids,
            "0 <= ehtimollik <= 1": sub["ehtimollik"].between(0, 1).all(),
        }
        for name, ok in checks.items():
            st.write(("✅" if ok else "❌") + " " + name)

    st.subheader("Распределение предсказаний")
    st.bar_chart(pd.cut(sub["ehtimollik"], bins=10).value_counts().sort_index())
else:
    st.info(
        "team_264FC1A2.csv ещё не загружен — здесь появится распределение "
        "предсказаний и проверка по официальным правилам сдачи, как только файл будет готов."
    )
