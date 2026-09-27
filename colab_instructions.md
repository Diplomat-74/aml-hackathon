# Запуск реального пайплайна в Google Colab

## Ячейка 1 — клонировать репозиторий
!git clone https://github.com/Diplomat-74/aml-hackathon.git
%cd aml-hackathon

## Ячейка 2 — установить зависимости
!pip install -q -r requirements.txt
# если requirements.txt лежит не в корне, а например в website/ или configs/, укажи правильный путь

## Ячейка 3 — положить реальные данные конкурса
# Если файлы (train/test signals csv, train/test transactions parquet, sample_submission)
# ещё не лежат в data/raw/ в самом репозитории — загрузи их вручную в Colab и скопируй:
from google.colab import files
uploaded = files.upload()  # выбери train_signals.csv, test_signals.csv,
                            # train_transactions.parquet, test_transactions.parquet, sample_submission.csv
!mkdir -p data/raw
!mv train_signals.csv test_signals.csv train_transactions.parquet test_transactions.parquet sample_submission.csv data/raw/

## Ячейка 4 — прогнать пайплайн на настоящих данных (НЕ --synthetic)
!python run_pipeline.py --profile fast   # быстрая проверка, ~несколько минут
# если всё ок:
!python run_pipeline.py                  # полный прогон на data/raw
!pytest -q                               # тесты на утечки/схему

## Ячейка 5 — проверить, что появился submission.csv и артефакты сайта
!ls outputs/
!ls reports/artifacts/

## Ячейка 6 — сгенерировать team_264FC1A2.csv по правилам конкурса
# (в репозитории уже есть логика в SUBMISSION.ipynb — либо открой и прогони его
# целиком в Colab вместо ручных шагов 1-5, он делает всё то же самое + валидацию)

## Ячейка 7 — закоммитить реальные артефакты обратно в GitHub
# чтобы задеплоенный Streamlit-сайт подхватил реальные данные
!git config --global user.email "you@example.com"
!git config --global user.name "Diplomat-74"
!git add reports/artifacts/ outputs/ team_264FC1A2.csv
!git commit -m "Real pipeline run on actual contest data"
# понадобится Personal Access Token вместо пароля при пуше:
!git remote set-url origin https://<TOKEN>@github.com/Diplomat-74/aml-hackathon.git
!git push origin main
