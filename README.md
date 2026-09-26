# AML Signal Escalation — production-grade пайплайн

> **Статус сдачи (Team 264FC1A2):** схема данных адаптирована под реальный датасет конкурса
> (см. `configs/pipeline.yaml`), реальные файлы лежат в `data/raw/`. Числа CV AUC 0.856±0.007 /
> hidden-test AUC 0.814, упомянутые где-либо ранее, — **из прогона на синтетических данных**, а не
> на реальном датасете конкурса. Настоящие метрики и `submission.csv` появятся только после
> реального запуска `python run_pipeline.py` (см. `SUBMISSION.ipynb`) в окружении с интернетом
> (локально или в Google Colab) — сам код это не заявляет, пока не увидит результат своими глазами.
> Сайт для сдачи — `website/app.py` (полноценный multipage EDA + model insights, читает реальные
> артефакты прогона), а не отдельный `streamlit_website.py`.


Предсказание эскалации AML-сигнала (`target`) по транзакционной истории клиента **строго до** `signal_sanasi`.
Конфиг-драйвен, воспроизводим одной командой, с формальными тестами на отсутствие утечки.

```bash
conda env create -f environment.yml && conda activate aml-escalation   # или pip install -r requirements.txt
python run_pipeline.py --synthetic --profile fast   # smoke end-to-end на синтетике (~3 мин)
python run_pipeline.py                              # полный прогон на data/raw
pytest -q                                           # leakage / schema / feature тесты
streamlit run website/app.py                        # EDA- и model-insights-сайт
```

## 1. Подключение реальных данных
1. Положите файлы в `data/raw/` (csv / parquet / feather / xlsx).
2. Отредактируйте **только** `configs/pipeline.yaml`: блок `data` (имена файлов) и `schema` (имена колонок).
   Внутри весь код работает с каноническими именами, поэтому больше ничего менять не нужно.
   * транзакции привязаны к сигналу напрямую → `schema.transactions.key: signal_id`, `schema.signals.key: signal_id`;
   * знак суммы кодирует направление → `signed_amount: true`;
   * нет client-id → `schema.signals.group: null` (скрытые группы найдутся по общим транзакциям).
3. `python run_pipeline.py`. Стадия `data` сначала проверит схему и остановится с понятной ошибкой, если что-то не так.

## 2. Структура
```
configs/            pipeline.yaml (пути, схема, CV, ансамбль), features.yaml, model_*.yaml, profiles/fast.yaml
src/
  config.py, logging_setup.py, io_utils.py, _mini_yaml.py (fallback, если нет PyYAML)
  data_loading.py   raw -> канонические имена, data/interim/*.parquet
  data_validation.py  schema/sanity-проверки (+ pandera, если установлен)
  features/         base_aggregates, temporal, decay_weighted, change_detection (CUSUM), domain_aml,
                    cold_start_handler, build_features (оркестрация, кодирование, pruning)
  validation/       purged_time_split (purge+embargo), group_aware_split (псевдо-группы), adversarial_validation
  models/           baseline_logreg, lightgbm, catboost, xgboost, sklearn_hgb, training (CV-движок),
                    stacking (rank-mean / веса / OOF-стекинг), cold_start_model, tune_optuna
  evaluation/       metrics, stability_report, importance (OOF permutation), feature_ablation, shap_report,
                    calibration, plots, reporting (model card + experiment log)
  inference/predict.py   bundle -> outputs/submission.csv (+ sanity checks)
  pipeline.py       стадии: data → features → eda → adversarial → tune → train → importance → ablation
                    → ensemble → evaluate → predict → report
tests/              test_no_leakage.py, test_schema_consistency.py, test_feature_pipeline.py
notebooks/          01–05: тонкие обёртки над src/ (генерируются scripts/build_notebooks.py)
website/            Streamlit: обзор → EDA → поведение перед эскалацией → model insights → выводы
reports/            model_card.md (автоген), experiment_log.csv, artifacts/ (всё, что читает сайт), figures/
```

## 3. Ключевые решения и почему

### Anti-leakage
* Единственная точка соединения сигналов с транзакциями — `features/common.prepare_event_frame`,
  фильтр `ts < signal_date − cutoff_offset`. Там же `assert`.
* Никаких статистик «по всей выборке» внутри фич (нет transductive leak между train и test).
* Тесты **инвариантности**: добавление транзакций в момент/после сигнала не меняет ни одной фичи; добавление
  прошлых — меняет (тест не вакуумный); фичи сигнала не зависят от того, какие ещё сигналы в выборке.
* Категории кодируются по train; pruning (константы / дубли по Spearman) — только по train.

### Фичи (7 групп, префикс = группа для ablation)
| группа | что | зачем |
|---|---|---|
| `sig__` | атрибуты сигнала, календарь | baseline без истории |
| `base__` | count/sum/mean/std/max/median, in/out, контрагенты, активные дни по окнам 1–180д + velocity | обязательный минимум |
| `tmp__` | recency, межтранзакционные интервалы, burstiness, ночь/выходные, пиковый день | структура во времени |
| `dec__` | экспоненциально взвешенные по давности суммы/счётчики (half-life 1–90д) + «ускорение» | разгон без обрыва на границе окна |
| `chg__` | CUSUM относительно **собственной** базовой линии клиента (с учётом даты открытия счёта), streak тревоги, «спящий счёт» | стандарт AML-мониторинга для внезапного дрейфа |
| `aml__` | structuring, круглые суммы, pass-through, быстрый/зеркальный вывод, fan-in/out, новые контрагенты, HHI, юрисдикции риска, наличные, Бенфорд | типологии FATF в числах |
| `cold__` | объём и длина истории, `is_cold_start`, `is_no_history` | отдельная стратегия для тонкой истории |

Семантическая импутация: счётчики/суммы при отсутствии транзакций = 0, статистики = NaN (GBDT ест нативно;
для линейных моделей — `SegmentAwareImputer`: медиана **warm**-сегмента + missing-индикатор).

### Валидация
* **Purged & embargoed walk-forward CV** (`cv.mode: walk_forward`) — train всегда раньше valid, embargo между ними,
  purge по `label_horizon_days`. Режим `purged_kfold` дополнительно пуржит train-строки **после** valid, чьё окно
  фич (`purge_window_days`) видит valid-период (иначе утекают последствия эскалации, например заморозка счёта).
  `purge_scope: group` пуржит только того же клиента — меньше потерь данных.
* **Group-aware: auto** — CV имитирует реальное отношение train/test: если клиенты теста уже есть в train
  (типично для временного сплита), жёсткий group-split сделал бы CV пессимистичной; если тест — новые клиенты,
  без него CV оптимистична. Решение и доля пересечения логируются.
* Early stopping — на **хвосте train-фолда** (с embargo), не на valid-фолде: иначе OOF смещён.
* **Adversarial validation** до обучения: AUC train-vs-test, KS по фичам, итеративный drop сдвигающих фич,
  опционально importance weights `p/(1−p)`.
* Метрика — **mean ± std AUC по фолдам**; выбор моделей и ансамблей по `mean − λ·std`.

### Модели и ансамбль
* LogReg (baseline, точный linear SHAP), LightGBM, CatBoost, XGBoost, sklearn HGB (всегда доступна, fallback).
  Недоступная библиотека пропускается с warning (`models.strict: true` — упасть).
* Финальный refit на всём train: `median(best_iter) × n_full/n_fold`, усреднение по нескольким сидам.
* Кандидаты: одиночные модели, rank-mean, **веса, оптимизированные под AUC** (Dirichlet search + Nelder–Mead),
  **OOF-стекинг** (логрег / неглубокий GBDT на logit-OOF). Оценка **nested по времени**: комбайнер для фолда k
  учится только на более ранних фолдах. Если ансамбль не лучше лучшей одиночной на `min_gain` — берём одиночную.
* Cold-start: отдельная регуляризованная модель на `sig__/cold__`, смешивание в rank-пространстве
  с alpha, подобранным на мета-фолдах; включается, только если даёт прирост.
* HPO: `--set tuning.enabled=true` — Optuna (TPE multivariate + MedianPruner по фолдам) поверх **той же**
  purged-схемы, objective = mean − penalty·std. Без optuna — random search по тому же пространству.

### Интерпретация и отчётность
* Permutation importance на **OOF** (модель фолда × его valid) + стабильность: доля фолдов с положительным
  вкладом, std ранга, Spearman/Jaccard между фолдами, список «топ в одном фолде, шум в другом».
  `feature_selection.stability_filter: true` — выкинуть нестабильные и переобучить.
* Ablation: инкрементально по группам + leave-one-group-out, усреднение по сидам.
* SHAP: нативный TreeSHAP LightGBM/XGBoost/CatBoost (пакет shap не обязателен), fallback — linear SHAP логрега.
* Калибровка (reliability + ECE), precision/recall@top-k (сколько эскалаций поймаем при ёмкости аналитиков k%).
* `reports/model_card.md` генерируется автоматически из артефактов; `reports/experiment_log.csv` — каждый прогон
  (run_id, commit, config_hash, метрики); MLflow — `--set tracking.mlflow=true`.

## 4. Частые команды
```bash
python run_pipeline.py --stages ensemble,evaluate,predict,report      # перезапуск части стадий
python run_pipeline.py --set cv.mode=purged_kfold --set cv.purge_scope=group
python run_pipeline.py --set "models.enabled=[lgbm,catboost]" --set ensemble.risk_aversion=1.0
python run_pipeline.py --set adversarial.drop_features_automatically=true
python -m src.inference.predict                                      # только submission из сохранённого bundle
```

## 5. Синтетика и аудит честности CV
`--synthetic` генерирует данные с заложенными типологиями (burst, structuring, pass-through, high-risk, новые
контрагенты, спящий счёт), cold-start клиентами, временным сдвигом и заморозкой счёта после эскалации.
У синтетики есть скрытые метки теста (`data/raw/_test_labels_hidden.csv`), поэтому пайплайн сравнивает CV AUC
с AUC на скрытом тесте — прямая проверка, что схема валидации не завышает качество. На реальных данных этого
файла нет, шаг пропускается.
