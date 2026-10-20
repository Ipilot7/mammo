# MammoAI

Сервис для маммограмм: определяет категорию BI-RADS всей груди (1–5), находит очаги, оценивает их
BI-RADS и подозрение на рак и показывает всё это в веб-просмотрщике.

> Исследовательский прототип, **не медицинское изделие**.

## Как устроено

```
снимок (PNG/JPG/DICOM) → окно яркости DICOM → обрезка по груди (mammo/preprocess.py)
   ├─ BI-RADS всей груди: EfficientNet-B3, 512×1024 → P(BI-RADS 1..5)        models/screening.pt
   └─ 1280 px → YOLO11s: рамки очагов + тип                                   models/detector.pt
        → кроп очага → EfficientNet-B3: BI-RADS находки + P(BI-RADS 4–5)       models/classifier.pt
итог = max(BI-RADS груди, BI-RADS худшей находки)
```

На снимке: **пунктиром** обводится вся грудь цветом её категории (так видны и BI-RADS 1–2, где очагов нет),
**рамками** — найденные очаги. Цвета: зелёный — BI-RADS 1–2, жёлтый — 3, красный — 4–5.

## Данные: только VinDr-Mammo

[VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/) (PhysioNet, нужен подписанный DUA):
20 000 снимков, 5 000 исследований, BI-RADS для каждой груди и рамки находок, официальное разбиение train/test.

- `scripts/vindr_fetch.py download` — DICOM с PhysioNet → обрезка по груди → PNG 1536 px → пачки по 1000 снимков
  в **приватный** HF-датасет `<логин>/mammoai-vindr` (~14 ГБ). DICOM (~300 ГБ) нигде не сохраняются.
  В новой сессии готовые пачки берутся из HF, недостающие докачиваются с PhysioNet.
  Скрипт отказывается работать с публичным репозиторием: DUA PhysioNet запрещает передавать данные.
- `mammo/labels.py` → `data/labels.csv`: строка на каждую рамку + строки для снимков без рамок;
  test — официальный, val — 15% исследований из training.

**Важно:** в VinDr нет результатов биопсии. «Подозрение на рак» — это вероятность того, что рентгенолог
поставил бы BI-RADS 4–5, а не вероятность рака по гистологии. Рамки в VinDr в основном у находок BI-RADS 3–5,
поэтому BI-RADS 1–2 определяются по всей груди, а зелёные рамки (BI-RADS 2) менее надёжны.

Предыдущий вариант на MammoGPS / KAU-BCMD / DMID лежит в `legacy/` и не используется.

## Обучение (Colab в VS Code)

`train/colab_vscode.ipynb` (собирается `python scripts/build_colab_notebook.py`, код проекта встроен):
1. Часть 1 — данные: можно на CPU-сервере. Нужны HF write-токен и логин/пароль PhysioNet (поля ввода, в файл не пишутся).
2. Часть 2 — обучение на GPU (~6 ч на T4): BI-RADS груди → классификатор очагов → детектор → оценка на test.
   Каждая модель сразу загружается в приватный `<логин>/mammoai-weights`.

Затем локально:
```sh
.venv/bin/hf download <логин>/mammoai-weights detector.pt classifier.pt screening.pt --local-dir models
```

## Запуск сервиса
```sh
python -m venv .venv && .venv/bin/pip install -r requirements-serve.txt
.venv/bin/uvicorn app.main:app --port 8000
```
http://localhost:8000 — перетащить снимок. Колесо — зум, перетаскивание — панорама, `F` — вписать,
`O` — разметка, `I` — инверсия, клик по находке — приблизить, экспорт PNG с разметкой.

```sh
curl -F file=@snimok.dcm "http://localhost:8000/api/analyze?conf=0.05"
```
Ответ: `overall_birads`, `screening` (`birads`, `birads_probs`, `p_suspicious` = P(≥3), `p_cancer` = P(4–5)),
`breast_contour`, `findings` (`bbox_xyxy`, `type`, `birads`, `malignancy_prob` = P(BI-RADS 4–5), `level`).

## Структура
| Путь | Назначение |
|---|---|
| `scripts/vindr_fetch.py` | загрузка VinDr ↔ приватный HF-датасет |
| `mammo/preprocess.py` | окно DICOM, маска/контур/обрезка груди |
| `mammo/labels.py`, `mammo/prepare_yolo.py` | метки и датасет YOLO (здоровые снимки — фон) |
| `mammo/screening.py`, `train/train_screening.py` | BI-RADS всей груди |
| `mammo/classifier.py`, `train/train_classifier.py` | BI-RADS очага |
| `train/train_detector.py`, `train/evaluate.py` | детектор, оценка |
| `mammo/inference.py`, `app/` | пайплайн, FastAPI, просмотрщик |
| `scripts/screenshots.py` | скриншоты просмотрщика через headless Chrome |
