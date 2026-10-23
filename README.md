# MammoAI

Сервис для маммограмм: определяет категорию BI-RADS всей груди (1–5), находит очаги, оценивает их
BI-RADS и P(BI-RADS 4–5) и показывает всё это в веб-просмотрщике.

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

Для уже распакованной локальной копии повторное скачивание и копирование DICOM не нужны:
```sh
mkdir -p data/vindr
ln -s /path/to/vindr-mammo-1.0.0 data/vindr/source
python scripts/vindr_fetch.py status --source data/vindr/source --limit 10 --no-upload
python scripts/vindr_fetch.py download --source data/vindr/source --limit 10 --no-upload
```
Символическая ссылка остаётся локальной и не включается в Git; подготовленные PNG сохраняются в `data/vindr/png`.

**Важно:** в VinDr нет результатов биопсии. `P(BI-RADS 4–5)` — вероятность оценки рентгенолога,
а не вероятность рака по гистологии. Классификатор очагов обучается только на размеченных находках
BI-RADS 3–5; категории 1–2 определяются моделью всей груди и не выдаются классификатором рамок.

Предыдущий вариант на MammoGPS / KAU-BCMD / DMID лежит в `legacy/` и не используется.

## Обучение (Colab в VS Code)

`train/colab_vscode.ipynb` (собирается `python scripts/build_colab_notebook.py`, код проекта встроен):
1. Часть 1 — данные: можно на CPU-сервере. Нужны HF write-токен и логин/пароль PhysioNet (поля ввода, в файл не пишутся).
2. Часть 2 — обучение на GPU (~6 ч на T4): BI-RADS груди → классификатор очагов → детектор → оценка на test.
   Каждая модель сразу загружается в приватный `<логин>/mammoai-weights`.

Затем локально:
```sh
.venv/bin/hf download <логин>/mammoai-weights \
  detector.pt detector.pt.sha256 classifier.pt classifier.pt.sha256 screening.pt screening.pt.sha256 \
  --local-dir models
(cd models && sha256sum -c detector.pt.sha256 classifier.pt.sha256 screening.pt.sha256)
```
Используйте только checkpoints из доверенного приватного репозитория и обязательно проверяйте SHA-256:
формат Ultralytics/PyTorch нельзя считать безопасным для загрузки файлов неизвестного происхождения.

## Запуск сервиса
```sh
python -m venv .venv && .venv/bin/pip install -r requirements-serve.txt
.venv/bin/uvicorn app.main:app --port 8000
```
Для запуска без GPU используйте `MAMMO_DEVICE=cpu .venv/bin/uvicorn app.main:app --port 8000`.
Сервис требует все три проверенных файла `models/detector.pt`, `models/classifier.pt` и
`models/screening.pt`. Для detector обязателен корректный `models/detector.pt.sha256`, который
проверяется до загрузки Ultralytics checkpoint. Размер HTTP body ограничен до multipart-разбора,
размер файла — 100 МБ; дополнительно ограничены декодированные пиксели и число кадров.
DICOM-ввод сервиса предназначен для однокадровых несжатых grayscale mammogram (Implicit/Explicit
VR Little Endian или Explicit VR Big Endian). Inference-запросы выполняются последовательно,
чтобы одновременные обращения не переполняли память GPU.

http://localhost:8000 — перетащить снимок. Колесо — зум, перетаскивание — панорама, `F` — вписать,
`O` — разметка, `I` — инверсия, клик по находке — приблизить, экспорт PNG с разметкой.

```sh
curl -F file=@snimok.dcm "http://localhost:8000/api/analyze?conf=0.05"
```
Ответ: `overall_birads`, `screening` (`birads`, `birads_probs`, `p_suspicious` = P(≥3),
`p_birads_4_5` = P(4–5)), `breast_contour`, `findings` (`bbox_xyxy`, `type`, `birads`,
`p_birads_4_5` = P(BI-RADS 4–5), `level`). Это прокси оценки рентгенолога, не вероятность рака по гистологии.

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
