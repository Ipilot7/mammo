"""Build train/colab_vscode.ipynb: a self-contained notebook for the Colab VS Code extension.

The project code is embedded as a base64 zip (files.upload/download are unavailable in VS Code),
and trained weights are returned through a private Hugging Face model repo.
Re-run this script after changing the code.
"""
# ruff: noqa: ISC004
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["mammo", "train", "scripts", "requirements.txt"]

# Helper injected into the notebook: streams command output, shows progress bars
# (which redraw with \r) as live-updating lines instead of dropping them.
RUN_HELPER = r"""import hashlib, re, time
from pathlib import Path
from IPython.display import display, HTML

ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
BAR_KEY = re.compile(r'^\s*([A-Za-z][A-Za-z ()._-]*?)\s*[:(]')

def run(cmd):
    # Run a shell command, stream its output with live progress bars, stop the notebook if it fails.
    print('$', cmd, flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    bars, handle, last = {}, None, 0.0

    def show(force=False):
        nonlocal handle, last
        if not bars or (not force and time.monotonic() - last < 1.0):
            return
        last = time.monotonic()
        text = '\n'.join(list(bars.values())[-4:])
        html = HTML('<pre style="margin:0">' + text.replace('&', '&amp;').replace('<', '&lt;') + '</pre>')
        if handle is None:
            handle = display(html, display_id=True)
        else:
            handle.update(html)

    buf = ''
    fd = p.stdout.fileno()
    while chunk := os.read(fd, 4096):
        buf += ANSI.sub('', chunk.decode(errors='replace'))
        while (i := min((buf.find(c) for c in '\r\n' if c in buf), default=-1)) >= 0:
            piece, sep, buf = buf[:i], buf[i], buf[i + 1:]
            if not piece.strip():
                continue
            if sep == '\n':
                print(piece, flush=True)
            else:  # progress-bar redraw: keep the latest state per bar
                m = BAR_KEY.match(piece)
                bars[m.group(1) if m else 'progress'] = piece.strip()
                show()
    show(force=True)
    if p.wait():
        raise RuntimeError(f'команда завершилась с ошибкой ({p.returncode}): {cmd}')

def upload(name):
    # Save trained weights to the private HF repo right away, so they survive a server disconnect.
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    repo = f"{api.whoami()['name']}/{HF_REPO}"
    api.create_repo(repo, private=True, exist_ok=True)
    model_path = Path('models') / name
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    checksum_path = Path(f'{model_path}.sha256')
    checksum_path.write_text(f'{digest}  {name}\n')
    api.upload_file(path_or_fileobj=model_path, path_in_repo=name, repo_id=repo)
    api.upload_file(path_or_fileobj=checksum_path, path_in_repo=f'{name}.sha256', repo_id=repo)
    print(f'{name} → huggingface.co/{repo}')"""


def pack() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(p.rglob("*"))
            for f in files:
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".ipynb":
                    z.write(f, f.relative_to(ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    code_b64 = pack()
    cells = [
        ("markdown", "# MammoAI — обучение на VinDr-Mammo через Colab в VS Code\n"
                     "**Часть 1 — данные** (можно на обычном CPU-сервере, GPU-квота не тратится): 20 000 DICOM с PhysioNet "
                     "→ обрезка по груди → PNG 1536 px → пачки по 1000 в ваш **приватный** HF-датасет `mammoai-vindr` (~14 ГБ). "
                     "Исходные DICOM (~300 ГБ) сохраняются только на ваш Google Drive, если `USE_DRIVE = True`. При обрыве просто запустите снова — продолжит с места остановки.\n\n"
                     "**Часть 2 — обучение** (нужен GPU): данные берутся из HF за минуты, PhysioNet уже не нужен. ~6 ч на T4, "
                     "каждая модель сразу сохраняется в HF-репозиторий `mammoai-weights`.\n\n"
                     "**Run All** — выполнит обе части (часть 2 остановится с понятной ошибкой, если сервер без GPU)."),
        ("code", "import os, subprocess, torch\n"
                 "from getpass import getpass\n"
                 "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет (подходит только для части 1)')\n"
                 "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass('Hugging Face write-токен: ')\n"
                 "os.environ['HF_TOKEN'] = HF_TOKEN\n"
                 "if not os.environ.get('PHYSIONET_USER'):\n"
                 "    # Only needed while some VinDr shards are not in your HF dataset yet.\n"
                 "    os.environ['PHYSIONET_USER'] = input('PhysioNet логин (Enter — если данные уже в HF): ').strip()\n"
                 "    if os.environ['PHYSIONET_USER']:\n"
                 "        os.environ['PHYSIONET_PASS'] = getpass('PhysioNet пароль: ')\n"
                 "HF_REPO = 'mammoai-weights'  # приватный репозиторий весов <ваш_логин>/mammoai-weights\n"
                 "SCREENING_EPOCHS = 12\nCLASSIFIER_EPOCHS = 30\nDETECTOR_EPOCHS = 40\n"
                 "RETRAIN = False  # True — обучать заново, даже если модель уже есть на этом сервере\n"
                 "USE_DRIVE = True  # True — сначала сырые DICOM на Google Drive через wget (1 соединение), потом конвертация оттуда\n"
                 "DRIVE_DIR = '/content/drive/MyDrive/vindr-mammo'  # папка с сырыми данными VinDr на Drive\n"
                 "\n"
                 + RUN_HELPER),
        ("code", "# Распаковка кода проекта (встроен в ноутбук)\n"
                 "import base64, io, zipfile\n"
                 f"CODE = \"{code_b64}\"\n"
                 "os.makedirs('/content/mammo', exist_ok=True)\n"
                 "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall('/content/mammo')\n"
                 "os.chdir('/content/mammo')\nprint(sorted(os.listdir('.')))"),
        ("code", "run('pip install -q timm ultralytics openpyxl pydicom pylibjpeg pylibjpeg-openjpeg')  # torch с CUDA уже есть"),
        ("markdown", "## Часть 1. Данные VinDr-Mammo\n"
                     "Печатает скорость (MB/s) и оставшееся время. Готовые пачки берутся из HF, недостающие делаются "
                     "из DICOM — с Google Drive (`USE_DRIVE = True`) или напрямую с PhysioNet."),
        ("markdown", "### Сырые DICOM на Google Drive (если `USE_DRIVE = True`)\n"
                     "Много параллельных соединений PhysioNet режет (таймауты подключения с IP Colab), поэтому датасет "
                     "качается `wget` в **одно** соединение прямо на Google Drive, а конвертация потом идёт с Drive.\n\n"
                     "- Полный датасет — **≈ 300+ ГБ DICOM**: на Google Drive нужно столько свободного места "
                     "(платный тариф Google One).\n"
                     "- Одной сессии Colab может не хватить — просто запустите ячейку снова, `wget` докачает с места остановки.\n"
                     "- В конце печатается, сколько `.dicom` уже на Drive из ожидаемых 20 000, и общий размер.\n\n"
                     "Сначала быстрая проверка `vindr_fetch.py status`: если все пачки уже в HF (`mammoai-vindr`), "
                     "`wget` не запускается. Если данные уже в HF — Drive вообще не нужен, можно поставить `USE_DRIVE = False`."),
        ("code", "if USE_DRIVE:\n"
                 "    try:\n"
                 "        from google.colab import drive\n"
                 "        drive.mount('/content/drive')\n"
                 "    except Exception as e:\n"
                 "        raise RuntimeError('Google Drive недоступен в этом окружении — поставьте USE_DRIVE = False '\n"
                 "                           'в первой ячейке') from e\n"
                 "    os.makedirs(DRIVE_DIR, exist_ok=True)\n"
                 "    # Cheap check: are any shards missing both here and in the HF dataset? (exit 0: none, 10: some)\n"
                 "    st = subprocess.run(['python', 'scripts/vindr_fetch.py', 'status', '--source', DRIVE_DIR],\n"
                 "                        capture_output=True, text=True)\n"
                 "    print(st.stdout + st.stderr)\n"
                 "    if st.returncode == 0:\n"
                 "        print('все пачки уже есть в HF — скачивание DICOM на Drive не нужно')\n"
                 "    elif st.returncode != 10:\n"
                 "        raise RuntimeError(f'не удалось проверить, какие пачки уже есть в HF (код {st.returncode}): '\n"
                 "                           'проверьте HF_TOKEN и сеть')\n"
                 "    elif os.environ.get('PHYSIONET_USER'):\n"
                 "        run(f'python scripts/vindr_to_drive.py --dest \"{DRIVE_DIR}\"')\n"
                 "    elif not os.path.exists(os.path.join(DRIVE_DIR, 'breast-level_annotations.csv')):\n"
                 "        print('ВНИМАНИЕ: части пачек нет в HF, а логин PhysioNet не задан и на Drive нет '\n"
                 "              'breast-level_annotations.csv — следующая ячейка не сможет сделать недостающие пачки. '\n"
                 "              'Перезапустите первую ячейку и введите логин PhysioNet.')\n"
                 "    else:\n"
                 "        print('логин PhysioNet не задан — wget пропущен, конвертация из того, что уже на Drive')\n"
                 "else:\n"
                 "    print('USE_DRIVE = False — DICOM будут качаться с PhysioNet напрямую в следующей ячейке')"),
        ("code", "if USE_DRIVE:\n"
                 "    run(f'python scripts/vindr_fetch.py download --source \"{DRIVE_DIR}\" --workers 8')\n"
                 "else:\n"
                 "    run('python scripts/vindr_fetch.py download --workers 4')  # больше соединений — PhysioNet режет IP\n"
                 "run('python -m mammo.labels')"),
        ("markdown", "## Часть 2. Обучение (GPU)\n### BI-RADS всей груди 1–5 (~1.5 ч)"),
        ("code", "assert torch.cuda.is_available(), 'Для обучения нужен GPU-сервер: Select Kernel → Colab → New Colab Server → GPU'\n"
                 "run('python -m mammo.prepare_yolo')\n"
                 "\n"
                 "def train_once(name, cmd):\n"
                 "    # Skip a model already trained on this server (e.g. after a restart), then save it to HF.\n"
                 "    if os.path.exists(f'models/{name}') and not RETRAIN:\n"
                 "        print(f'models/{name} уже есть — пропускаю')\n"
                 "    else:\n"
                 "        run(cmd)\n"
                 "    upload(name)\n"
                 "\n"
                 "train_once('screening.pt', f'python -m train.train_screening --epochs {SCREENING_EPOCHS} --batch 12 --workers 2 --device cuda')"),
        ("markdown", "### Классификатор очагов: BI-RADS находки + P(BI-RADS 4–5), не диагноз рака (~0.5 ч)"),
        ("code", "train_once('classifier.pt', f'python -m train.train_classifier --epochs {CLASSIFIER_EPOCHS} --batch 16 --workers 2 --device cuda')"),
        ("markdown", "### Детектор очагов YOLO11s, 1280 px (~3.5 ч)"),
        ("code", "train_once('detector.pt', f'python -m train.train_detector --epochs {DETECTOR_EPOCHS} --batch 8 --device 0')"),
        ("markdown", "### Оценка на официальном test VinDr (1000 исследований)\n"
                     "Главное: `auc_1_2_vs_4_5`, чувствительность/специфичность, `false boxes per image` на здоровых, mAP.\n\n"
                     "Локально потом: `.venv/bin/hf download <логин>/mammoai-weights --local-dir models`"),
        ("code", "run('python -m train.evaluate --device cuda')"),
    ]
    nb = {
        "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                   **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
        "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "train" / "colab_vscode.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
