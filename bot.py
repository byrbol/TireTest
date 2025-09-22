import os
import io
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile

# ---------- логирование ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yolo-aiogram")

# =========================================================
# Конфиг из переменных окружения (Heroku Config Vars)
# =========================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CONF = float(os.getenv("YOLO_CONF", "0.40"))
GDRIVE_ID = os.getenv("YOLO_GDRIVE_ID", "").strip()

HELP_TEXT = (
    "Пришлите фото/картинку — верну аннотированное PNG и сводку по дефектам.\n"
    f"Порог уверенности: {CONF:.2f} (YOLO_CONF)\n"
    "Команды: /start, /help"
)

# =========================================================
# Загрузка модели: Google Drive -> /tmp/model.pt (через gdown)
# =========================================================
def _resolve_model_path() -> Path:
    if GDRIVE_ID:
        dst = Path("/tmp/model.pt")
        if not dst.exists():
            log.info("Скачиваю модель с Google Drive (id=%s) -> %s", GDRIVE_ID, dst)
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, str(dst), quiet=False)
        return dst
    raise RuntimeError("Не задан YOLO_GDRIVE_ID в Config Vars")

MODEL_PATH = _resolve_model_path()
log.info("Загружаю YOLO модель: %s", MODEL_PATH)
MODEL = YOLO(str(MODEL_PATH))

# =========================================================
# Параметры инференса
# =========================================================
TILE_SIZE = 416      # меньше — экономия памяти
TILE_OVERLAP = 0.20
NMS_IOU = 0.3

# =========================================================
# Утилиты
# =========================================================
def decode_image_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Не удалось прочитать изображение. Проверь формат.")
    return img

def _tile_positions(dim: int, tile: int, overlap: float) -> List[int]:
    if dim <= tile:
        return [0]
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, dim - tile + 1, stride))
    if xs[-1] != dim - tile:
        xs.append(dim - tile)
    return xs

def _to_xyxy(boxes_tensor) -> np.ndarray:
    return boxes_tensor.detach().cpu().numpy().astype(np.float32)

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def _draw_annotations(img_bgr: np.ndarray,
                      boxes: np.ndarray,
                      scores: np.ndarray,
                      classes: np.ndarray,
                      names: dict):
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(int(cls_id), str(int(cls_id)))
        text = f"{label} {score:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, text, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def _encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Не удалось кодировать результат в PNG.")
    return buf.tobytes()

# =========================================================
# Инференс: возвращает только итоговое изображение и сводку
# =========================================================
def infer_and_render(img_bgr: np.ndarray, conf: float) -> Tuple[bytes, dict]:
    H, W = img_bgr.shape[:2]
    xs = _tile_positions(W, TILE_SIZE, TILE_OVERLAP)
    ys = _tile_positions(H, TILE_SIZE, TILE_OVERLAP)

    all_boxes, all_scores, all_classes = [], [], []

    for y in ys:
        for x in xs:
            crop = img_bgr[y:min(y + TILE_SIZE, H), x:min(x + TILE_SIZE, W)]
            pad_h = TILE_SIZE - crop.shape[0]
            pad_w = TILE_SIZE - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

            res = MODEL.predict(crop, conf=conf, imgsz=TILE_SIZE, verbose=False)[0]
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.cls is None or len(boxes) == 0:
                continue

            xyxy = _to_xyxy(boxes.xyxy)
            confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
            cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)

            xyxy[:, [0, 2]] += x
            xyxy[:, [1, 3]] += y
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W - 1)
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H - 1)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W - 1)
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H - 1)

            all_boxes.append(xyxy)
            all_scores.append(confs)
            all_classes.append(cls_ids)

    if len(all_boxes) == 0:
        annotated_whole_png = _encode_png(img_bgr.copy())
        summary = {"total": 0, "per_class": {}, "avg_conf": 0.0}
        return annotated_whole_png, summary

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    keep_indices = []
    for cls in np.unique(all_classes):
        mask = (all_classes == cls)
        b = all_boxes[mask]
        s = all_scores[mask]
        idxs = _nms(b, s, iou_thr=NMS_IOU)
        global_ids = np.where(mask)[0][idxs]
        keep_indices.extend(global_ids.tolist())

    keep_indices = np.array(keep_indices, dtype=int)
    kept_boxes = all_boxes[keep_indices]
    kept_scores = all_scores[keep_indices]
    kept_classes = all_classes[keep_indices]

    annotated = img_bgr.copy()
    _draw_annotations(annotated, kept_boxes, kept_scores, kept_classes, MODEL.names)
    annotated_whole_png = _encode_png(annotated)

    per_class = {}
    for cls_id in np.unique(kept_classes):
        name = MODEL.names.get(int(cls_id), str(int(cls_id)))
        per_class[name] = int((kept_classes == cls_id).sum())
    avg_conf = float(kept_scores.mean()) if len(kept_scores) else 0.0

    summary = {
        "total": int(len(kept_boxes)),
        "per_class": per_class,
        "avg_conf": avg_conf,
    }
    return annotated_whole_png, summary

# =========================================================
# Telegram utils
# =========================================================
def build_summary_from_dict(summary: dict) -> str:
    lines = [f"Найдено дефектов: {summary['total']}"]
    if summary["per_class"]:
        for name, cnt in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {name}: {cnt}")
    if summary["total"] > 0:
        lines.append(f"Средняя уверенность: {summary['avg_conf']:.2f}")
    return "\n".join(lines)

async def reply_with_annotated(message: Message, png_bytes: bytes, caption: str):
    await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
    photo = BufferedInputFile(png_bytes, filename="annotated.png")
    await message.reply_photo(photo=photo, caption=caption)

# =========================================================
# Хендлеры
# =========================================================
async def cmd_start(message: Message):
    await message.answer(HELP_TEXT, parse_mode="Markdown")

async def cmd_help(message: Message):
    await message.answer(HELP_TEXT, parse_mode="Markdown")

async def on_photo(message: Message):
    file = await message.bot.get_file(message.photo[-1].file_id)
    bio = io.BytesIO()
    await message.bot.download_file(file.file_path, bio)
    img = decode_image_from_bytes(bio.getvalue())
    ann, summary = infer_and_render(img, conf=CONF)
    caption = build_summary_from_dict(summary)
    await reply_with_annotated(message, ann, caption)

# =========================================================
# Entrypoint
# =========================================================
async def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN пуст — задай в Config Vars")

    bot = Bot(TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(on_photo, F.photo)

    log.info("Бот запущен (polling)...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
