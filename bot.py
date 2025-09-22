import os
import io
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile, InputMediaPhoto


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yolo-aiogram")

# --- конфиг ---
TELEGRAM_TOKEN = "8496513536:AAF7UxzZnZMOTd9XvaYkQ33PKaigSxZknUY"
MODEL_PATH = r"C:\Users\hsudnik\PycharmProjects\BOTYOLO\best.pt"

# --- модель (загрузим один раз) ---
MODEL = YOLO(MODEL_PATH)
# При наличии GPU:
# MODEL.to("cuda")

HELP_TEXT = (
    "Привет! Пришли мне фото — я верну аннотированную картинку.\n"
    "Можно присылать как *фото*, так и *документ* (jpg/png/webp/bmp).\n"
    "Команды: /start, /help"
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yolo-aiogram")

# =========================
# Конфиг
# =========================

CONF = float(os.getenv("YOLO_CONF", "0.40"))

# =========================
# Загрузка модели
# =========================
log.info(f"Загружаю YOLO модель: {MODEL_PATH}")
MODEL = YOLO(str(MODEL_PATH))
# Если есть GPU:
# MODEL.to("cuda")
DEST_GROUP_ID = '-1002838445249'
HELP_TEXT = (
    "Пришлите фото/картинку — бот вернёт аннотированное PNG и сводку по дефектам.\n"
    f"Порог уверенности: {CONF:.2f} (можно задать через YOLO_CONF)\n"
    "Поддержка: photo или документ (jpg/png/webp/bmp).\n"
    "Команды: /start, /help"
)

# =========================
# Тайлинговый инференс (640x640)
# =========================
# =========================
TILE_SIZE = 480         # размер тайла
TILE_OVERLAP = 0.20     # доля перекрытия
NMS_IOU = 0.3           # IoU-порог для NMS (после склейки)
MAX_MEDIA_PER_ALBUM = 10  # Telegram ограничение

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
        cv2.putText(img_bgr, text, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def _encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Не удалось кодировать результат в PNG.")
    return buf.tobytes()

def infer_and_render_with_tiles(
    img_bgr: np.ndarray, conf: float
) -> Tuple[bytes, dict, List[Tuple[bytes, bytes, str]], dict]:
    """
    Возвращает:
      - annotated_whole_png: аннотированное целое изображение,
      - summary: словарь по дефектам ({total, per_class, avg_conf}),
      - tiles_all: список по ВСЕМ тайлам [(raw_tile_png, annotated_tile_png, caption), ...],
      - meta: {"H": int, "W": int, "num_tiles": int}
    """
    H, W = img_bgr.shape[:2]
    xs = _tile_positions(W, TILE_SIZE, TILE_OVERLAP)
    ys = _tile_positions(H, TILE_SIZE, TILE_OVERLAP)
    num_tiles = len(xs) * len(ys)

    all_boxes = []
    all_scores = []
    all_classes = []

    tiles_all: List[Tuple[bytes, bytes, str]] = []
    t_idx = 0

    for y in ys:
        for x in xs:
            crop = img_bgr[y:min(y + TILE_SIZE, H), x:min(x + TILE_SIZE, W)]
            pad_h = TILE_SIZE - crop.shape[0]
            pad_w = TILE_SIZE - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop = cv2.copyMakeBorder(
                    crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            t_idx += 1

            # инференс тайла
            res = MODEL.predict(crop, conf=conf, imgsz=TILE_SIZE, verbose=False)[0]
            boxes = getattr(res, "boxes", None)

            # подготовим версии тайла для группы: RAW и Annotated (даже если боксов нет)
            raw_tile_png = _encode_png(crop)
            tile_annotated_bgr = res.plot() if (boxes is not None and len(boxes) > 0) else crop.copy()
            annotated_tile_png = _encode_png(tile_annotated_bgr)
            caption_tile = f"tile#{t_idx} — x={x}, y={y}"
            tiles_all.append((raw_tile_png, annotated_tile_png, caption_tile))

            # перенос детекций в координаты исходника
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
        meta = {"H": H, "W": W, "num_tiles": num_tiles}
        return annotated_whole_png, summary, tiles_all, meta

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    # class-wise NMS
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

    # рисуем на исходнике
    annotated = img_bgr.copy()
    _draw_annotations(annotated, kept_boxes, kept_scores, kept_classes, MODEL.names)
    annotated_whole_png = _encode_png(annotated)

    # сводка по дефектам
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
    meta = {"H": H, "W": W, "num_tiles": num_tiles}
    return annotated_whole_png, summary, tiles_all, meta

def build_summary_from_dict(summary: dict) -> str:
    lines = [f"Найдено дефектов: {summary['total']}"]
    if summary["per_class"]:
        for name, cnt in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {name}: {cnt}")
    if summary["total"] > 0:
        lines.append(f"Средняя уверенность: {summary['avg_conf']:.2f}")
    return "\n".join(lines)

def build_group_meta_text(meta: dict) -> str:
    return f"Исходное изображение: {meta['W']}×{meta['H']}\nТайлов 640×640: {meta['num_tiles']}"

async def reply_with_annotated(message: Message, png_bytes: bytes, caption: str):
    await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
    photo = BufferedInputFile(png_bytes, filename="annotated.png")
    await message.reply_photo(photo=photo, caption=caption)

async def send_tiles_album_to_group(bot: Bot, tiles_all: List[Tuple[bytes, bytes, str]], group_id: int, meta_text: str):
    """
    Отправляет в группу ВСЕ тайлы (RAW + Annotated), батчами по 10 медиа.
    Дополнительно шлёт текстовую сводку meta_text.
    """
    if not group_id:
        return
    if tiles_all:
        media: List[InputMediaPhoto] = []
        for idx, (raw_png, ann_png, caption) in enumerate(tiles_all, start=1):
            media.append(InputMediaPhoto(media=BufferedInputFile(raw_png, filename=f"tile_raw_{idx}.png"),
                                         caption=f"{caption} — RAW"))
            media.append(InputMediaPhoto(media=BufferedInputFile(ann_png, filename=f"tile_ann_{idx}.png"),
                                         caption=f"{caption} — Annotated"))

        for i in range(0, len(media), MAX_MEDIA_PER_ALBUM):
            chunk = media[i:i+MAX_MEDIA_PER_ALBUM]
            await bot.send_chat_action(group_id, ChatAction.UPLOAD_PHOTO)
            await bot.send_media_group(chat_id=group_id, media=chunk)

    # В конце — текстовая сводка по размеру и числу тайлов
    if meta_text:
        await bot.send_message(chat_id=group_id, text=meta_text)

# =========================
# Хендлеры
# =========================
async def cmd_start(message: Message):
    await message.answer(HELP_TEXT, parse_mode="Markdown")

async def cmd_help(message: Message):
    await message.answer(HELP_TEXT, parse_mode="Markdown")

async def _process_bytes(message: Message, raw_bytes: bytes):
    img_bgr = decode_image_from_bytes(raw_bytes)

    annotated_png, summary, tiles_all, meta = infer_and_render_with_tiles(img_bgr, conf=CONF)

    # 1) пользователю — целое изображение с подписью
    caption = build_summary_from_dict(summary)
    await reply_with_annotated(message, annotated_png, caption=caption)

    # 2) в группу — ВСЕ тайлы (RAW + Annotated) + сводка размера/кол-ва тайлов
    if DEST_GROUP_ID:
        try:
            meta_text = build_group_meta_text(meta)
            await send_tiles_album_to_group(message.bot, tiles_all, DEST_GROUP_ID, meta_text)
        except Exception as e:
            log.exception("Не удалось отправить тайлы в группу: %s", e)

async def on_photo(message: Message):
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    bio = io.BytesIO()
    await message.bot.download_file(file.file_path, bio)
    await _process_bytes(message, bio.getvalue())

async def on_document(message: Message):
    doc = message.document
    name = (doc.file_name or "").lower()
    is_image_ext = name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    is_image_mime = (doc.mime_type or "").startswith("image/")
    if is_image_ext or is_image_mime:
        file = await message.bot.get_file(doc.file_id)
        bio = io.BytesIO()
        await message.bot.download_file(file.file_path, bio)
        await _process_bytes(message, bio.getvalue())
    else:
        await message.reply("Пришлите изображение как документ (jpg/png/webp/bmp).")

# =========================
# Entrypoint
# =========================
async def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN пуст. Задай в .env или окружении.")
    bot = Bot(TELEGRAM_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_document, F.document & ~F.via_bot)

    log.info("Бот запущен (polling). Ожидаю сообщения...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
