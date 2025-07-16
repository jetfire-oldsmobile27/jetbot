#!/usr/bin/env python3
# bot.py

import cv2
import os
import tempfile
import logging
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

BOT_TOKEN = "7639692292:AAELfPguE_-DbZq-BUfixw885VYPwHPhErs"  # ← Вставьте токен или задайте через переменную окружения
LOG_FILE = os.path.join(os.path.dirname(__file__), 'bot.log')

print(cv2.getBuildInformation())
cap = cv2.VideoCapture(0)
print("Backend:", cap.getBackendName())
ret, frame = cap.read()
print(ret, frame.shape if frame is not None else None)
exit

# Настройка логирования
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Глобальная переменная для хранения авторизованного пользователя
authorized_user_id = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global authorized_user_id
    user = update.effective_user
    username = user.username or "unknown"
    user_id = user.id

    if authorized_user_id is None:
        authorized_user_id = user_id
        logging.info(f"🔓 Доступ выдан: {username} (ID: {user_id})")
        await update.message.reply_text(
            f"Вы получили доступ к боту.\nВаш ID: {user_id}"
        )
    elif user_id == authorized_user_id:
        await update.message.reply_text("Вы уже авторизованы.")
    else:
        await update.message.reply_text("⛔ Доступ к боту разрешён только одному пользователю.")


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global authorized_user_id
    user = update.effective_user
    username = user.username or "unknown"
    user_id = user.id

    if user_id != authorized_user_id:
        logging.warning(f"❌ Несанкционированный доступ: {username} (ID: {user_id})")
        await update.message.reply_text("⛔ У вас нет доступа к этой функции.")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        await update.message.reply_text("Не удалось открыть веб-камеру.")
        return

    ret, frame = cam.read()
    cam.release()
    if not ret:
        await update.message.reply_text("Не удалось захватить изображение.")
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame)
        tmp_path = tmp.name

    # Логируем фото-запрос
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"📸 Фото отправлено пользователю {username} (ID: {user_id}) в {timestamp}")

    with open(tmp_path, "rb") as img:
        await update.message.reply_photo(photo=img)

    os.remove(tmp_path)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("photo", photo))
    app.run_polling()


if __name__ == "__main__":
    main()
