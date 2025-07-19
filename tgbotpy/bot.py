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
    ContextTypes
)
from pathlib import Path

BOT_TOKEN = open((Path(__file__).parent.parent / "token.txt"), "r", encoding="utf-8").read().strip()
LOG_FILE = os.path.join(os.path.dirname(__file__), 'bot.log')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

authorized_user_id = None


async def cpuinfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет содержимое /proc/cpuinfo"""
    try:
        with open("/proc/cpuinfo", "r") as f:
            text = f.read()
        if len(text) < 3500:
            await update.message.reply_text(text)
        else:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(text.encode())
                tmp_path = tmp.name
            await update.message.reply_document(document=open(tmp_path, "rb"), filename="cpuinfo.txt")
            os.remove(tmp_path)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

async def temp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет температуру с датчиков"""
    report = []
    
    thermal_dir = "/sys/class/thermal"
    if os.path.exists(thermal_dir):
        for entry in os.listdir(thermal_dir):
            if entry.startswith("thermal_zone"):
                zone_path = os.path.join(thermal_dir, entry)
                type_file = os.path.join(zone_path, "type")
                temp_file = os.path.join(zone_path, "temp")
                
                if os.path.exists(type_file) and os.path.exists(temp_file):
                    with open(type_file, "r") as f:
                        type_str = f.read().strip()
                    with open(temp_file, "r") as f:
                        temp_str = f.read().strip()
                    
                    try:
                        temp = int(temp_str) / 1000.0
                        report.append(f"[{entry}] {type_str}: {temp:.2f} °C")
                    except:
                        report.append(f"[{entry}] {type_str}: invalid ({temp_str})")
    
    hwmon_dir = "/sys/class/hwmon"
    if os.path.exists(hwmon_dir):
        for entry in os.listdir(hwmon_dir):
            entry_path = os.path.join(hwmon_dir, entry)
            name_file = os.path.join(entry_path, "name")
            
            if os.path.exists(name_file):
                with open(name_file, "r") as f:
                    chip = f.read().strip()
            else:
                chip = entry
            
            for f in os.listdir(entry_path):
                if f.startswith("temp") and "_input" in f:
                    idx = f[4:f.find("_input")]
                    label_file = os.path.join(entry_path, f"temp{idx}_label")
                    label = ""
                    
                    if os.path.exists(label_file):
                        with open(label_file, "r") as fl:
                            label = fl.read().strip()
                    else:
                        label = f"temp{idx}"
                    
                    input_file = os.path.join(entry_path, f)
                    with open(input_file, "r") as fi:
                        temp_str = fi.read().strip()
                    
                    try:
                        temp = int(temp_str) / 1000.0
                        report.append(f"[{entry}] {chip} {label}: {temp:.2f} °C")
                    except:
                        report.append(f"[{entry}] {chip} {label}: invalid ({temp_str})")
    
    if not report:
        await update.message.reply_text("❌ Не найдены температурные датчики.")
    else:
        await update.message.reply_text("\n".join(report))

async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет лог-файл"""
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("📄 Лог за сегодня не найден.")
    else:
        await update.message.reply_document(document=open(LOG_FILE, "rb"), filename="bot.log")

# === Основные команды ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start"""
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
    """Сделать фото с камерой"""
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
    
    for _ in range(2):
        ret, frame = cam.read()
        if not ret:
            cam.release()
            await update.message.reply_text("Не удалось захватить изображение.")
            return
    
    cam.release()
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, enhanced)
        tmp_path = tmp.name
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"📸 Фото отправлено пользователю {username} (ID: {user_id}) в {timestamp}")
    
    try:
        with open(tmp_path, "rb") as img:
            await update.message.reply_photo(photo=img)
    finally:
        os.remove(tmp_path)

def main() -> None:
    """Запуск бота"""
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("photo", photo))
    app.add_handler(CommandHandler("cpuinfo", cpuinfo))
    app.add_handler(CommandHandler("temp", temp))
    app.add_handler(CommandHandler("logs", logs))
    
    app.run_polling()

if __name__ == "__main__":
    main()