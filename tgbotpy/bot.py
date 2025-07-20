#!/usr/bin/env python3
# bot.py
import cv2
import os
import tempfile
import logging
import sqlite3
import csv
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from pathlib import Path

# --- Configuration ---
BOT_TOKEN = open((Path(__file__).parent.parent / "token.txt"), "r", encoding="utf-8").read().strip()
LOG_FILE = os.path.join(os.path.dirname(__file__), 'bot.log')
DB_PATH = os.path.join(os.path.dirname(__file__), 'bot_stats.db')

# --- Logging setup ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# --- Database setup ---
DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
DB_CONN.execute("PRAGMA foreign_keys = ON;")

def init_db():
    cur = DB_CONN.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensors (
            sensor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            zone        TEXT NOT NULL,
            type        TEXT NOT NULL,
            UNIQUE(zone, type)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS function_calls (
            call_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            function_name TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            reading_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            sensor_id        INTEGER NOT NULL,
            value            REAL NOT NULL,
            call_id          INTEGER,
            FOREIGN KEY(sensor_id) REFERENCES sensors(sensor_id),
            FOREIGN KEY(call_id) REFERENCES function_calls(call_id)
        );
    """)
    DB_CONN.commit()

init_db()

authorized_user_id = None

# --- Helper functions ---
def record_function_call(function_name: str):
    ts = datetime.now(timezone.utc).isoformat()
    cur = DB_CONN.cursor()
    cur.execute(
        "INSERT INTO function_calls (timestamp, function_name) VALUES (?, ?)",
        (ts, function_name)
    )
    DB_CONN.commit()
    return cur.lastrowid

async def record_readings(context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    last_run = getattr(context.job, 'data', {}).get('last_run')
    cur = DB_CONN.cursor()
    call_id = None
    if last_run:
        cur.execute(
            "SELECT call_id FROM function_calls WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 1", 
            (last_run.isoformat(),)
        )
        row = cur.fetchone()
        if row:
            call_id = row[0]

    # Read thermal zones
    thermal_dir = "/sys/class/thermal"
    if os.path.exists(thermal_dir):
        for entry in os.listdir(thermal_dir):
            if entry.startswith("thermal_zone"):
                zone_path = os.path.join(thermal_dir, entry)
                type_file = os.path.join(zone_path, "type")
                temp_file = os.path.join(zone_path, "temp")
                if os.path.exists(type_file) and os.path.exists(temp_file):
                    with open(type_file) as f:
                        type_str = f.read().strip()
                    with open(temp_file) as f:
                        raw = f.read().strip()
                    try:
                        value = int(raw) / 1000.0
                    except ValueError:
                        continue
                    cur.execute(
                        "INSERT OR IGNORE INTO sensors(zone, type) VALUES (?, ?)",
                        (entry, type_str)
                    )
                    cur.execute(
                        "SELECT sensor_id FROM sensors WHERE zone = ? AND type = ?",
                        (entry, type_str)
                    )
                    sensor_id = cur.fetchone()[0]
                    cur.execute(
                        "INSERT INTO readings(timestamp, sensor_id, value, call_id) VALUES (?, ?, ?, ?)",
                        (ts, sensor_id, value, call_id)
                    )
    DB_CONN.commit()
    context.job.data = {'last_run': now}

# --- Command handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global authorized_user_id
    user = update.effective_user
    username = user.username or "unknown"
    user_id = user.id
    record_function_call('start')
    if authorized_user_id is None:
        authorized_user_id = user_id
        logging.info(f"🔓 Access granted: {username} (ID: {user_id})")
        await update.message.reply_text(f"Access granted. Your ID: {user_id}")
    elif user_id == authorized_user_id:
        await update.message.reply_text("You are already authorized.")
    else:
        await update.message.reply_text("⛔ Access restricted.")

async def cpuinfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    record_function_call('cpuinfo')
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
        await update.message.reply_text(f"❌ Error: {e}")

async def temp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    record_function_call('temp')
    await update.message.reply_text("Temperature logging active every 40 seconds.")

async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    record_function_call('logs')
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("📄 Log not found.")
    else:
        await update.message.reply_document(document=open(LOG_FILE, "rb"), filename="bot.log")

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global authorized_user_id
    user = update.effective_user
    user_id = user.id
    record_function_call('photo')
    if user_id != authorized_user_id:
        await update.message.reply_text("⛔ Access denied.")
        return
    cam = None
    for i in range(0, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cam = cap
            break
    if not cam:
        await update.message.reply_text("⚠️ Cannot open camera.")
        return
    ret, frame = cam.read()
    cam.release()
    if not ret:
        await update.message.reply_text("❌ Capture failed.")
        return
    # Enhance
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, enhanced)
        path = tmp.name
    with open(path, 'rb') as img:
        await update.message.reply_photo(photo=img)
    os.remove(path)

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    record_function_call('export')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
        w = csv.writer(tmp)
        w.writerow(['timestamp','zone','type','value','function'])
        cur = DB_CONN.cursor()
        cur.execute(
            "SELECT r.timestamp, s.zone, s.type, r.value, f.function_name "
            "FROM readings r "
            "LEFT JOIN sensors s ON r.sensor_id=s.sensor_id "
            "LEFT JOIN function_calls f ON r.call_id=f.call_id "
            "ORDER BY r.timestamp"
        )
        for row in cur.fetchall():
            w.writerow(row)
        tmp_path = tmp.name
    with open(tmp_path,'rb') as f:
        await update.message.reply_document(document=f, filename='readings.csv')
    os.remove(tmp_path)

# --- Main ---
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cpuinfo", cpuinfo))
    app.add_handler(CommandHandler("temp", temp))
    app.add_handler(CommandHandler("logs", logs))
    app.add_handler(CommandHandler("photo", photo))
    app.add_handler(CommandHandler("export", export))
    job = app.job_queue.run_repeating(record_readings, interval=40, first=0)
    job.data = {'last_run': None}
    app.run_polling()

if __name__ == "__main__":
    main()
