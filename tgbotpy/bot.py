# bot.py
import cv2
import os
import tempfile
import logging
import sqlite3
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
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
# Ensure SQLite connection is thread-safe for job queue
DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
DB_CONN.execute("PRAGMA foreign_keys = ON;")
def init_db():
    """Create tables: sensors, function_calls, readings."""
    cur = DB_CONN.cursor()
    # sensors: one per zone/type
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensors (
            sensor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            zone        TEXT NOT NULL,
            type        TEXT NOT NULL,
            UNIQUE(zone, type)
        );
    """)
    # function calls
    cur.execute("""
        CREATE TABLE IF NOT EXISTS function_calls (
            call_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            function_name TEXT NOT NULL
        );
    """)
    # readings: references sensor and (optionally) function call
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

# Initialize DB on import
init_db()

# Authorized user
authorized_user_id = None

# --- Helper functions ---
def record_function_call(function_name: str):
    """Insert a function call record and return its call_id."""
    ts = datetime.now(timezone.utc).isoformat()
    cur = DB_CONN.cursor()
    cur.execute(
        "INSERT INTO function_calls (timestamp, function_name) VALUES (?, ?)",
        (ts, function_name)
    )
    DB_CONN.commit()
    return cur.lastrowid

async def record_readings(job_context: ContextTypes.DEFAULT_TYPE) -> None:
    """Background task: read sensors and insert readings every interval."""
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    # get last_run from context
    last_run = job_context.job.context.get('last_run')
    # query for most recent call since last_run
    cur = DB_CONN.cursor()
    if last_run:
        cur.execute(
            "SELECT call_id FROM function_calls WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 1", 
            (last_run.isoformat(),)
        )
        row = cur.fetchone()
        call_id = row[0] if row else None
    else:
        call_id = None

    thermal_dir = "/sys/class/thermal"
    # collect zones
    if os.path.exists(thermal_dir):
        for entry in os.listdir(thermal_dir):
            if not entry.startswith("thermal_zone"): continue
            zone_path = os.path.join(thermal_dir, entry)
            type_file = os.path.join(zone_path, "type")
            temp_file = os.path.join(zone_path, "temp")
            if not (os.path.exists(type_file) and os.path.exists(temp_file)): continue

            with open(type_file) as f:
                type_str = f.read().strip()
            with open(temp_file) as f:
                raw = f.read().strip()
            try:
                value = int(raw) / 1000.0
            except ValueError:
                continue
            # sensor upsert
            cur.execute(
                "INSERT OR IGNORE INTO sensors(zone, type) VALUES (?, ?)",
                (entry, type_str)
            )
            cur.execute(
                "SELECT sensor_id FROM sensors WHERE zone = ? AND type = ?", 
                (entry, type_str)
            )
            sensor_id = cur.fetchone()[0]
            # insert reading
            cur.execute(
                "INSERT INTO readings(timestamp, sensor_id, value, call_id) VALUES (?, ?, ?, ?)",
                (ts, sensor_id, value, call_id)
            )
    DB_CONN.commit()
    # update last_run
    job_context.job.context['last_run'] = now

# --- Bot command handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global authorized_user_id
    user = update.effective_user
    username = user.username or "unknown"
    user_id = user.id

    call_id = record_function_call('start')

    if authorized_user_id is None:
        authorized_user_id = user_id
        logging.info(f"🔓 Access granted: {username} (ID: {user_id})")
        await update.message.reply_text(f"Access granted. Your ID: {user_id}")
    elif user_id == authorized_user_id:
        await update.message.reply_text("You are already authorized.")
    else:
        await update.message.reply_text("⛔ Bot access is restricted to one user.")

async def cpuinfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    call_id = record_function_call('cpuinfo')
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
    call_id = record_function_call('temp')
    report = []
    # original sensor reporting logic...
    # For brevity, omit here; data is logged by background job
    await update.message.reply_text("Temperature data is being logged every 40 seconds.")

async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    call_id = record_function_call('logs')
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("📄 Log file not found.")
    else:
        await update.message.reply_document(document=open(LOG_FILE, "rb"), filename="bot.log")

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    call_id = record_function_call('photo')
    # ... existing photo logic ...
    # omit for brevity

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export logged readings to CSV."""
    call_id = record_function_call('export')
    import csv
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['timestamp', 'zone', 'type', 'value', 'function_call'])
        cur = DB_CONN.cursor()
        cur.execute(
            "SELECT r.timestamp, s.zone, s.type, r.value, f.function_name "
            "FROM readings r "
            "JOIN sensors s ON r.sensor_id = s.sensor_id "
            "LEFT JOIN function_calls f ON r.call_id = f.call_id "
            "ORDER BY r.timestamp"
        )
        for row in cur.fetchall():
            writer.writerow(row)
        tmp_path = tmp.name
    with open(tmp_path, 'rb') as f:
        await update.message.reply_document(document=f, filename='readings.csv')
    os.remove(tmp_path)

# === Bot setup ===
def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    # commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("photo", photo))
    app.add_handler(CommandHandler("cpuinfo", cpuinfo))
    app.add_handler(CommandHandler("temp", temp))
    app.add_handler(CommandHandler("logs", logs))
    app.add_handler(CommandHandler("export", export))
    # schedule readings job every 40 seconds
    job = app.job_queue.run_repeating(record_readings, interval=40, first=0)
    job.context = {'last_run': None}
    # start bot
    app.run_polling()

if __name__ == "__main__":
    main()
