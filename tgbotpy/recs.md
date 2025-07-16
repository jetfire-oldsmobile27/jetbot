В версии `python-telegram-bot` ≥ 20 флаг `use_context` убран, а сам `Updater` заменён на более современный `Application`. Ниже — обновлённый код вашего бота под v20+ (токен по‑прежнему оставлен пустым), а также скорректированный `requirements.txt`.

```python
#!/usr/bin/env python3
# bot.py

import cv2
import os
import tempfile
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# Вставьте ваш токен между кавычками или через переменную окружения BOT_TOKEN
BOT_TOKEN = os.getenv("BOT_TOKEN", "")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я могу отправить фото с веб-камеры. Используй команду /photo."
    )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cam = cv2.VideoCapture(0)  # /dev/video0
    if not cam.isOpened():
        await update.message.reply_text("Не удалось открыть веб-камеру.")
        return

    ret, frame = cam.read()
    cam.release()
    if not ret:
        await update.message.reply_text("Не удалось захватить изображение.")
        return

    # Сохраняем снимок во временный файл в безопасной папке
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as img:
        await update.message.reply_photo(photo=img)

    os.remove(tmp_path)


def main() -> None:
    # Собираем приложение
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Регистрируем команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("photo", photo))

    # Запускаем поллинг
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

### requirements.txt

```
python-telegram-bot>=20.4
opencv-python>=4.7.0.72
```

---

> **Краткая инструкция по установке на Orange Pi OS (Arch Linux ARM64)**
> *(Arch Linux ARM, ядро ARM64)*

1. **Обновите систему и установите базовый набор**

   ```bash
   sudo pacman -Syu
   sudo pacman -S --needed base-devel git python python-pip v4l-utils
   ```

2. **Убедитесь, что видеоустройство доступно**

   ```bash
   ls /dev/video0
   ```

   Если `/dev/video0` не появился, загрузите модули:

   ```bash
   sudo modprobe bcm2835-v4l2   # или другой драйвер вашей платы
   ```

3. **(Опционально) Создайте и активируйте виртуальное окружение**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Установите зависимости**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Задайте токен и запустите бота**

   ```bash
   export BOT_TOKEN="ВАШ_ТОКЕН_ТЕЛЕГРАМ"
   python bot.py
   ```

После этого бот будет доступен в Telegram:

* `/start` — приветствие
* `/photo` — снятие и отправка кадра с веб‑камеры.
