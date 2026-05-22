#!/usr/bin/env python3
"""
gamepad_driver.py — Xbox-compatible gamepad → robot socket driver

Usage:
    python3 gamepad_driver.py <robot_ip> [--port 8327] [--deadzone 0.15]
    python3 gamepad_driver.py <robot_ip> --stick right   # переключить на правый стик

Requirements:
    pip install inputs

Controls:
    Левый ИЛИ правый стик (--stick left|right):
        Up/Down       forward / backward
        Left/Right    pivot left / right
        Diagonal      curve
        Centered      stop
    B       emergency stop
    Start   quit

Если стик не реагирует — запусти gamepad_probe.py и посмотри коды осей,
потом передай их через --axis-x и --axis-y.
"""

import socket, sys, time, argparse, threading

try:
    from inputs import get_gamepad, UnpluggedError
except ImportError:
    sys.exit("pip install inputs")

PORT     = 8327
LOOP_HZ  = 20

# Известные коды осей для разных стиков
STICK_AXES = {
    "left":  {"x": "ABS_X",  "y": "ABS_Y"},
    "right": {"x": "ABS_RX", "y": "ABS_RY"},
}

class State:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.btn_b     = False
        self.btn_start = False
        self.lock = threading.Lock()

state     = State()
stop_flag = threading.Event()
axis_x    = "ABS_X"
axis_y    = "ABS_Y"

def gamepad_reader():
    CENTER = 127.5
    HALF_RANGE = 127.5
    
    while not stop_flag.is_set():
        try:
            events = get_gamepad()
        except UnpluggedError:
            print("\n[gamepad] unplugged, retry in 2s…")
            time.sleep(2); continue
        except Exception as e:
            print(f"\n[gamepad] {e}"); time.sleep(0.5); continue

        with state.lock:
            for ev in events:
                t, c, v = ev.ev_type, ev.code, ev.state
                if t == "Absolute":
                    if c == axis_x:
                        state.x = (v - CENTER) / HALF_RANGE
                    elif c == axis_y:
                        state.y = -(v - CENTER) / HALF_RANGE  # invert: up=+
                elif t == "Key":
                    if   c == "BTN_EAST":  state.btn_b     = bool(v)
                    elif c == "BTN_START": state.btn_start = bool(v)

def dz(v, d):
    if abs(v) < d: return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - d) / (1.0 - d)

def connect(ip, port):
    for i in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip, port))
            s.settimeout(2.0)
            print(f"[net] {s.recv(64).decode().strip()}")
            return s
        except OSError as e:
            print(f"[net] {i+1}/10: {e}"); time.sleep(1)
    sys.exit("[net] failed")

def send_cmd(sock, cmd):
    try:
        sock.sendall((cmd + "\n").encode())
        return sock.recv(64).decode().strip()
    except OSError:
        return None

def control_loop(sock, args):
    interval = 1.0 / LOOP_HZ
    last_cmd = ""
    print(f"[ctrl] stick={args.stick}  axes: x={axis_x} y={axis_y}")
    print("[ctrl] B=stop  Start=quit")

    while not stop_flag.is_set():
        t0 = time.monotonic()

        with state.lock:
            x       = dz(state.x, args.deadzone)
            y       = dz(state.y, args.deadzone)
            btn_b   = state.btn_b
            btn_s   = state.btn_start

        if btn_s:
            send_cmd(sock, "QUIT"); stop_flag.set(); break

        if btn_b or (x == 0.0 and y == 0.0):
            cmd = "STOP"
        else:
            DIAG = 0.4
            # Определяем направление, ms=0 для непрерывного движения
            if abs(y) >= DIAG and abs(x) >= DIAG:
                # диагональ → кривая
                if y > 0:
                    cmd = "CURVE_FR 0" if x > 0 else "CURVE_FL 0"
                else:
                    cmd = "CURVE_BR 0" if x > 0 else "CURVE_BL 0"
            elif abs(y) >= abs(x):
                # преимущественно вертикаль
                cmd = "FORWARD 0" if y > 0 else "BACKWARD 0"
            else:
                # поворот на месте
                cmd = "RIGHT 0" if x > 0 else "LEFT 0"

        if cmd != last_cmd:
            resp = send_cmd(sock, cmd)
            if resp is None:
                print("\n[net] lost"); stop_flag.set(); break
            last_cmd = cmd
            print(f"\r[ctrl] {cmd:<35} {resp:<6}", end="", flush=True)

        time.sleep(max(0.0, interval - (time.monotonic() - t0)))

def main():
    global axis_x, axis_y

    p = argparse.ArgumentParser()
    p.add_argument("ip")
    p.add_argument("--port",     type=int,   default=PORT)
    p.add_argument("--deadzone", type=float, default=0.15)
    p.add_argument("--stick",    choices=["left","right"], default="left")
    p.add_argument("--axis-x",   default=None,
                   help="override X axis code, e.g. ABS_X, ABS_RX, ABS_HAT0X")
    p.add_argument("--axis-y",   default=None,
                   help="override Y axis code, e.g. ABS_Y, ABS_RY")
    args = p.parse_args()

    axes   = STICK_AXES[args.stick]
    axis_x = args.axis_x or axes["x"]
    axis_y = args.axis_y or axes["y"]

    sock   = connect(args.ip, args.port)
    t      = threading.Thread(target=gamepad_reader, daemon=True)
    t.start()

    try:
        control_loop(sock, args)
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        send_cmd(sock, "STOP")
        sock.close()
        stop_flag.set()
        print("\n[main] bye")

if __name__ == "__main__":
    main()