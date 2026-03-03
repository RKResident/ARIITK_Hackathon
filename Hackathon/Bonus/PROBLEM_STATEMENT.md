# Bonus Level — Operation FREEFALL: Autonomous Precision Landing

> **Team Aerial Robotics IITK| Y25 Recruitment Hackathon**

---

## 1. Context
Your autonomous drone must execute a **precision landing sequence** inside a high-fidelity 3D physics simulator.

Beneath your drone are landing pads marked with ArUco markers. Your objective is to design a closed-loop flight controller that uses Computer Vision to detect the marker in real-time (capturing the simulator screen using a library such as PyAutoGUI for screenshot acquisition), calculates the positional error, and sends flight commands to land the drone precisely on the target — without ever hardcoding coordinates.

You will not write code inside the simulator. Instead, your Python script (`auto_land.py`) acts as an **external flight computer**, communicating with the simulator over the network.

---

## 2. The Mission Phases

This level is divided into two escalating challenges:

### Phase 1 — The Static Dropzone

| Parameter | Detail |
|-----------|--------|
| Target | A stationary landing pad equipped with an ArUco marker |
| Objective | Take off, visually acquire the marker, align X and Z axes, and steadily decrease altitude (Y-axis) until touchdown |
| Success Criteria | The drone rests completely inside the marker's boundary for at least **3 seconds** without tipping over |

### Phase 2 — The Moving Platform (Dynamic Dropzone)

| Parameter | Detail |
|-----------|--------|
| Target | An ArUco marker mounted on an Autonomous Ground Vehicle (AGV) moving unpredictably across the ground |
| Objective | Intercept the moving target, match its velocity, maintain vertical alignment, and execute a dynamic landing |
| Success Criteria | Touchdown and stable rest on the moving platform |

---

## 3. Simulator Communication Protocol (The API)

Your Python script communicates with the simulator via **network sockets** and **HTTP requests**. The simulator binaries are too large for GitHub — download them from the link below.

> **Download Simulator:** [Google Drive — Bonus Level Simulator](https://drive.google.com/drive/folders/1j8hUUgTsx_cl2vQLKtHT_-zmelxxZHoK?usp=sharing)
>
> | File | Platform |
> |------|----------|
> | `drone (1).exe` | Windows |
> | `drone.x86_64` | Linux |
> | `drone.zip` | MacOS |

### A. Telemetry Stream (Simulator → Your Script)

The simulator broadcasts real-time physics data over a **TCP socket** at `127.0.0.1` on port **8081**.

**Manual test (Netcat):**
```bash
nc 127.0.0.1 8081
```

**Data Format:** Continuous JSON strings. Note that the **Y-axis represents altitude**.

```json
{
  "name": "Drone",
  "position": [0.0, 1.35060441493988, 0.0],
  "rotation": [0.0, 0.0, 0.0],
  "velocity": [0.0, 0.232620656490326, 0.0]
}
```

| Field | Description |
|-------|-------------|
| `position` | `[X, Y, Z]` — Y is altitude |
| `rotation` | `[pitch, yaw, roll]` in degrees |
| `velocity` | `[vX, vY, vZ]` — vY is vertical velocity |

### B. Flight Control (Your Script → Simulator)

Send flight commands via **HTTP POST** to `http://localhost:8080`.

**Manual test (cURL):**
```bash
curl localhost:8080 --json '{ "pitch": -1.0 }'
```

**Command Dictionary:** Values typically range from `-1.0` to `1.0`.

| Command | Effect |
|---------|--------|
| `"pitch"` | Forward / backward tilt |
| `"roll"` | Left / right tilt |
| `"yaw"` | Left / right rotation |
| `"throttle"` | Vertical thrust |

You can send any combination of these in a single JSON payload:
```json
{ "pitch": -0.3, "roll": 0.1, "throttle": 0.8 }
```

---

## 4. Your Task — The Control Loop

Write a Python script `auto_land.py` that implements a continuous feedback loop at approximately **30–60 Hz**:

```
┌─────────────────────────────────────────────────────────┐
│                    CONTROL LOOP                         │
│                                                         │
│  1. READ     ← Telemetry from port 8081                │
│              ← Camera frame from /video                 │
│                                                         │
│  2. PROCESS  → Detect ArUco marker (OpenCV)            │
│              → Calculate pixel error from frame center  │
│                                                         │
│  3. COMPUTE  → PID Controller converts visual error     │
│                into smooth flight commands               │
│                                                         │
│  4. ACT      → Send pitch, roll, throttle, yaw to 8080 │
│                                                         │
│  └── Repeat ──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 5. Constraints & Guidelines

| Rule | Detail |
|------|--------|
| **Allowed Libraries** | `requests`, `socket`, `opencv-python` (cv2), `numpy`, `simple_pid`, `pyautogui` |
| **No Hardcoding** | You **cannot** hardcode the target's physical coordinates. The drone **MUST** use the camera feed and ArUco detection to find the landing zone. |
| **Out of Bounds** | Flying the drone outside the simulated arena walls results in immediate failure. |
| **ArUco Dictionary** | Must use `DICT_4X4_50` |

---

## 6. Deliverables

Teams must submit a single GitHub repository containing:

| # | File | Description |
|---|------|-------------|
| 1 | `auto_land.py` | Your main control loop script. Must connect to the simulator, detect ArUco markers, and land the drone autonomously. |
| 2 | `landing_video.mp4` | A screen recording of a successful landing (Phase 1 minimum, Phase 2 for full marks). |
| 3 | `report.pdf` | Maximum 2 pages. Must include: your PID tuning approach, your ArUco detection pipeline, how you handle Phase 2's moving target, and a block diagram of your control loop. |

### `report.pdf` Must Answer These Questions

- How does your controller detect and lock onto the ArUco marker?
- How do you convert pixel error to physical flight commands?
- What PID gains did you use and how did you tune them?
- How does your system handle the marker going out of the camera frame?
- How does your approach change for the moving platform in Phase 2?

---

## 7. Scoring

| Metric | Weightage |
|--------|--------|
| **Phase 1 Completion** (static landing) | **30%** |
| **Phase 2 Completion** (moving platform) | **50%** |
| **Landing Smoothness** | Up to **20%** bonus (touchdown velocity closer to 0.0 m/s → higher score) |
| **Time to Land** | Faster landings break ties |

---

## 8. Running the Simulator

**Step 1:** Download the simulator from [Google Drive](https://drive.google.com/drive/folders/1j8hUUgTsx_cl2vQLKtHT_-zmelxxZHoK?usp=sharing).

**Step 2:** Run the simulator for your platform:

### Windows
```bash
.\"drone (1).exe"
```

### Linux
```bash
chmod +x drone.x86_64
./drone.x86_64
```

**Step 3:** In a separate terminal, run your controller:
```bash
python auto_land.py
```

> The simulator window will show the drone responding to your commands in real-time.
---
