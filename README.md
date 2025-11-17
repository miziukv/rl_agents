# RL Agents - Unity ML-Agents

A minimal Unity + ML-Agents project for training a navigation agent to reach a goal in a simple challenge course

## What's Included

**Unity scene** (`Assets/Scenes/ChallengeCourse.unity`) with:
- Ground (Plane), Goal (Empty with a Sphere child), optional Hazard_1 (Cube)
- Agent (Capsule) configured with Rigidbody, Behavior Parameters, Decision Requester, Ray Perception Sensor

**Core scripts** (`Assets/Scripts`):
- `CourseAgent.cs` – the RL agent (observations, actions, rewards, resets)
- `FollowOrbitCamera.cs` – simple third-person orbit camera

**Training configs** (`config/`):
- `challenge_ppo.yaml` – PPO trainer settings

## Requirements

- **Unity**: 2022.3 LTS or later (Editor LTS recommended)
- **ML-Agents Unity Package**: installed via Package Manager
- **Python**: 3.9–3.11 (virtual environment recommended)
- **OS**: Windows/macOS/Linux

---

## 1) Unity Setup

### Install Unity

- Use Unity Hub. Install Unity 2022.3 LTS (or a similar LTS).
- Create/open the project in Unity Hub.

### Open the Project

- Launch Unity → open the repo's project folder.
- Open the scene: `Assets/Scenes/ChallengeCourse.unity`.

### Install Packages in Unity

1. **Window → Package Manager**
2. Top-left **"+"** → **Add package by name** → enter:
   ```
   com.unity.ml-agents
   ```
   → **Add**
   
   _(If needed, "Add package from git URL": `https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents`)_

3. Install **Barracuda** (search "Barracuda", package id `com.unity.barracuda`).

### Verify Scene Objects

_(Already set up in repo, but for reference)_

- **Ground**: Plane, Tag = `Ground`.
- **Goal**: Empty parent at ~(8, 0.5, 0); child Sphere with Sphere Collider (Is Trigger ✓) and Tag = `Goal`.
- **Agent**: Capsule at ~(-8, 1, 0) with:
  - Rigidbody (Use Gravity ✓, Freeze Rotation X/Z ✓)
  - Behavior Parameters (Behavior Name = `CourseAgent`, Actions = Continuous (Size=4))
  - Decision Requester (Decision Period = 5 for training; 1 for manual testing)
  - Ray Perception Sensor 3D (Detectable Tags = `Ground`, `Goal`, `Hazard`)
  - `CourseAgent` (Script) attached and goal reference assigned to the Goal object

### Optional Camera

- Main Camera has `FollowOrbitCamera` attached; drag the Agent into target.
- Hold right mouse to orbit (or set `holdRightMouseToOrbit = false` in the component).

---

## 2) Python / Trainer Setup

_Do this once per machine._

### Windows (PowerShell or Git Bash)

```bash
python -m venv .venv

# PowerShell:
. .venv/Scripts/Activate.ps1

# Git Bash:
source .venv/Scripts/activate

python -m pip install --upgrade pip
pip install mlagents torch tensorboard
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlagents torch tensorboard
```

---

## 3) Run a Quick Manual Sanity Check (Optional)

1. In Unity, select the Agent object.
2. **Behavior Parameters → Behavior Type** = `Heuristic Only`.
3. **Decision Requester → Decision Period** = `1` (more responsive).
4. Press **Play** and verify:
   - **WASD** to move (if reversed, flip `ca[1] = -Input.GetAxis("Vertical")` in Heuristic).
   - **QE** to turn, **Space** to jump.
   - Touching the goal sphere should end the episode (resets position).
     - If not, ensure the Sphere collider is Trigger and Tag = `Goal`.
5. Stop Play when done.

---

## 4) PPO Training

### Trainer Config

_(Already in the repo as `config/challenge_ppo.yaml`)_

```yaml
behaviors:
  CourseAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 16384
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.995
        strength: 1.0
    time_horizon: 128
    max_steps: 3.0e6
    summary_freq: 5000
    checkpoint_interval: 200000
```

### Start the Python Trainer

From repo root:

```bash
mlagents-learn config/challenge_ppo.yaml --run-id=course_ppo_v1 --train
```

You should see logs: `"Listening on port 5004 …"`

### Connect Unity

1. In Unity, select the Agent:
   - **Behavior Parameters → Behavior Type** = `Default`.
   - **Decision Requester → Decision Period** = `5`.
   - **Max Step** = `1000` (recommended).
2. Press **Play**. The trainer should report it connected and start stepping.

### Monitor Training

Optional: 

```bash
tensorboard --logdir results
```

Look for mean reward trending upward and more frequent goal hits.

### Stop Training

- Stop Play in Unity, then **Ctrl+C** in the terminal.

---

## 5) Use the Trained Policy in the Editor

1. Find the exported `.onnx` model in `results/<run-id>/` (or export via ML-Agents menu).
2. Copy it into `Assets/Models/` (create the folder if needed).
3. Select the Agent → **Behavior Parameters**:
   - **Behavior Type** = `Inference Only`
   - **Model** = (assign the .onnx)
4. Press **Play** to watch the trained agent.

---

## 6) Project / Repo Tips

### Recommended Unity Settings for Faster Training

- **Edit → Project Settings → Quality**: VSync Count = `Don't Sync`
- Use a lower quality preset (e.g., "Very Low")
- Hide Scene view while training; watch only Game view
- **Edit → Preferences → Script Changes While Playing**: `"Stop Playing and Recompile"` (prevents long hot-reload stalls mid-run)

---

## 7) Troubleshooting

### Trainer says "The Unity environment took too long to respond"

- Keep Play mode running (don't pause/stop).
- Behavior Type must be `Default` while training.
- Avoid saving scripts during Play (will trigger recompile pauses). If needed, set Script Changes While Playing as above.
- Reconnect by re-pressing Play, or restart `mlagents-learn`.

### "More/Fewer observations than vector observation size" warnings

- Set **Behavior Parameters → Vector Observation → Space Size** = `10`
  - _(We add 10 floats: 3 velocity + 3 normalized direction to goal + 1 distance + 3 forward.)_

### Agent doesn't reset on goal contact

- The Sphere (child) must have **Sphere Collider** (Is Trigger ✓) and Tag = `Goal`.

### WASD seems reversed

- Rotate the Agent Y = 180°, or in Heuristic invert vertical: `ca[1] = -Input.GetAxis("Vertical");`

### Agent slides forever during manual control

- We set velocity to match the desired horizontal speed each step (no perpetual sliding).
- If needed, set Rigidbody **Linear Damping** to 1–2.
