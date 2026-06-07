# Deep Reinforcement Learning – Lab Assignment 1

# Group No: 204


**Total Marks:** 10 (5M MAB + 5M DP)

## Intended Learning Outcomes
- Understand the basic functionality of Multi-Armed Bandit (MAB) and Dynamic Programming (DP).
- Implement MAB and DP concepts.

## Prerequisites
- Lectures CS1–CS5
- Webinar demonstrations

---

# Part 1 – Multi-Armed Bandit (5 Marks)

## Title
Adaptive Treatment Recommendation System using Multi-Armed Bandit Learning

## Problem Scenario
Develop a recommendation system for selecting the best medicine in a clinical trial setting. Each medicine is an arm in a Multi-Armed Bandit problem.

### Dataset Creation Rules

Let group number = **G**

```python
random.seed(G)
numpy.random.seed(G)
```

### Number of Medicines

```text
K = (G mod 3) + 5
```

### Hidden Success Probability

For medicine i:

```text
Pi = 0.4 + (((G + i) mod 6) × 0.07)
```

### Patient Severity

Generate exactly 1000 patients.

```text
Severity = (patient_id mod 5) + 1
```

Severity values range from 1–5.

### Clinical Outcome

- Recovered = 1 with probability Pi
- Not Recovered = 0 with probability (1 − Pi)

### Utility Score

```text
UtilityScore = clinical_outcome × (1 − Severity/10)
```

Examples:

- Severity = 1 and recovered → 0.9
- Severity = 5 and recovered → 0.5
- Not recovered → 0

### Dataset Schema

| Column | Description |
|----------|----------|
| patient_id | Patient index (0–999) |
| severity_score | Disease severity (1–5) |

During execution:

| Column | Description |
|----------|----------|
| assigned_medicine | Selected medicine |
| clinical_outcome | Recovery (0/1) |
| utility_score | Final reward |

### Important
- Use `clinical_outcome` to update bandit statistics.
- Use `utility_score` to compute cumulative reward.

## Tasks

### Task 1 – Dataset Design (1 Mark)

1. Generate synthetic environment.
2. Display:
   - Group number G
   - Total medicines K
   - Hidden success probabilities
3. Print first 10 rows.

### Task 2 – Immediate Exploitation Strategy (1 Mark)

Policy:

> Once a treatment appears best, continue prescribing only that treatment.

Requirements:

- Test each medicine exactly 10 times initially.
- Continue using the currently best-performing medicine.
- Run 1000 iterations.
- Evaluate cumulative reward.

### Task 3 – Controlled Clinical Trial Strategy (1.5 Marks)

Policy:

> Mostly use the best treatment but occasionally explore alternatives.

Requirements:

- 10% exploration.
- Run for 1000 patients.
- Evaluate cumulative reward.
- Analyze exploration rates:
  - 1%
  - 50%

### Task 4 – Confidence-Based Strategy (1 Mark)

Implement:

- UCB1 Algorithm

Evaluate cumulative reward.

### Task 5 – Comparative Analysis (0.5 Mark)

Generate:

- Cumulative Reward vs Number of Patients graph.

Answer:

1. Highest cumulative reward?
2. Fastest convergence?
3. Most stable performance?
4. Best strategy for real deployment?

Provide a 3–5 sentence summary.

---

# Part 2 – Dynamic Programming (5 Marks)

## Title

Autonomous Drone Rescue Using Dynamic Programming

## Objective

Implement a Drone Rescue environment and solve it using:

- Value Iteration **or**
- Policy Iteration

## Scenario

A rescue drone operates in a disaster-hit city represented as a grid world.

The drone must:

- Rescue civilians
- Avoid danger zones
- Manage battery
- Use charging stations

### Grid Size

Student ID ending:

- 0–4 → 5×5 grid
- 5–9 → 6×6 grid

### Environment Configuration

#### If ID ends with 0–4

- 2 rescue targets
- 1 charging station
- 3 danger zones
- 2 blocked cells

#### If ID ends with 5–9

- 3 rescue targets
- 2 charging stations
- 4 danger zones
- 3 blocked cells

### Cell Types

| Symbol | Meaning |
|----------|----------|
| S | Start |
| F | Free Cell |
| D | Danger Zone |
| R | Rescue Target |
| C | Charging Station |
| W | Wind Zone |
| X | Blocked Cell |

### Drone Behaviour

- Start at top-left corner (S)
- Limited battery
- Battery decreases every move
- Rescue yields rewards

### Actions

- Up
- Down
- Left
- Right
- Hover

### Hover Action

At Charging Station:

- Battery +2
- Cannot exceed maximum capacity

Else:

- Consumes 1 battery

---

## Environment Rules

### Battery

Every action consumes 1 battery.

Maximum battery:

- Even student ID → 10
- Odd student ID → 15

Battery = 0 → Episode ends.

### Charging Station

Entering C:

- Battery refills to full.

### Rescue Target

Entering R:

- Reward obtained.
- Target removed.
- Cell becomes F.

### Danger Zone

Entering D:

- Reward = −10
- Episode continues.

### Wind Zone

If current cell is W:

Movement may change randomly.

Wind probability:

- 20% for IDs ending 0–4
- 30% for IDs ending 5–9

### Blocked Cell

If movement enters X:

- Drone stays in place.
- Battery still decreases.

---

## Episode Termination

Episode ends when:

- Battery reaches 0
- All rescue targets rescued
- Step limit exceeded

Step limits:

- 50 (5×5)
- 75 (6×6)

---

## MDP Design

### State Space

State must include:

- Drone position
- Battery level
- Rescue target status

### Action Space

Actions:

- Up
- Down
- Left
- Right
- Hover

Implement valid action function.

### Rewards

| Event | Reward |
|---------|---------|
| Rescue Target | +20 |
| Enter Danger Zone | -10 |
| Battery Exhausted | -20 |
| Reach Charging Station | +5 |
| Normal Movement | -1 |

---

## Expected Outcomes

### 1. Custom Environment (1 Mark)

Implement:

- reset()
- step(action)
- render()

Must handle:

- Battery updates
- Rescue target removal
- Wind stochasticity
- Charging stations
- Obstacles
- Reward computation

### 2. Dynamic Programming Solution (2 Marks)

Using Value Iteration or Policy Iteration:

1. Enumerate reachable states.
2. Compute:
   - Optimal Value Function V*(s)
   - Optimal Policy π*(s)

Stopping threshold:

```text
θ = 10^-3
```

Show:

- Convergence iterations
- Runtime
- Final delta/error

### 3. Policy Visualization (1 Mark)

Visualize:

- Movement directions
- Rescue sequence
- Charging behavior
- Danger avoidance

Possible methods:

- Arrows
- Heatmaps
- Grid overlays
- Animation

### 4. State-Value Analysis (1 Mark)

Choose a state-space slice.

Example:

- Fixed battery level
- Fixed rescue status
- Vary drone position

Plot heatmap of V*(s) and explain patterns.

### 5. DP Scalability Discussion (1 Mark)

Discuss:

- Curse of dimensionality
- Effect of larger grids
- Additional rescue targets
- Dynamic weather

Explain:

- Why DP becomes difficult
- How Deep RL can help
- Real-world relevance

---

## Submission Notes

- Submit PDF only.
- Include code, outputs, and iteration details.
- Include comments in code.
- Execute in virtual lab.
- Attach screenshots with timestamps.
- Print VM ID and execution timestamp at the top of notebook.
