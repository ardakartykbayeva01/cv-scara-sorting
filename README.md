# Vision-Based Colour Sorting with a SCARA Robot (CoppeliaSim)

Final project for **PED 744 Advanced Robotics and Modeling** (SDU).
Topic: **Computer Vision in Robotics**.

A SCARA robot sorts red, green and blue cubes into colour zones. An overhead camera
finds the cubes (HSV segmentation), a homography converts pixels to robot coordinates,
and our own inverse kinematics with quintic trajectories moves the robot.
The whole scene is built from code (`simulation/build_scene.py`) through the
CoppeliaSim ZeroMQ Remote API.

```
camera frame -> HSV segmentation -> pixel centre -> homography -> (x, y) -> IK -> pick & place
```

## Team

| Member | Role | Main folders |
|---|---|---|
| Ardak Artykbayeva | Student 1: robotics & control | `simulation/`, `control/` |
| ansxxsar | Student 2: computer vision & calibration | `vision/` |

## Repository structure

```
simulation/   build_scene.py (builds the scene), test_robot.py (FK + camera check), cv_scara_scene.ttt
control/      kinematics.py, trajectory.py, robot_control.py,
              test_pick_place.py, run_on_play.py, run_experiments.py (E1, E2), run_e3.py (E3)
vision/       config.py, detect.py, camera.py, calibration.py, calibrate.py, demo.py, calibration.json
tests/        unit tests (kinematics, detection, homography)
results/      CSV data, summaries and figures of the experiments
docs/         report and slides
```

## Setup

- CoppeliaSim Edu 4.10, Python 3.12
- `pip3 install -r requirements.txt`

## How to run

All commands are run in the Terminal (not in the CoppeliaSim console),
with CoppeliaSim open and the simulation stopped.

```bash
# 1. Build the scene (once) and check the robot and camera
cd simulation && python3 build_scene.py && python3 test_robot.py

# 2. Calibrate the camera (once)
cd ../vision && python3 calibrate.py

# 3. Live demo: press PLAY in CoppeliaSim and the robot sorts the cubes
cd ../control && python3 run_on_play.py

# 4. Experiments
python3 run_experiments.py      # E1 + E2
python3 run_e3.py               # E3

# 5. Unit tests (no CoppeliaSim needed)
cd .. && python3 -m unittest discover tests
```

## Results

**Calibration** (25 grid points, 20 new validation points): mean error 0.58 mm, max 1.00 mm.

**E1: 30 random layouts, 90 cubes, vision-based sorting**

| Metric | Result |
|---|---|
| Detection recall | 100% |
| Localization error, mean / max | 0.53 / 1.19 mm |
| Grasp success | 100% |
| Sorting success | 100% |
| Placement error, mean | 0.53 mm |
| Cycle time per cube | 16.17 ± 1.78 s |

**E2: Gaussian image noise (no filtering).** Recall is 100% up to σ = 40, then falls
(81% at σ = 50, 62% at σ = 60, 18% at σ = 80, 0% at σ = 100). Precision stays near 100%:
the system misses cubes rather than picking wrong ones.

**E3: Gaussian blur before HSV thresholding.** A 5×5 blur keeps recall at 100% up to
σ = 150 (mean localization error 1.95 mm), at the cost of +0.11 mm error on clean images.

| σ | no blur recall | 5×5 blur recall |
|---|---|---|
| 60 | 60.0% | 100% |
| 100 | 0% | 100% |
| 150 | 0% | 100% |

Figures: `results/e1_results.png`, `results/e2_noise.png`, `results/e3_blur.png`,
`results/e3_example_sigma100.png`.

## Limitations

- Simulation only: ideal lighting, matte cubes, synthetic Gaussian noise.
- The suction gripper is modelled by parenting the cube to the tool (no contact physics).
- The homography assumes all cubes lie on one plane (the mat) and have the same size.

## Use of AI tools

AI assistance (Claude) was used for parts of the code and text, in line with the course policy.
All code was run, checked and explained by the team.
