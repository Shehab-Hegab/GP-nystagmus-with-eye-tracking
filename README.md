# GP-Nystagmus Eye Tracking & Training System

[![Unity](https://img.shields.io/badge/Platform-Unity-ffb400?logo=unity)]()
[![Python](https://img.shields.io/badge/Backend-Python-blue?logo=python)]()
[![License](https://img.shields.io/github/license/Shehab-Hegab/GP-nystagmus-with-eye-tracking)]()

A real-time interactive system for **eye tracking**, **diagnosis**, and **visual rehabilitation** for patients with **nystagmus** or other oculomotor disorders.  
Combines **Unity**, **Machine Learning**, **WebSocket integration**, and **signal visualization**.

---

## 📚 Table of Contents

- [🎯 Project Goal](#-project-goal)
- [⚙️ How It Works](#️-how-it-works)
- [🎮 Game Modules](#-game-modules)
- [🧠 ML Pipeline](#-ml-pipeline)
- [🔗 Integration](#-integration)
- [📈 Data Logging & Plots](#-data-logging--plots)
- [📂 Resources](#-resources)
- [🖼️ Screenshots](#-screenshots)
- [🤝 Contribution](#-contribution)
- [📧 Contact](#-contact)

---

## 🎯 Project Goal

✅ **Step 1:** Start ML model → webcam tracks eye movement → classify **Normal** or **Patient**  
✅ **Step 2:** If Patient → play **Train Game** or **Rubik’s Cube Fixation Game** → gradually increase difficulty.  
✅ **Step 3:** All sessions saved in Excel → data feeds into CNN & LSTM → monitor improvement → next steps.

---

## ⚙️ How It Works

- Eye tracking via webcam.
- Train the model to classify: Healthy vs Patient.
- If Patient → start games → train gaze fixation.
- Games adapt: speed, color, shapes, levels.
- Data saved → CNN + LSTM → visualize signals → track progress.

---

## 🎮 Game Modules

### 🚆 Train Tracking

- Follow a moving train on screen.
- Adjustable: color, speed, cars, obstacles.
- Purpose: Improve **smooth pursuit**.

**Modes:**  
- [Full Train](https://drive.google.com/file/d/1_lrU5ZZ7M_gU2Fe3pQ_poHGJD09g1Ga6/view?usp=drive_link)  
- [Single Train](https://drive.google.com/file/d/1uQnZRPEIM_3OgR7_sidVaX8bNVZxBI8W/view?usp=drive_link)

---

### 🧊 Rubik’s Cube Gaze Fixation

- Patient focuses on preferred color on rotating Rubik’s Cube.
- Levels: starts **2×2×2**, up to **4×4×4**.
- Tracks gaze fixations under motion.

---

## 🧠 ML Pipeline

- **CNN**: Classifies eye movement (Healthy vs Patient).
- **LSTM**: Uses saved data to check improvement and next steps.
- **Signal Visualization**: See eye behavior second by second.

---

## 🔗 Integration

- **Unity** linked to Python backend via **WebSocket**.
- Unity runs games → backend handles ML → Excel logging → plots.

---

## 📈 Data Logging & Plots

- Saves:
  - Eye coordinates
  - Timestamps
  - Level, speed, mode
- Excel → Plots → Analyze signals → Diagnose → Improve training.

---

## 📂 Resources

- 📅 [Gantt & Timeline](https://docs.google.com/spreadsheets/d/1TkLX_q5t6vAm9R57QAuC5okDFoFYQ8wd57yXewiJa3M/edit?usp=sharing)
- 📋 Surveys:  
  - [Patient (Arabic)](https://docs.google.com/forms/d/e/1FAIpQLSfhU_CxO59pTOpdHfrxsELWIn-23gpyVego-ujayt2F48EqSg/viewform)  
  - [Patient (English)](https://forms.gle/nXjsG7GL7pAjySASA)  
  - [Doctor](https://docs.google.com/forms/d/e/1FAIpQLSciKyrwdtClstPlHFf0jYcR8N-ioTminI9EbtT88zsERKBKZg/viewform)

---

## 🖼️ Screenshots

### System Architecture
![System Architecture](https://github.com/user-attachments/assets/7de2ecc5-b16d-4cc5-b4e7-b312d7f127f5)

### Process Flow
![Process Diagram](https://github.com/user-attachments/assets/1502fac5-6160-49bb-bd6f-02e62fb42eb3)

### Gameplay – Train
![Full Train](https://github.com/user-attachments/assets/e88a1ea3-178a-4f93-9001-25e3d3edc743)
![Gameplay View](https://github.com/user-attachments/assets/1870d759-da74-462f-bde5-db7a476c92c5)
![Gameplay View](https://github.com/user-attachments/assets/c719bd25-d1b4-4de2-a418-7ca8258a7538)
![Gameplay View](https://github.com/user-attachments/assets/7e6c1f20-1fce-475d-9b43-01978fa4219b)
![Gameplay View](https://github.com/user-attachments/assets/64cfd325-3fb2-476a-8001-76bc5a6a9a54)
![Gameplay View](https://github.com/user-attachments/assets/2914c7b7-351a-48a4-b3b8-8300d532151f)
![Gameplay View](https://github.com/user-attachments/assets/7b1ee11d-1a4e-4cf9-b47c-f731e8e0fb66)
![Gameplay View](https://github.com/user-attachments/assets/6c5072b5-b5ce-472d-a7b0-9df170daea3d)

### Upload Mode
![Upload Videos](https://github.com/user-attachments/assets/8a77392b-bd8f-4082-93c3-9f816c7ab1f3)

### Eye Only Tracking
![Eye Only](https://github.com/user-attachments/assets/918ef336-7b7b-41e4-ac6c-f8c70c944590)
![Eye Plot](https://github.com/user-attachments/assets/61c0f74b-25c9-44e2-873d-b0f7a6afbbd5)

### Data Plots
![Data Plot 1](https://github.com/user-attachments/assets/092d6997-75dd-45d1-8052-30825cb1814d)
![Data Plot 2](https://github.com/user-attachments/assets/07f905d1-d35b-4cf5-b772-ea066cde9900)
![Data Plot 3](https://github.com/user-attachments/assets/62e0bde9-637d-4508-bf5d-7512f558b505)
![Data Plot 4](https://github.com/user-attachments/assets/b7bc01ce-d7ab-497f-91f3-d82b43809e9b)
![Data Plot 5](https://github.com/user-attachments/assets/f2e2468a-0426-4082-9cb8-f926df8252fa)
![Data Plot 6](https://github.com/user-attachments/assets/d268187b-d2a9-4113-b6fe-cd77c8ae6cec)
![Data Plot 7](https://github.com/user-attachments/assets/e1ff3bf8-60c6-4505-a5ac-28232e8ba544)
![Data Plot 8](https://github.com/user-attachments/assets/a8f77591-5252-489b-81ea-94df2618481d)
![Data Plot 9](https://github.com/user-attachments/assets/73c6e15a-ea34-4eb8-87dc-63c96ae91409)
![Data Plot 10](https://github.com/user-attachments/assets/f9205721-649c-4c1d-b96a-087f08d58ecb)

### Rubik’s Cube – Gaze Fixation
![Rubik 1](https://github.com/user-attachments/assets/866840e4-0804-474b-bd45-6c6eb1e5a392)
![Rubik 2](https://github.com/user-attachments/assets/70d847cf-04b9-4bfd-bd50-b19bfb3dcdff)
![Rubik 3](https://github.com/user-attachments/assets/87c3b1b7-f9c2-405a-8bf0-3b79293406d3)
![Rubik 4](https://github.com/user-attachments/assets/b447d3c5-0600-4cfa-990e-dc0f3b0a0ee3)
![Rubik 5](https://github.com/user-attachments/assets/73a7e3e5-22a4-4291-ac75-34caa421ecfb)
![Rubik 6](https://github.com/user-attachments/assets/fc21dd9c-8cbd-43d7-aff9-6a273a067a1b)
![Rubik 7](https://github.com/user-attachments/assets/5ccab184-62ab-46ef-8305-b72bcaedd89e)
![Rubik 8](https://github.com/user-attachments/assets/ffc51250-16c6-4213-bcc3-179cdbea6ecf)
![Rubik 9](https://github.com/user-attachments/assets/4346c7d6-7cd2-4768-bcd0-bd6c2a9f5c14)
![Rubik 10](https://github.com/user-attachments/assets/891c4d67-2739-4114-9b88-8cf4f2f270c7)
![Rubik 11](https://github.com/user-attachments/assets/e1582d77-0a94-4b24-bbe7-40387f919bb9)
![Rubik 12](https://github.com/user-attachments/assets/097c3714-1e8b-4ddf-821c-a1d22ca2c06d)
![Rubik 13](https://github.com/user-attachments/assets/147f283c-a50d-4f66-8b08-5cb2584a0a4e)

---

## 🤝 Contribution

✔️ Report bugs  
✔️ Suggest features  
✔️ Fork → Improve → Pull Request

---

## 📧 Contact

For collaboration, research, or technical support:  
📬 [GitHub Profile](https://github.com/Shehab-Hegab)  
📋 [Surveys & Forms](#-resources)

> **Graduation Project – Biomedical AI & Assistive Tech Lab**

---
