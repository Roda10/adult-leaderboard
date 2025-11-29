# 🏆 Mini-Kaggle: Adult Income Leaderboard

A **Streamlit web app** for a *mini-competition* based on the [Adult Income dataset (UCI / OpenML)](https://www.openml.org/d/1590).
Students build machine-learning models to predict whether an individual earns **more than $50K per year**, submit their results, and get instant feedback through a **live leaderboard**.

---

## 🎯 Objective

The project simulates a small-scale Kaggle competition for educational use.
Participants:

* Train models on `train.csv`
* Generate probability predictions for `test.csv`
* Upload `submission.csv` (`id,income_prob`) and see their scores ranked by **ROC-AUC** (and Log Loss as tiebreaker)

---

## ⚙️ Features

| Feature                   | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| **Streamlit Leaderboard** | Automatic scoring (ROC-AUC / Log-Loss) with upload form                    |
| **Admin Mode**            | Secure upload of ground truth (`ground.csv`) and optional `public_ids.csv` |
| **Minimal UI**            | Clean leaderboard: Rank • Team • Public AUC • Private AUC • Overall AUC    |
| **Archiving**             | All submissions are automatically stored with timestamps                   |
| **SQLite persistence**    | Scores saved locally between restarts                                      |
| **Public/Private split**  | Optional public leaderboard vs hidden final scores                         |

---

## 🧠 Educational Context

This project was designed as part of an **advanced machine-learning practical**.
It allows students to:

* Experiment with various classifiers (Decision Trees, Random Forest, AdaBoost, Gradient Boosting, SVM, etc.)
* Learn model evaluation (AUC, Log-Loss)
* Understand fair validation and leaderboard protocols
* Experience the workflow of real Kaggle competitions in a controlled environment

---

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/<your-username>/adult-leaderboard.git
cd adult-leaderboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Default structure

```
adult-leaderboard/
├── app.py                 # Streamlit leaderboard
├── prepare_adult_competition.py  # Script to generate data splits
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🔐 Admin Setup

1. Set an environment variable for the admin password:

   ```bash
   export LEADERBOARD_ADMIN="your_password_here"
   ```
2. Run the app and open it in the browser.
3. In the sidebar → check **Mode administrateur** → enter the password.
4. Upload your private files:

   * `ground.csv` → contains `id,income` (hidden truth)
   * (optional) `public_ids.csv` → contains only `id` for public leaderboard split.

---

## 📊 Expected File Formats

| File             | Purpose                   | Columns                 |
| ---------------- | ------------------------- | ----------------------- |
| `train.csv`      | Training set              | All features + `income` |
| `test.csv`       | Test set (no label)       | All features + `id`     |
| `submission.csv` | Student submission        | `id,income_prob`        |
| `ground.csv`     | Hidden truth (admin only) | `id,income`             |
| `public_ids.csv` | Optional                  | `id`                    |

---

## 🧩 Dataset Preparation

To generate the dataset splits locally:

```bash
python prepare_adult_competition.py --test_size 0.2 --seed 42 --make_public_ids --public_frac 0.5
```

This creates:

* `train.csv`
* `test.csv`
* `ground.csv`
* `sample_submission.csv`
* `public_ids.csv` (optional)

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App** → choose:

   * Repo: `<your-username>/adult-leaderboard`
   * Branch: `main`
   * File: `app.py`
3. In **Settings → Secrets**, add:

   ```toml
   LEADERBOARD_ADMIN="your_super_secret_password"
   ```

Your leaderboard will be live within seconds. 🎉

---

## 🧑‍💻 Author

**Rodéo Oswald Y. TOHA**  
Computer Vision Researcher | Data Scientist | Educator

> I’m passionate about going beyond formulas and building unforgettable intuition.
---

If you’d like to discuss research directions in **3D Computer Vision**, **Generative Modeling**, or **Probabilistic perspective of Machine Learning**, feel free to reach out.

📩 [rodeooswald@gmail.com](mailto:rodeooswald@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/rodeooswald/) • [GitHub](https://github.com/Roda10)
