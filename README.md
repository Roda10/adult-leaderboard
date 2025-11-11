# Adult Income Competition Leaderboard

A mini Kaggle-style competition platform for predicting adult income using Streamlit.

## Features
- Submit predictions via CSV uploads
- Leaderboard with live rankings
- ROC AUC and Log Loss scoring
- Admin controls for competition management

## Local Development

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Roda10/adult-leaderboard.git
cd adult-leaderboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export LEADERBOARD_ADMIN="your_admin_password"
export LEADERBOARD_DB="leaderboard.db"
export GROUND_CSV="ground.csv"
export PUBLIC_IDS_CSV="public_ids.csv"
export SUBMISSIONS_DIR="submissions"
export MAX_SUBS_PER_DAY="5"
export MAX_FILE_SIZE_MB="10"
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "Deploy an app"
4. Select your GitHub repository
5. Select the branch and file path (`project_2/app.py`)
6. Configure secrets in Streamlit Cloud settings if needed
7. Click "Deploy"

## Project Structure
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `ground.csv` - Ground truth labels for validation
- `public_ids.csv` - Public test set IDs
- `submissions/` - Directory for team submissions
