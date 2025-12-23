# Deployment Instructions for Render

This folder contains all the necessary files to deploy your Risk Calculator to Render.

## Prerequisites (Missing Files)

Due to file access restrictions, I could not copy the following files. **You must manually copy them into this folder before deploying:**

1.  **`scaler.pkl`**: Copy this file from your project root to this `deployment_package` folder.
2.  **`data.csv`**: Copy your standardized training data (originally named `训练集_标准化后.csv`) to this folder and **rename it to `data.csv`**.

## How to Deploy to Render

1.  **Upload to GitHub**:
    *   Create a new repository on GitHub.
    *   Upload the contents of this `deployment_package` folder (including the manually copied files) to the repository.

2.  **Deploy on Render**:
    *   Go to [render.com](https://render.com) and sign up/login.
    *   Click **New +** -> **Web Service**.
    *   Connect your GitHub account and select the repository you just created.
    *   **Name**: Give your service a name (e.g., `ckd-risk-calculator`).
    *   **Runtime**: Select **Python 3**.
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn app:app`
    *   Click **Create Web Service**.

3.  **Wait for Deployment**:
    *   Render will install dependencies and start your app.
    *   Once finished, you will see a URL like `https://ckd-risk-calculator.onrender.com`.

## Notes

*   **Fonts**: The code has been modified to use default fonts if 'Times New Roman' is not available on the server.
*   **Memory**: The application uses a large dataset for SHAP explanations. If the free tier of Render runs out of memory, consider reducing the number of background samples in `app.py` (search for `shap.kmeans`).
