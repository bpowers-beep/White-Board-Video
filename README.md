# Powers Video Creation App

A Streamlit app that converts uploaded sketch images and narration into a whiteboard-style video.

## Deployment

This app can be deployed to Streamlit Cloud.

### Steps

1. Push this repository to GitHub.
2. Sign in to https://streamlit.io/cloud.
3. Click **New app**.
4. Connect your GitHub account.
5. Select the `bpowers-beep/Whiteboard-Studio-for-Streamlit` repository.
6. Set the main file to `app.py`.
7. Deploy.

## Requirements

- Python 3.11+
- `requirements.txt` includes all dependencies.

## Notes

- The app uses `edge-tts` for AI voice narration.
- Custom audio uploads are supported.
- Background music is loaded from `assets/music/`.
