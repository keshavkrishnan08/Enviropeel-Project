# Enviropeel Project

This project helps someone upload an industrial component photo and get two simple results:

- a best-match identification against saved reference images
- an estimated size and coating estimate based on the image

It runs as a small local website on your computer.

## What You See

When the app is running, there are two main pages:

- `/` for image identification
- `/estimate` for a fuller estimate with sizing, coating recommendation, and similar past examples

You upload a photo, and the app compares it to the saved catalog of images already included in the project.

## What You Need

Before running the project, make sure your computer has:

- Python 3.12 or close to it
- internet access the first time, so the AI models can download

## How To Run

Open Terminal in this project folder and run:

```bash
pip install -r requirements.txt
python migrate_db.py
python assign_splits.py
python seed_bearing_dataset.py
python sync_catalog_metadata.py
python app.py
```

Then open this in your browser:

`http://localhost:5000`

## What Each Step Does

- `pip install -r requirements.txt`: installs the Python packages the app needs
- `python migrate_db.py`: prepares the local database
- `python assign_splits.py`: organizes stored images into catalog and test groups
- `python seed_bearing_dataset.py`: loads the included sample bearing images
- `python sync_catalog_metadata.py`: makes sure the saved image labels are clean and consistent
- `python app.py`: starts the website

## Optional Step

If you want more reference images from Wikimedia Commons, run:

```bash
python download_industrial_images.py
```

This is optional. The project can already run without it.

## Recommended Hosting (Render)

This project runs best as a normal Python web service (not serverless) because it uses a heavy ML model.

### Why Render

- easiest way to host this exact project
- no need to split frontend and backend
- gives you one shareable link for your boss

### One-time setup

1. Push this repository to GitHub.
2. Create a Render account and choose **New Web Service**.
3. Connect your GitHub repo.
4. Render will detect `render.yaml` automatically.
5. Deploy and wait until status is **Live**.
6. Open the Render URL and share it.

### Notes for non-technical use

- First request can be slower because the model loads.
- The app URL works from phone or desktop browser.
- If you edit the app and push to GitHub, Render redeploys automatically.

## Folder Guide

- `app.py`: starts the website
- `similarity.py`: finds the most similar saved images
- `estimate.py`: produces sizing and coating estimates
- `database.py`: stores image records and metadata
- `assets/`: bundled source images
- `reference_images/`: saved catalog and downloaded reference images
- `static/`: website pages

## If Something Fails

Try these first:

```bash
python test_all.py
```

If the app still does not start:

- check that Python is installed
- check that `pip install -r requirements.txt` finished successfully
- make sure port `5000` is not already in use
- re-run `python sync_catalog_metadata.py`

## Notes

- The app uses the local image catalog to identify parts.
- The sizing and coating outputs are estimates, not final engineering measurements.
- Uploaded images are stored locally in the `uploads/` folder while you use the app.
- On Vercel, uploaded files are temporary and may disappear between requests or deployments.
