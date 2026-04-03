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

## Vercel Hosting

This project can also be deployed to Vercel with the current interface.

What works on Vercel:

- the current identification page
- the current estimate page
- reference image browsing
- image upload for analysis during a session

What is different on Vercel:

- uploaded files are stored temporarily
- adding new reference images from the admin form is disabled on hosted deployments

To deploy:

1. Push this repository to GitHub.
2. Import the repository into Vercel.
3. Keep the default Python runtime.
4. Deploy.

The app is configured to use the bundled catalog database and temporary upload storage in a serverless environment.

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
