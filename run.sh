#!/bin/bash
# Run Fulcrum - Bearing Identification & ML Coating Estimates
set -e
cd "$(dirname "$0")"

echo "1. Running migration..."
python migrate_db.py

echo ""
echo "2. Assigning catalog/test split (50/50)..."
python assign_splits.py 2>/dev/null || true

echo ""
echo "3. Seeding bearing dataset (if assets exist)..."
python seed_bearing_dataset.py 2>/dev/null || echo "   (Skip - add images to assets/ and run: python seed_bearing_dataset.py)"

echo ""
echo "4. Downloading industrial images (optional, may skip if rate limited)..."
python download_industrial_images.py 2>/dev/null || echo "   (Skip - run manually later: python download_industrial_images.py)"

echo ""
echo "5. Starting app on http://localhost:5000"
python app.py
