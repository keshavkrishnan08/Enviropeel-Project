#!/usr/bin/env python3
"""
Comprehensive test suite for Fulcrum - runs all tests in one process
(CLIP loads once, faster).
"""
import os
import sys

def run_tests():
    print("=" * 60)
    print("FULCRUM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    failures = 0

    print("\n[1] DATABASE & MIGRATION")
    try:
        from flask import Flask
        from database import db, ImageRecord, init_db
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)
        with app.app_context():
            init_db()
            total = ImageRecord.query.count()
            catalog = ImageRecord.query.filter(
                (ImageRecord.split == 'catalog') | (ImageRecord.split.is_(None))
            ).count()
            test_s = ImageRecord.query.filter(ImageRecord.split == 'test').count()
        print(f"  ✓ DB OK: {total} total, {catalog} catalog, {test_s} test")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[2] FLASK APP & ROUTES")
    try:
        from app import app, matcher
        with app.app_context():
            matcher.load_reference_images()
        print(f"  ✓ Loaded {len(matcher.reference_images)} catalog refs for matching")
        with app.test_client() as c:
            r1 = c.get('/')
            r2 = c.get('/estimate')
            assert r1.status_code == 200
            assert r2.status_code == 200
        print("  ✓ GET / : 200")
        print("  ✓ GET /estimate : 200")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[3] API ENDPOINTS (read)")
    try:
        from app import app
        with app.test_client() as c:
            r1 = c.get('/api/reference/list')
            r2 = c.get('/api/coatings/list')
            r3 = c.get('/api/bearing-types')
            d1 = r1.get_json()
            d2 = r2.get_json()
            d3 = r3.get_json()
        print(f"  ✓ /api/reference/list: {len(d1.get('images', []))} images")
        print(f"  ✓ /api/coatings/list: {len(d2.get('images', []))} catalog")
        print(f"  ✓ /api/bearing-types: {d3.get('bearing_types')}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[4] API UPLOAD (Identify)")
    try:
        from app import app
        test_img = 'reference_images/coatings/bearing_pillow_block_IMG-20211001-WA0000-.png'
        if not os.path.exists(test_img):
            test_img = 'reference_images/downloads/Bearing_01.JPG.jpg'
        if not os.path.exists(test_img):
            print("  ⚠ No test image, skip")
        else:
            with app.test_client() as c:
                with open(test_img, 'rb') as f:
                    r = c.post('/api/upload', data={'image': (f, 'test.png')},
                               content_type='multipart/form-data')
                d = r.get_json()
                matches = d.get('matches', [])
            if r.status_code != 200:
                raise Exception(d.get('error', r.status_code))
            print(f"  ✓ POST /api/upload: 200")
            print(f"  ✓ Matches: {len(matches)}")
            if matches:
                m = matches[0]
                print(f"  ✓ Top: {m.get('bearing_type')} @ {m.get('similarity_score', 0)*100:.1f}%")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[5] API ESTIMATE")
    try:
        from app import app
        test_img = 'reference_images/coatings/bearing_pillow_block_IMG-20211001-WA0000-.png'
        if not os.path.exists(test_img):
            test_img = 'reference_images/downloads/Bearing_01.JPG.jpg'
        if not os.path.exists(test_img):
            print("  ⚠ No test image, skip")
        else:
            with app.test_client() as c:
                with open(test_img, 'rb') as f:
                    r = c.post('/api/estimate', data={'image': (f, 'test.png')},
                               content_type='multipart/form-data')
                d = r.get_json()
            if r.status_code != 200:
                raise Exception(d.get('error', r.status_code))
            print(f"  ✓ POST /api/estimate: 200")
            print(f"  ✓ component_type: {d.get('component_type')}")
            print(f"  ✓ surface_area: {d.get('surface_area_sqft')} sq ft")
            print(f"  ✓ total cost: ${d.get('costs', {}).get('total', 0):.2f}")
            pw = d.get('previous_work', [])
            print(f"  ✓ previous_work: {len(pw)} matches")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[6] ML ESTIMATE PIPELINE")
    try:
        from estimate import get_coating_estimate
        test_img = 'reference_images/coatings/bearing_pillow_block_IMG-20211001-WA0000-.png'
        if not os.path.exists(test_img):
            test_img = 'reference_images/downloads/Bearing_01.JPG.jpg'
        if os.path.exists(test_img):
            r = get_coating_estimate(test_img)
            assert 'costs' in r and r['costs']['total'] > 0
            print(f"  ✓ get_coating_estimate: ${r['costs']['total']:.2f}")
        else:
            print("  ⚠ No test image, skip")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[7] SEED SCRIPT (import)")
    try:
        import seed_bearing_dataset
        print("  ✓ seed_bearing_dataset imports OK")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n[8] DOWNLOAD SCRIPT (import)")
    try:
        import download_industrial_images
        print("  ✓ download_industrial_images imports OK")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"{failures} TEST(S) FAILED ✗")
    print("=" * 60)
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(run_tests())
