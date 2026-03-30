import os
from database import db, ImageRecord
from flask import Flask

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def main():
    from caption import get_caption
    print("Loading caption model...")
    with app.app_context():
        updated = 0
        for rec in ImageRecord.query.all():
            if not os.path.exists(rec.filepath):
                continue
            desc = get_caption(rec.filepath)
            if desc and desc != rec.description:
                rec.description = desc
                updated += 1
                print(rec.filename, "->", desc[:50])
        db.session.commit()
        print("Updated", updated, "labels.")


if __name__ == "__main__":
    main()
