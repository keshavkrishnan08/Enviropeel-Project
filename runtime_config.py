import os
import shutil


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
TEMP_ROOT = os.path.join("/tmp", "enviropeel")


def _project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def get_reference_folder():
    return _project_path("reference_images")


def get_static_folder():
    return _project_path("static")


def get_upload_folder():
    if IS_VERCEL:
        path = os.path.join(TEMP_ROOT, "uploads")
    else:
        path = _project_path("uploads")
    os.makedirs(path, exist_ok=True)
    return path


def get_source_database_path():
    candidates = [
        _project_path("image_database.db"),
        _project_path("instance", "image_database.db"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def get_runtime_database_path():
    source_path = get_source_database_path()
    if not IS_VERCEL:
        return source_path

    os.makedirs(TEMP_ROOT, exist_ok=True)
    runtime_path = os.path.join(TEMP_ROOT, "image_database.db")
    if os.path.exists(source_path):
        needs_copy = (
            not os.path.exists(runtime_path)
            or os.path.getmtime(source_path) > os.path.getmtime(runtime_path)
        )
        if needs_copy:
            shutil.copy2(source_path, runtime_path)
    return runtime_path


def get_database_uri():
    return "sqlite:///" + get_runtime_database_path()


def should_persist_runtime_uploads():
    return not IS_VERCEL
