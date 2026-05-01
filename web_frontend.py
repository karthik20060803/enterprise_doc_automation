
import base64
import datetime as dt
import io
import os
import pickle
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, g
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

from evaluate_accuracy import ALL_TEXT_CATEGORIES, CORE_TEXT_CATEGORIES, aggregate



app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB upload cap
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = os.environ.get("APP_SECRET_KEY", "supersecretkey123")
APP_VERSION = "3.0.0"
MODEL_DIR = Path("models")
MODEL_READER_PATH = MODEL_DIR / "easyocr_reader.pkl"
MODEL_PIPELINE_CONFIG_PATH = MODEL_DIR / "pipeline_config.pkl"
MODEL_PROCESSING_RESULTS_PATH = MODEL_DIR / "processing_results.pkl"

# --- SQLITE USER DB ---
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db

def init_user_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()
# --- LOGIN/SIGNUP/LOGOUT ROUTES ---
def is_logged_in():
    return session.get("user") is not None

@app.route("/signup", methods=["GET", "POST"])
def signup():
    init_user_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("signup.html", error="Username and password required.")
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
            db.commit()
            return render_template("signup.html", success="Account created! Please log in.")
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Username already exists.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    init_user_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")
    if is_logged_in():
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- PROTECT MAIN ROUTES ---
from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

SUBJECT_KEYWORDS = {
    "ENGLISH",
    "HINDI",
    "MATHEMATICS",
    "PHYSICS",
    "CHEMISTRY",
    "BIOLOGY",
    "COMPUTER",
    "ECONOMICS",
    "HISTORY",
    "GEOGRAPHY",
}

USELESS_STANDALONE_TOKENS = {"DAUGHTER", "SON", "SMT", "SHRI", "OF"}

KEY_FIELD_HINTS = {
    "NAME",
    "UNIQUE ID",
    "ROLL",
    "NO",
    "GRADE",
    "RESULT",
    "PERCENTAGE",
    "SUBJECT",
    "STATEMENT OF MARKS",
}


@lru_cache(maxsize=1)
def _load_pickle_artifacts() -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "reader": None,
        "pipeline_config": {},
        "processing_results": {},
        "errors": [],
    }

    if MODEL_READER_PATH.exists():
        try:
            with MODEL_READER_PATH.open("rb") as handle:
                loaded_reader = pickle.load(handle)
            if isinstance(loaded_reader, easyocr.Reader):
                artifacts["reader"] = loaded_reader
            else:
                artifacts["errors"].append(
                    f"{MODEL_READER_PATH.name} is type {type(loaded_reader).__name__}, expected easyocr.Reader."
                )
        except Exception as exc:
            artifacts["errors"].append(f"Failed loading {MODEL_READER_PATH.name}: {exc}")
    else:
        artifacts["errors"].append(f"Missing {MODEL_READER_PATH.name}")

    if MODEL_PIPELINE_CONFIG_PATH.exists():
        try:
            with MODEL_PIPELINE_CONFIG_PATH.open("rb") as handle:
                loaded_config = pickle.load(handle)
            if isinstance(loaded_config, dict):
                artifacts["pipeline_config"] = loaded_config
            else:
                artifacts["errors"].append(
                    f"{MODEL_PIPELINE_CONFIG_PATH.name} is type {type(loaded_config).__name__}, expected dict."
                )
        except Exception as exc:
            artifacts["errors"].append(f"Failed loading {MODEL_PIPELINE_CONFIG_PATH.name}: {exc}")
    else:
        artifacts["errors"].append(f"Missing {MODEL_PIPELINE_CONFIG_PATH.name}")

    if MODEL_PROCESSING_RESULTS_PATH.exists():
        try:
            with MODEL_PROCESSING_RESULTS_PATH.open("rb") as handle:
                loaded_results = pickle.load(handle)
            if isinstance(loaded_results, dict):
                artifacts["processing_results"] = loaded_results
            else:
                artifacts["errors"].append(
                    f"{MODEL_PROCESSING_RESULTS_PATH.name} is type {type(loaded_results).__name__}, expected dict."
                )
        except Exception as exc:
            artifacts["errors"].append(f"Failed loading {MODEL_PROCESSING_RESULTS_PATH.name}: {exc}")
    else:
        artifacts["errors"].append(f"Missing {MODEL_PROCESSING_RESULTS_PATH.name}")

    return artifacts


def _model_health() -> Dict[str, Any]:
    artifacts = _load_pickle_artifacts()
    return {
        "reader_from_pkl": isinstance(artifacts.get("reader"), easyocr.Reader),
        "pipeline_config_loaded": bool(artifacts.get("pipeline_config")),
        "processing_results_loaded": bool(artifacts.get("processing_results")),
        "errors": artifacts.get("errors", []),
    }


def _resolve_prediction_reader(use_gpu: bool) -> Tuple[easyocr.Reader, str]:
    artifacts = _load_pickle_artifacts()
    pkl_reader = artifacts.get("reader")
    if isinstance(pkl_reader, easyocr.Reader):
        return pkl_reader, MODEL_READER_PATH.name
    return _get_reader("en", use_gpu), "runtime-easyocr"


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, min_value), max_value)


def _to_numpy_rgb(file_storage) -> np.ndarray:
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)


def _resize_if_needed(image_rgb: np.ndarray, max_side: int = 2200) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_rgb
    scale = max_side / float(longest)
    new_width = max(int(width * scale), 1)
    new_height = max(int(height * scale), 1)
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def _preprocess_for_ocr(image_rgb: np.ndarray, mode: str = "enhanced") -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "enhanced":
        # Aggressive denoising and contrast enhancement
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
    else:
        # Original pipeline
        gray = cv2.bilateralFilter(gray, 7, 40, 40)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
    return processed


def _preprocess_for_ocr_otsu(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


@lru_cache(maxsize=4)
def _get_reader(language_key: str, use_gpu: bool) -> easyocr.Reader:
    languages = [lang for lang in language_key.split(",") if lang]
    return easyocr.Reader(languages, gpu=use_gpu, verbose=False)


def _polygon_to_bbox(points: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = max(max(xs), 0), max(max(ys), 0)
    return x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _rows_from_easyocr_results(results, source_tag: str) -> List[Dict]:
    rows: List[Dict] = []
    for item in results:
        if len(item) < 3:
            continue
        bbox_points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if not text:
            continue
        x, y, width, height = _polygon_to_bbox(bbox_points)
        rows.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "ocr_text": text,
                "ocr_confidence": confidence,
                "source": source_tag,
            }
        )
    return rows


def _merge_ocr_rows(rows: List[Dict], iou_threshold: float = 0.55) -> List[Dict]:
    merged: List[Dict] = []
    for row in sorted(rows, key=lambda item: item["ocr_confidence"], reverse=True):
        current_box = (row["x"], row["y"], row["width"], row["height"])
        duplicate_idx = -1
        for idx, existing in enumerate(merged):
            existing_box = (
                existing["x"],
                existing["y"],
                existing["width"],
                existing["height"],
            )
            if _bbox_iou(current_box, existing_box) >= iou_threshold:
                duplicate_idx = idx
                break

        if duplicate_idx == -1:
            merged.append(row)
            continue

        existing = merged[duplicate_idx]
        choose_new = False
        if row["ocr_confidence"] > existing["ocr_confidence"] + 0.03:
            choose_new = True
        elif (
            abs(row["ocr_confidence"] - existing["ocr_confidence"]) <= 0.03
            and len(row["ocr_text"]) > len(existing["ocr_text"])
        ):
            choose_new = True

        if choose_new:
            merged[duplicate_idx] = row
    return merged


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.upper()).strip()
    return cleaned


def _is_useful_text(text: str, confidence: float) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    tokens = set(normalized.split())
    if tokens and tokens.issubset(USELESS_STANDALONE_TOKENS):
        return False
    if confidence < 0.28:
        return False
    if any(hint in normalized for hint in KEY_FIELD_HINTS):
        return True
    if any(keyword in normalized for keyword in SUBJECT_KEYWORDS):
        return True
    if re.search(r"\d{2,}", normalized):
        return True
    return len(normalized.split()) >= 2 and len(normalized) <= 40


def _group_rows_into_lines(dataframe: pd.DataFrame, y_tolerance: int = 16) -> List[List[Dict]]:
    lines: List[List[Dict]] = []
    if dataframe.empty:
        return lines

    for _, row in dataframe.sort_values(["y", "x"]).iterrows():
        placed = False
        for line in lines:
            line_y = np.mean([item["y"] for item in line])
            if abs(float(row["y"]) - line_y) <= y_tolerance:
                line.append(row.to_dict())
                placed = True
                break
        if not placed:
            lines.append([row.to_dict()])

    for line in lines:
        line.sort(key=lambda item: item["x"])
    return lines


def _extract_marksheet_fields(dataframe: pd.DataFrame) -> Dict:
    fields: Dict[str, object] = {}
    lines = _group_rows_into_lines(dataframe)
    line_texts = [" ".join(item["ocr_text"] for item in line).strip() for line in lines]
    upper_lines = [_normalize_text(text) for text in line_texts]

    for line in upper_lines:
        if "STATEMENT OF MARKS" in line:
            fields["document_type"] = "Statement of Marks"

        if "RESULT" in line and "PASS" in line:
            fields["result"] = "PASS"

        if "GRADE" in line:
            grade_match = re.search(r"\bGRADE\b[:\-\s]*([A-E][1-2]?)\b", line)
            if grade_match:
                fields["grade"] = grade_match.group(1)

        class_match = re.search(r"\bCLASS\b[:\-\s]*([XIV0-9]+)\b", line)
        if class_match:
            fields["class"] = class_match.group(1)

        year_match = re.search(r"\bYEAR\b[:\-\s]*(20\d{2})\b", line)
        if year_match:
            fields["year"] = year_match.group(1)

        candidate_match = re.search(r"\b(\d{5,8}/\d{2,4})\b", line)
        if candidate_match and "candidate_id" not in fields:
            fields["candidate_id"] = candidate_match.group(1)

        roll_match = re.search(r"\b(?:NO\.?\s*)?([A-Z]{1,3}\s?\d{6,10})\b", line)
        if roll_match and "roll_no" not in fields:
            fields["roll_no"] = roll_match.group(1).replace(" ", "")

        if "UNIQUE" in line and "ID" in line:
            uid_match = re.search(r"\b(\d{5,12})\b", line)
            if uid_match:
                fields["unique_id"] = uid_match.group(1)

        if "NAME" in line and "student_name" not in fields:
            cleaned = re.sub(r"\bNAME\b[:\-\s]*", "", line).strip()
            cleaned = re.sub(r"[^A-Z\s]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                fields["student_name"] = cleaned.title()

        if "SMT" in line and "mother_name" not in fields:
            mother = line.split("SMT", 1)[-1].strip()
            mother = re.sub(r"[^A-Z\s]", " ", mother)
            mother = re.sub(r"\s+", " ", mother).strip()
            if mother:
                fields["mother_name"] = mother.title()

        if "SHRI" in line and "father_name" not in fields:
            father = line.split("SHRI", 1)[-1].strip()
            father = re.sub(r"[^A-Z\s]", " ", father)
            father = re.sub(r"\s+", " ", father).strip()
            if father:
                fields["father_name"] = father.title()

    subject_rows = []
    for line in upper_lines:
        for subject in SUBJECT_KEYWORDS:
            if subject in line:
                subject_rows.append(subject.title())
                break
    if subject_rows:
        fields["subjects"] = sorted(set(subject_rows))

    percentages = []
    for line in upper_lines:
        if "PERCENTAGE" in line or "MARKS" in line:
            matches = re.findall(r"\b([0-9]{2,3})\b", line)
            for match in matches:
                value = int(match)
                if 0 <= value <= 100:
                    percentages.append(value)
    if percentages:
        fields["marks_detected"] = percentages

    return fields


def _run_ocr(image_rgb: np.ndarray, reader: easyocr.Reader, paragraph: bool, preprocess_mode: str = "enhanced") -> pd.DataFrame:
    base_results = reader.readtext(image_rgb, detail=1, paragraph=paragraph)
    enhanced_preprocessed = _preprocess_for_ocr(image_rgb, mode=preprocess_mode)
    otsu_preprocessed = _preprocess_for_ocr_otsu(image_rgb)
    enhanced_results = reader.readtext(enhanced_preprocessed, detail=1, paragraph=paragraph)
    otsu_results = reader.readtext(otsu_preprocessed, detail=1, paragraph=paragraph)

    all_rows = (
        _rows_from_easyocr_results(base_results, "base")
        + _rows_from_easyocr_results(enhanced_results, preprocess_mode)
        + _rows_from_easyocr_results(otsu_results, "otsu")
    )
    merged_rows = _merge_ocr_rows(all_rows)

    rows: List[Dict] = []
    for idx, item in enumerate(merged_rows, start=1):
        rows.append(
            {
                "element_id": idx,
                "category": "Text",
                "x": int(item["x"]),
                "y": int(item["y"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "area_px": int(item["width"] * item["height"]),
                "ocr_text": item["ocr_text"],
                "ocr_confidence": float(item["ocr_confidence"]),
                "source": item["source"],
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.sort_values(["y", "x"]).reset_index(drop=True)
    dataframe["reading_order"] = np.arange(1, len(dataframe) + 1)
    dataframe["is_useful"] = dataframe.apply(
        lambda row: _is_useful_text(str(row["ocr_text"]), float(row["ocr_confidence"])),
        axis=1,
    )
    return dataframe


def _run_ocr_with_fallback(
    image_rgb: np.ndarray,
    primary_reader: easyocr.Reader,
    use_gpu: bool,
    paragraph: bool,
    primary_source: str,
    preprocess_mode: str = "enhanced",
) -> Tuple[pd.DataFrame, str, List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []


    def attempt(reader: easyocr.Reader, paragraph_mode: bool, reader_source: str, mode: str = preprocess_mode) -> pd.DataFrame:
        dataframe = _run_ocr(image_rgb, reader, paragraph=paragraph_mode, preprocess_mode=mode)
        attempts.append(
            {
                "reader_source": reader_source,
                "paragraph": paragraph_mode,
                "preprocess_mode": mode,
                "regions_detected": int(dataframe.shape[0]),
            }
        )
        return dataframe

    dataframe = attempt(primary_reader, paragraph, primary_source, preprocess_mode)
    if not dataframe.empty:
        return dataframe, primary_source, attempts

    if paragraph:
        dataframe = attempt(primary_reader, False, primary_source, preprocess_mode)
        if not dataframe.empty:
            return dataframe, primary_source, attempts

    if primary_source != "runtime-easyocr":
        runtime_reader = _get_reader("en", use_gpu)
        dataframe = attempt(runtime_reader, paragraph, "runtime-easyocr", preprocess_mode)
        if not dataframe.empty:
            return dataframe, "runtime-easyocr", attempts
        if paragraph:
            dataframe = attempt(runtime_reader, False, "runtime-easyocr", preprocess_mode)
            if not dataframe.empty:
                return dataframe, "runtime-easyocr", attempts

    return dataframe, primary_source, attempts


def _is_important_text(text: str) -> bool:
    normalized = _normalize_text(text)
    important_keywords = [
        "CERTIFICATE", "COURSE", "NAME", "AWARDED", "COMPLETION", "SUBJECT", "TITLE", "ROLL", "ID", "GRADE", "RESULT", "PERCENTAGE", "ISSUED", "DATE", "WORKSHOP", "TRAINING", "INSTITUTE", "COLLEGE", "UNIVERSITY"
    ]
    return any(keyword in normalized for keyword in important_keywords)

def _draw_overlay(image_rgb: np.ndarray, dataframe: pd.DataFrame, threshold: float) -> np.ndarray:
    annotated = image_rgb.copy()
    for _, row in dataframe.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])
        confidence = float(row["ocr_confidence"])
        text = str(row["ocr_text"])
        is_important = _is_important_text(text)
        # Green for important, else green if high confidence, else red
        if is_important:
            color = (0, 180, 0)  # vivid green
            thickness = 4
        elif confidence >= threshold:
            color = (45, 180, 70)
            thickness = 2
        else:
            color = (235, 90, 60)
            thickness = 2
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, thickness)
        cv2.putText(
            annotated,
            f"{confidence:.2f}",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        if is_important:
            cv2.putText(
                annotated,
                "IMPORTANT",
                (x, y + height + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return annotated


def _to_base64_png(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_base64_jpeg(image_rgb: np.ndarray, max_side: int = 1200, quality: int = 85) -> str:
    preview = _resize_if_needed(image_rgb, max_side=max_side)
    buffer = io.BytesIO()
    Image.fromarray(preview).save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_output(
    file_name: str,
    image_rgb: np.ndarray,
    dataframe: pd.DataFrame,
    extracted_fields: Dict,
    useful_dataframe: pd.DataFrame,
) -> Dict:
    full_text = [
        {
            "type": "text",
            "text": row["ocr_text"],
            "confidence": float(row["ocr_confidence"]),
        }
        for _, row in dataframe.iterrows()
    ]

    return {
        "metadata": {
            "file_name": file_name,
            "image_width": int(image_rgb.shape[1]),
            "image_height": int(image_rgb.shape[0]),
            "total_elements": int(len(dataframe)),
            "processed_at": dt.datetime.now().isoformat(),
            "ocr_engine": "EasyOCR",
            "pipeline_version": f"web-frontend-{APP_VERSION}",
        },
        "layout_elements": dataframe.to_dict(orient="records"),
        "useful_layout_elements": useful_dataframe.to_dict(orient="records"),
        "extracted_fields": extracted_fields,
        "document_structure": {
            "title": "",
            "page_header": "",
            "page_footer": "",
            "sections": [],
            "tables": [],
            "figures": [],
            "footnotes": [],
            "formulas": [],
            "full_text": full_text,
        },
    }


def _confidence_summary(confidences: List[float], threshold: float) -> Dict:
    if not confidences:
        return {"count": 0, "mean": 0.0, "threshold_hits": 0, "threshold_pct": 0.0}

    mean_value = float(np.mean(confidences))
    threshold_hits = int(sum(value >= threshold for value in confidences))
    return {
        "count": len(confidences),
        "mean": mean_value,
        "threshold_hits": threshold_hits,
        "threshold_pct": (threshold_hits / len(confidences)) * 100.0,
    }


def _project_metrics(profile: str, threshold: float) -> Dict:
    files = [Path("output") / "idp_output_1.json", Path("output") / "idp_batch_output.json"]
    existing_files = [path for path in files if path.exists()]
    if not existing_files:
        return {
            "files": [],
            "count": 0,
            "mean": 0.0,
            "threshold_hits": 0,
            "threshold_pct": 0.0,
            "categories": [],
        }

    selected_categories: Set[str] = (
        CORE_TEXT_CATEGORIES if profile == "core-text" else ALL_TEXT_CATEGORIES
    )
    confidences, by_category = aggregate(existing_files, selected_categories)
    summary = _confidence_summary(confidences, threshold)
    summary["files"] = [str(path) for path in existing_files]
    summary["categories"] = [
        {
            "category": category,
            "count": len(values),
            "mean_confidence": float(np.mean(values)),
        }
        for category, values in sorted(by_category.items())
    ]
    return summary


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return (
        jsonify(
            {
                "error": (
                    "Uploaded image is too large. "
                    "Please use a file smaller than 60 MB or reduce image resolution."
                )
            }
        ),
        413,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    return jsonify({"error": f"Unexpected server error: {error}"}), 500


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



@app.route("/")
@login_required
def index():
    asset_version = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return render_template(
        "index.html",
        gpu_available=torch.cuda.is_available(),
        app_version=APP_VERSION,
        asset_version=asset_version,
    )


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "app_version": APP_VERSION,
            "gpu_available": torch.cuda.is_available(),
            "model": _model_health(),
        }
    )


@app.get("/api/metrics")
def api_metrics():
    profile = request.args.get("profile", "core-text")
    if profile not in {"core-text", "all-text"}:
        profile = "core-text"

    threshold = _parse_float(
        request.args.get("threshold", "0.5"),
        default=0.5,
        min_value=0.0,
        max_value=1.0,
    )

    metrics = _project_metrics(profile=profile, threshold=threshold)
    return jsonify({"profile": profile, "threshold": threshold, "metrics": metrics})


@app.post("/api/process")
def api_process():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field in request."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    threshold = _parse_float(
        request.form.get("threshold", "0.5"),
        default=0.5,
        min_value=0.0,
        max_value=1.0,
    )
    paragraph = _parse_bool(request.form.get("paragraph", "false"), default=False)
    user_gpu = _parse_bool(request.form.get("use_gpu", "false"), default=False)
    use_gpu = user_gpu and torch.cuda.is_available()
    focus_mode = request.form.get("focus_mode", "useful").strip().lower()
    if focus_mode not in {"all", "useful"}:
        focus_mode = "useful"

    try:
        image_rgb = _to_numpy_rgb(file)
        image_rgb = _resize_if_needed(image_rgb)
    except Exception as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    preprocess_mode = request.form.get("preprocess_mode", "enhanced")
    try:
        reader, reader_source = _resolve_prediction_reader(use_gpu)
        dataframe, effective_reader_source, detection_attempts = _run_ocr_with_fallback(
            image_rgb=image_rgb,
            primary_reader=reader,
            use_gpu=use_gpu,
            paragraph=paragraph,
            primary_source=reader_source,
            preprocess_mode=preprocess_mode,
        )
    except Exception as exc:
        return jsonify({"error": f"OCR processing failed: {exc}"}), 500

    if dataframe.empty:
        return jsonify(
            {
                "message": (
                    "No text regions were detected. Try a clearer marksheet image, keep Paragraph Mode off, "
                    "or paste the image with Ctrl+V and run again."
                ),
                "stats": {"count": 0, "mean": 0.0, "threshold_hits": 0, "threshold_pct": 0.0},
                "elements": [],
                "display_elements": [],
                "result_json": {},
                "original_image": _to_base64_jpeg(image_rgb),
                "overlay_image": _to_base64_jpeg(image_rgb),
                "extracted_fields": {},
                "prediction_model": {
                    "reader_source": effective_reader_source,
                    "detection_attempts": detection_attempts,
                    "health": _model_health(),
                },
            }
        )

    useful_dataframe = dataframe[dataframe["is_useful"]].copy()
    if useful_dataframe.empty:
        useful_dataframe = dataframe.copy()
    display_dataframe = useful_dataframe if focus_mode == "useful" else dataframe

    extracted_fields = _extract_marksheet_fields(useful_dataframe)
    confidences = dataframe["ocr_confidence"].tolist()
    stats = _confidence_summary(confidences, threshold)
    overlay = _draw_overlay(image_rgb, display_dataframe, threshold=threshold)
    output_payload = _build_output(
        file.filename,
        image_rgb,
        dataframe,
        extracted_fields=extracted_fields,
        useful_dataframe=useful_dataframe,
    )

    return jsonify(
        {
            "stats": stats,
            "elements": dataframe.to_dict(orient="records"),
            "display_elements": display_dataframe.to_dict(orient="records"),
            "result_json": output_payload,
            "original_image": _to_base64_jpeg(image_rgb),
            "overlay_image": _to_base64_jpeg(overlay),
            "gpu_used": use_gpu,
            "focus_mode": focus_mode,
            "useful_elements_count": int(useful_dataframe.shape[0]),
            "extracted_fields": extracted_fields,
            "prediction_model": {
                "reader_source": effective_reader_source,
                "detection_attempts": detection_attempts,
                "health": _model_health(),
            },
            "app_version": APP_VERSION,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("IDP_FRONTEND_PORT", "5055"))
    debug_mode = _parse_bool(os.environ.get("IDP_FRONTEND_DEBUG", "false"), default=False)
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
