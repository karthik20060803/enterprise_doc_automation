import datetime as dt
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from evaluate_accuracy import ALL_TEXT_CATEGORIES, CORE_TEXT_CATEGORIES, aggregate


def _to_numpy_rgb(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


@st.cache_resource(show_spinner=False)
def _get_reader(languages: Tuple[str, ...], use_gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)


def _polygon_to_bbox(points: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = max(max(xs), 0), max(max(ys), 0)
    return x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)


def run_ocr_pipeline(image_rgb: np.ndarray, reader: easyocr.Reader, paragraph: bool) -> pd.DataFrame:
    results = reader.readtext(image_rgb, detail=1, paragraph=paragraph)
    rows: List[Dict] = []
    for idx, item in enumerate(results, start=1):
        if len(item) < 3:
            continue
        bbox_points, text, confidence = item[0], item[1], item[2]
        x, y, w, h = _polygon_to_bbox(bbox_points)
        rows.append(
            {
                "element_id": idx,
                "category": "Text",
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_px": int(w * h),
                "ocr_text": str(text).strip(),
                "ocr_confidence": float(confidence),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["y", "x"]).reset_index(drop=True)
    df["reading_order"] = np.arange(1, len(df) + 1)
    return df


def draw_overlay(image_rgb: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    annotated = image_rgb.copy()
    for _, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])
        conf = float(row["ocr_confidence"])
        color = (45, 180, 70) if conf >= 0.5 else (235, 90, 60)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def build_output(file_name: str, image_rgb: np.ndarray, df: pd.DataFrame) -> Dict:
    full_text = [
        {"type": "text", "text": row["ocr_text"], "confidence": float(row["ocr_confidence"])}
        for _, row in df.iterrows()
    ]
    return {
        "metadata": {
            "file_name": file_name,
            "image_width": int(image_rgb.shape[1]),
            "image_height": int(image_rgb.shape[0]),
            "total_elements": int(len(df)),
            "processed_at": dt.datetime.now().isoformat(),
            "ocr_engine": "EasyOCR",
            "pipeline_version": "frontend-1.0.0",
        },
        "layout_elements": df.to_dict(orient="records"),
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


def _confidence_summary(confidences: List[float], threshold: float) -> Dict[str, float]:
    if not confidences:
        return {"count": 0, "mean": 0.0, "threshold_hits": 0, "threshold_pct": 0.0}
    mean_conf = float(np.mean(confidences))
    hits = int(sum(c >= threshold for c in confidences))
    return {
        "count": len(confidences),
        "mean": mean_conf,
        "threshold_hits": hits,
        "threshold_pct": (hits / len(confidences)) * 100.0,
    }


def _project_metric_summary(profile: str, threshold: float) -> Dict:
    output_files = [Path("output") / "idp_output_1.json", Path("output") / "idp_batch_output.json"]
    existing_files = [p for p in output_files if p.exists()]
    if not existing_files:
        return {"files": [], "count": 0, "mean": 0.0, "threshold_hits": 0, "threshold_pct": 0.0}

    categories = CORE_TEXT_CATEGORIES if profile == "core-text" else ALL_TEXT_CATEGORIES
    confidences, by_category = aggregate(existing_files, categories)
    summary = _confidence_summary(confidences, threshold)
    summary["files"] = [str(p) for p in existing_files]
    summary["by_category"] = by_category
    return summary


def render_upload_tab() -> None:
    st.subheader("Upload and run OCR")
    st.caption("This mode processes any uploaded image directly with EasyOCR.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col_b:
        paragraph = st.checkbox("Paragraph mode", value=False)
    with col_c:
        use_gpu = st.checkbox("Use GPU if available", value=torch.cuda.is_available())

    uploaded = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])
    if not uploaded:
        return

    image_rgb = _to_numpy_rgb(uploaded)
    reader = _get_reader(("en",), use_gpu)
    with st.spinner("Running OCR..."):
        df = run_ocr_pipeline(image_rgb, reader, paragraph)

    if df.empty:
        st.warning("No text regions were detected.")
        return

    confidences = df["ocr_confidence"].tolist()
    stats = _confidence_summary(confidences, threshold)
    output_payload = build_output(uploaded.name, image_rgb, df)
    overlay = draw_overlay(image_rgb, df)

    c1, c2 = st.columns([1.3, 1.0])
    with c1:
        st.image(image_rgb, caption="Original", use_container_width=True)
        st.image(overlay, caption="OCR overlay", use_container_width=True)
    with c2:
        st.metric("Detected text regions", stats["count"])
        st.metric("Mean OCR confidence", f"{stats['mean'] * 100:.2f}%")
        st.metric(
            f"Threshold >= {threshold:.2f}",
            f"{stats['threshold_pct']:.2f}% ({stats['threshold_hits']}/{stats['count']})",
        )

    show_df = df.copy()
    show_df["ocr_confidence"] = show_df["ocr_confidence"].map(lambda x: f"{x:.4f}")
    st.dataframe(
        show_df[["reading_order", "x", "y", "width", "height", "ocr_confidence", "ocr_text"]],
        use_container_width=True,
        hide_index=True,
    )

    json_bytes = json.dumps(output_payload, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "Download JSON result",
        data=json_bytes,
        file_name=f"idp_frontend_output_{Path(uploaded.name).stem}.json",
        mime="application/json",
    )

    img_buf = io.BytesIO()
    Image.fromarray(overlay).save(img_buf, format="PNG")
    st.download_button(
        "Download overlay image",
        data=img_buf.getvalue(),
        file_name=f"idp_overlay_{Path(uploaded.name).stem}.png",
        mime="image/png",
    )


def render_metrics_tab() -> None:
    st.subheader("Project output metrics")
    st.caption("Reads existing JSON outputs in the output folder.")

    left, right = st.columns(2)
    with left:
        profile = st.selectbox("Metric profile", options=["core-text", "all-text"], index=0)
    with right:
        threshold = st.slider("Project threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    summary = _project_metric_summary(profile, threshold)
    if not summary["files"]:
        st.warning("No output JSON files found. Expected output/idp_output_1.json or output/idp_batch_output.json.")
        return

    st.write("Files used:", ", ".join(summary["files"]))
    st.metric("Text regions evaluated", int(summary["count"]))
    st.metric("OCR confidence accuracy", f"{summary['mean'] * 100:.2f}%")
    st.metric(
        f"Threshold >= {threshold:.2f}",
        f"{summary['threshold_pct']:.2f}% ({summary['threshold_hits']}/{summary['count']})",
    )

    by_category = summary.get("by_category", {})
    if by_category:
        rows = []
        for category in sorted(by_category):
            values = by_category[category]
            rows.append(
                {
                    "category": category,
                    "count": len(values),
                    "mean_confidence": float(np.mean(values)) * 100.0,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Enterprise Document Automation", layout="wide")
    st.title("Enterprise Document Automation Frontend")
    st.write(
        "Frontend for your IDP model: upload a document image, run OCR, inspect confidence, and "
        "download structured JSON for backend/frontend integration."
    )

    tab_upload, tab_metrics = st.tabs(["Upload OCR", "Project Metrics"])
    with tab_upload:
        render_upload_tab()
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
