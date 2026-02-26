#!/usr/bin/env python
"""

Usage:
  # Process all data with default 10 workers
  python scripts/aliyun_caption.py

  # Process with more workers (faster but higher API load)
  python scripts/aliyun_caption.py --workers 20

  # Process first 5 samples only (for testing)
  python scripts/aliyun_caption.py --limit 5 --workers 1

  # Use custom CSV file
  python scripts/aliyun_caption.py --csv dataset/another.csv --limit 10
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import List, Optional
from threading import Lock
import numpy as np
import pandas as pd

try:
    import dashscope
    from dashscope import Generation
except ImportError:  # pragma: no cover - dashscope may be absent locally
    dashscope = None
    Generation = None


@dataclass
class SeriesConfig:
    seq_len: int = 48
    segments: int = 12
    minutes_per_step: int = 30  # 48 points / day -> 30min per step


WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
HOLIDAY_RANGES = [
    # 2009年
    ("New Year's Day", "2009-01-01", "2009-01-03"),
    ("Spring Festival", "2009-01-25", "2009-01-31"),
    ("Qingming Festival", "2009-04-04", "2009-04-06"),
    ("Labor Day", "2009-05-01", "2009-05-03"),
    ("Dragon Boat Festival", "2009-05-28", "2009-05-30"),
    ("National Day", "2009-10-01", "2009-10-07"),
    ("Mid-Autumn Festival", "2009-10-03", "2009-10-03"),  # 与国庆重叠
    # 2010年
    ("New Year's Day", "2010-01-01", "2010-01-03"),
    ("Spring Festival", "2010-02-13", "2010-02-19"),
    ("Qingming Festival", "2010-04-03", "2010-04-05"),
    ("Labor Day", "2010-05-01", "2010-05-03"),
    ("Dragon Boat Festival", "2010-06-14", "2010-06-16"),
    ("Mid-Autumn Festival", "2010-09-22", "2010-09-24"),
    ("National Day", "2010-10-01", "2010-10-07"),
]


def _expand_holiday_ranges() -> dict:
    """Turn the holiday ranges into a date->holiday_name mapping."""
    mapping = {}
    for name, start_str, end_str in HOLIDAY_RANGES:
        start = pd.to_datetime(start_str).date()
        end = pd.to_datetime(end_str).date()
        cur = start
        while cur <= end:
            mapping[cur] = name
            cur += timedelta(days=1)
    return mapping


HOLIDAY_LOOKUP = _expand_holiday_ranges()


def calendar_info(day_str: str) -> dict:
    """Return weekday/holiday context for the given YYYY-M-D string."""
    day_dt: date = pd.to_datetime(day_str).date()
    weekday_en = day_dt.strftime("%A")
    weekday_cn = WEEKDAY_CN[day_dt.weekday()]
    holiday = HOLIDAY_LOOKUP.get(day_dt)
    is_weekend = day_dt.weekday() >= 5

    if holiday:
        day_type = f"Holiday: {holiday}"
    elif is_weekend:
        day_type = "Weekend"
    else:
        day_type = "Regular working day"

    context = f"{day_dt.isoformat()} ({weekday_en}/{weekday_cn}); {day_type}"
    return {
        "date": day_dt.isoformat(),
        "weekday_en": weekday_en,
        "weekday_cn": weekday_cn,
        "holiday": holiday,
        "is_weekend": is_weekend,
        "context": context,
    }


def summarize_daily_metrics(series: np.ndarray) -> dict:
    """Compute high-level daily aggregates for extra grounding in the prompt."""
    arr = np.asarray(series, dtype=np.float32).reshape(-1)
    return {
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _split_series(values: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    """Chunk a 1D array into [segments, slot_len] windows."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < cfg.seq_len:
        raise ValueError(f"series length {arr.size} < expected seq_len={cfg.seq_len}")
    trimmed = arr[: cfg.seq_len]
    slot_len = max(1, cfg.seq_len // cfg.segments)
    slots = trimmed.reshape(cfg.segments, slot_len)
    return slots


def _window_caption(window: np.ndarray, idx: int, cfg: SeriesConfig) -> str:
    """Generate a textual summary for a single window (no LLM yet)."""
    start_step = idx * window.size
    start_min = start_step * cfg.minutes_per_step
    end_min = start_min + window.size * cfg.minutes_per_step
    w_min = float(window.min())
    w_max = float(window.max())
    w_mean = float(window.mean())
    # Treat near-identical readings as a constant segment so it is not mislabeled as "flat".
    if w_max - w_min <= 1e-6:
        trend = "constant"
    else:
        trend = "up" if window[-1] > window[0] else ("down" if window[-1] < window[0] else "flat")
    return (
        f"{start_min // 60:02d}:{start_min % 60:02d}-{end_min // 60:02d}:{end_min % 60:02d}: "
        f"avg {w_mean:.3f}, min {w_min:.3f}, max {w_max:.3f}, trend {trend}"
    )


def build_prompt(
    series: np.ndarray,
    cfg: SeriesConfig,
    *,
    calendar_context: Optional[str] = None,
    daily_metrics: Optional[dict] = None,
) -> str:
    """Convert one series into the textual prompt we send to DashScope."""
    slots = _split_series(series, cfg)
    slot_desc = "\n".join(_window_caption(window, idx, cfg) for idx, window in enumerate(slots))
    meta_parts = []
    if calendar_context:
        meta_parts.append(f"Calendar context: {calendar_context}")
    if daily_metrics:
        meta_parts.append(
            "Daily aggregates (sum across all 30-minute intervals): "
            f"total={daily_metrics['total']:.3f}, "
            f"avg={daily_metrics['mean']:.3f}, "
            f"min={daily_metrics['min']:.3f}, "
            f"max={daily_metrics['max']:.3f}"
        )
    meta_block = "\n".join(meta_parts)
    meta_block = f"{meta_block}\n" if meta_block else ""
    return (
        "You are a data annotator. "
        "Given the following 24-hour load summary broken into 4-hour slots, "
        "summarize the consumption pattern in natural language (highlight peaks, valleys, steady segments). "
        "Use the calendar context (weekday/holiday info) and the daily aggregates so the description mentions them. "
        "Don't add explanations or questions; provide one concise paragraph. "
        f"{meta_block}"
        f"{slot_desc}\n"
        "Respond with a concise descriptive paragraph only."
    )



def call_dashscope(prompt: str, model: str, api_key: Optional[str]) -> str:
    """Invoke Aliyun DashScope Generation API and return the text output."""
    if Generation is None:
        raise RuntimeError("dashscope SDK is not installed; run `pip install dashscope`.")

    # 硬编码的默认密钥（优先级：传入参数 > 环境变量 > 硬编码密钥）
    DEFAULT_API_KEY = "sk-ad2870f9d15845248a194f72b9573636"
    key = api_key or os.getenv("DASHSCOPE_API_KEY") or DEFAULT_API_KEY

    dashscope.api_key = key
    response = Generation.call(
        model=model,
        prompt=prompt,
        enable_search=False,
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope error {response.status_code}: {response.message}")

    output = response.output.get("text")
    if isinstance(output, list):
        return "".join(output)
    return str(output)

def load_series(csv_path: Path, start: int, limit: Optional[int], id_col: str, day_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"id_col {id_col} not in columns: {list(df.columns)[:10]}...")
    if day_col not in df.columns:
        raise ValueError(f"day_col {day_col} not in columns: {list(df.columns)[:10]}...")
    cols = [c for c in df.columns if ":" in c]  # 00:00, 00:30, ...
    # limit 为 None 时处理整个文件
    end = start + limit if limit is not None else None
    subset = df.iloc[start:end].copy()
    subset["series_values"] = subset[cols].values.tolist()
    subset = subset.rename(columns={id_col: "user_id", day_col: "day"})
    keep_cols = ["user_id", "day", "series_values"]
    return subset[keep_cols]


def process_single_sample(
    idx,
    row,
    cfg: SeriesConfig,
    args,
    writer_lock: Lock,
    writer,
    completed_count: List[int],
    total: int,
):
    """Process a single sample with retry logic."""
    series = np.asarray(row["series_values"], dtype=np.float32)
    user_id = row["user_id"]
    day = row["day"]
    day_context = calendar_info(str(day))
    daily_metrics = summarize_daily_metrics(series)
    prompt = build_prompt(
        series,
        cfg,
        calendar_context=day_context["context"],
        daily_metrics=daily_metrics,
    )
    if args.include_meta_in_prompt:
        prompt = f"User ID: {user_id}\nDate: {day}\n{prompt}"

    # Retry logic with exponential backoff
    caption = None
    last_error = None
    for attempt in range(args.max_retries):
        try:
            caption = call_dashscope(prompt, args.model, args.api_key)
            break
        except Exception as exc:
            last_error = exc
            wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
            print(f"[Sample {idx}] attempt {attempt + 1}/{args.max_retries} failed: {exc}. Retry in {wait_time}s...")
            time.sleep(wait_time)
    
    if caption is None:
        print(f"[Sample {idx}] FAILED after {args.max_retries} retries: {last_error}")
        return None

    caption_fields = [
        f"UserID={user_id}",
        f"Day={day}",
        f"Weekday={day_context['weekday_en']}({day_context['weekday_cn']})",
    ]
    if day_context["holiday"]:
        caption_fields.append(f"Holiday={day_context['holiday']}")
    elif day_context["is_weekend"]:
        caption_fields.append("Weekend")
    caption_fields.append(f"DailyTotal={daily_metrics['total']:.3f}")
    caption_prefix = "; ".join(caption_fields)
    caption_text = f"{caption_prefix}; {caption}"

    # Thread-safe write
    with writer_lock:
        completed_count[0] += 1
        progress = completed_count[0] / total * 100
        print(f"[{completed_count[0]}/{total} {progress:.1f}%] User={user_id}, Day={day}")
        if writer:
            if not args.full_output:
                record = {
                    "user_id": user_id,
                    "day": day,
                    "caption_text": caption_text,
                }
            else:
                record = {
                    "user_id": user_id,
                    "day": day,
                    "prompt": prompt,
                    "caption": caption,
                    "calendar_info": day_context,
                    "daily_metrics": daily_metrics,
                    "caption_text": caption_text,
                    "source": str(args.csv),
                }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
    
    return caption_text


def main():
    parser = argparse.ArgumentParser(description="Generate captions via Aliyun DashScope.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/File1_test_daily_flag_4_2.csv"),
        help="Path to the CSV file (default: dataset/File1_test_daily_flag_4_2.csv).",
    )
    parser.add_argument("--seq-len", type=int, default=48, help="Number of points per sample.")
    parser.add_argument("--segments", type=int, default=12, help="How many windows to describe.")
    parser.add_argument("--minutes-per-step", type=int, default=30, help="Sampling interval minutes.")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="DashScope model name.")
    parser.add_argument("--api-key", type=str, default=None, help="Override DASHSCOPE_API_KEY.")
    parser.add_argument("--start", type=int, default=0, help="Row offset in the CSV.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many samples to caption (default: process all data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/cai/dataset/captions_output.jsonl"),
        help="JSONL output path (default: /root/cai/dataset/captions_output.jsonl)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per caption.")
    parser.add_argument("--id-col", type=str, default="ID", help="Column name for user ID.")
    parser.add_argument("--day-col", type=str, default="DAY", help="Column name for date/day index.")
    parser.add_argument(
        "--include-meta-in-prompt",
        action="store_true",
        help="Prepend user ID and date into the prompt so the caption text carries them.",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="If set, store prompt/calendar/daily_metrics for debugging. Default output is minimal.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers for API calls (default: 10).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per request (default: 3).",
    )

    args = parser.parse_args()

    cfg = SeriesConfig(seq_len=args.seq_len, segments=args.segments, minutes_per_step=args.minutes_per_step)
    data = load_series(args.csv, args.start, args.limit, args.id_col, args.day_col)
    total = len(data)
    
    print(f"Loaded {total} samples from {args.csv}")
    print(f"Using {args.workers} concurrent workers")
    print(f"Output will be saved to: {args.output}")
    print("-" * 50)

    writer = None
    writer_lock = Lock()
    completed_count = [0]  # Use list for mutable reference
    
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = args.output.open("a", encoding="utf-8")

    results: List[str] = []
    
    try:
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_sample,
                    idx,
                    row,
                    cfg,
                    args,
                    writer_lock,
                    writer,
                    completed_count,
                    total,
                ): idx
                for idx, row in data.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                result = future.result()
                if result:
                    results.append(result)
        
        print("-" * 50)
        print(f"Completed: {len(results)}/{total} samples")
        if len(results) < total:
            print(f"Failed: {total - len(results)} samples")
            
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
