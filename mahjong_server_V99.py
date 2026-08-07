# # mahjong_server_V93.py
# Mahjong Local Web Server V66
# HTML/phone browser uploads Mahjong photo to Python server.
# Server runs YOLO detection + AI recognition + simple HK fan scoring.

from pathlib import Path
from datetime import datetime
import traceback
import shutil
import json
import re

import cv2
import numpy as np
import joblib

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


APP_TITLE = "Mahjong Web Recognizer V99"

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_FILE = BASE_DIR / "yolo_runs" / "mahjong_tile_detector" / "weights" / "best.pt"
AI_MODEL_FILE = BASE_DIR / "ai_model" / "mahjong_ai_classifier.joblib"
OUTPUT_DIR = BASE_DIR / "server_output"
WEB_HTML_FILE = BASE_DIR / "mahjong_web_V99.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DISPLAY_NAMES = {
    "1_wan": "一萬", "2_wan": "二萬", "3_wan": "三萬",
    "4_wan": "四萬", "5_wan": "伍萬", "6_wan": "六萬",
    "7_wan": "七萬", "8_wan": "八萬", "9_wan": "九萬",

    "1_tong": "一筒", "2_tong": "二筒", "3_tong": "三筒",
    "4_tong": "四筒", "5_tong": "五筒", "6_tong": "六筒",
    "7_tong": "七筒", "8_tong": "八筒", "9_tong": "九筒",

    "1_sok": "一索", "2_sok": "二索", "3_sok": "三索",
    "4_sok": "四索", "5_sok": "五索", "6_sok": "六索",
    "7_sok": "七索", "8_sok": "八索", "9_sok": "九索",

    "east": "東", "south": "南", "west": "西", "north": "北",
    "red": "中", "green": "發", "white": "白",

    "flower_spring": "春", "flower_summer": "夏",
    "flower_autumn": "秋", "flower_winter": "冬",
    "flower_plum": "梅", "flower_orchid": "蘭",
    "flower_chrysanthemum": "菊", "flower_bamboo": "竹",
    "tile_back": "背",
}


class MahjongEngine:
    def __init__(self):
        self.yolo_model = None
        self.ai_model = None
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 96),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

    def load_models(self):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")

        if not YOLO_MODEL_FILE.exists():
            raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_FILE}")

        if not AI_MODEL_FILE.exists():
            raise FileNotFoundError(f"AI classifier not found: {AI_MODEL_FILE}")

        if self.yolo_model is None:
            self.yolo_model = YOLO(str(YOLO_MODEL_FILE))

        if self.ai_model is None:
            bundle = joblib.load(AI_MODEL_FILE)
            self.ai_model = bundle.get("model")
            if self.ai_model is None:
                raise RuntimeError("AI model file loaded but no 'model' found inside joblib bundle.")

    def recognize_image(
        self,
        image_path: Path,
        conf: float = 0.10,
        orientation: str = "off",
        side_rotation: bool = True,
        self_draw: bool = False,
        concealed: bool = False,
        flower_count: int = 0,
        seat_wind: str = "none",
        round_wind: str = "east",
        rule_set: str = "hk",
        extra_options=None,
        expected_count: int = 0,
        auto_retry: bool = True,
        trim_to_expected: bool = True,
    ):
        self.load_models()

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        conf = max(0.001, min(float(conf), 0.99))
        expected_count = int(expected_count or 0)

        boxes, used_conf, detection_attempts = self.detect_yolo_boxes_smart(
            bgr,
            conf=conf,
            expected_count=expected_count,
            auto_retry=auto_retry,
            trim_to_expected=trim_to_expected,
        )

        tiles = []
        preview = bgr.copy()

        for idx, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            x1 = max(0, min(x1, bgr.shape[1] - 1))
            y1 = max(0, min(y1, bgr.shape[0] - 1))
            x2 = max(0, min(x2, bgr.shape[1] - 1))
            y2 = max(0, min(y2, bgr.shape[0] - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = bgr[y1:y2, x1:x2]
            label, ai_score, top3, used_orientation = self.predict_ai(
                crop,
                orientation=orientation,
                side_rotation=side_rotation
            )

            display = DISPLAY_NAMES.get(label, label)

            tiles.append({
                "index": idx,
                "label": label,
                "display": display,
                "ai_score": float(ai_score),
                "orientation": used_orientation,
                "yolo_score": float(box.get("score", 0.0)),
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "top3": [
                    {"label": name, "display": DISPLAY_NAMES.get(name, name), "score": float(score)}
                    for name, score in top3
                ],
            })

            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                preview,
                str(idx),
                (x1 + 8, max(28, y1 + 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            self.draw_text_cv2(preview, display, (x1 + 6, min(bgr.shape[0] - 12, y2 - 8)))

        tile_labels = [t["label"] for t in tiles]

        score_result = score_by_rule(
            tile_labels,
            rule_set=rule_set,
            self_draw=self_draw,
            concealed=concealed,
            flower_count=flower_count,
            seat_wind=seat_wind,
            round_wind=round_wind,
            extra_options=extra_options
        )

        run_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        preview_file = run_dir / "detected_preview.png"
        cv2.imwrite(str(preview_file), preview)

        result = {
            "ok": True,
            "tile_count": len(tiles),
            "tiles": tiles,
            "recognized_row": [t["display"] for t in tiles],
            "recognized_labels": tile_labels,
            "score": score_result,
            "preview_url": f"/output/{run_dir.name}/detected_preview.png",
            "settings": {
                "requested_conf": conf,
                "used_conf": used_conf,
                "expected_count": expected_count,
                "auto_retry": auto_retry,
                "trim_to_expected": trim_to_expected,
                "detection_attempts": detection_attempts,
                "orientation": orientation,
                "side_rotation": side_rotation,
                "self_draw": self_draw,
                "concealed": concealed,
                "flower_count": flower_count,
                "seat_wind": seat_wind,
                "round_wind": round_wind,
                "rule_set": rule_set,
            }
        }

        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def clean_yolo_boxes(self, boxes):
        """
        V8: remove obvious false YOLO boxes, especially huge low-confidence
        background boxes that cover several real tiles.
        """
        if not boxes:
            return []

        valid = []
        for b in boxes:
            w = max(0, int(b["x2"]) - int(b["x1"]))
            h = max(0, int(b["y2"]) - int(b["y1"]))
            if w >= 8 and h >= 8:
                nb = dict(b)
                nb["_w"] = w
                nb["_h"] = h
                nb["_area"] = w * h
                valid.append(nb)

        if len(valid) <= 2:
            return sorted(valid, key=lambda b: (b["x1"] + b["x2"]) / 2)

        areas = sorted([b["_area"] for b in valid])
        widths = sorted([b["_w"] for b in valid])
        heights = sorted([b["_h"] for b in valid])

        median_area = areas[len(areas) // 2]
        median_w = widths[len(widths) // 2]
        median_h = heights[len(heights) // 2]

        cleaned = []
        for b in valid:
            if b["_area"] > median_area * 2.8:
                continue
            if b["_w"] > median_w * 2.4:
                continue
            if b["_h"] > median_h * 2.2:
                continue
            cleaned.append(b)

        if not cleaned:
            cleaned = valid

        cleaned = sorted(cleaned, key=lambda b: b.get("score", 0.0), reverse=True)
        final = []
        for b in cleaned:
            if all(self.box_iou(b, fb) <= 0.50 for fb in final):
                final.append(b)

        final = sorted(final, key=lambda b: (b["x1"] + b["x2"]) / 2)
        for b in final:
            b.pop("_w", None)
            b.pop("_h", None)
            b.pop("_area", None)
        return final

    def box_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def detect_yolo_boxes_smart(
        self,
        bgr,
        conf: float = 0.03,
        expected_count: int = 0,
        auto_retry: bool = True,
        trim_to_expected: bool = True,
    ):
        """
        V16 web smart detection.

        This now follows the stable desktop Recognition app idea:
        - requested Conf first
        - then 0.03 / 0.05 / 0.10
        - merge valid non-overlapping boxes
        - no expected count means no trimming
        - no 0.01 auto step
        """
        conf = max(0.03, min(float(conf), 0.99))

        try:
            expected_count = int(expected_count or 0)
        except Exception:
            expected_count = 0

        if not auto_retry:
            raw_boxes = self.detect_yolo_boxes(bgr, conf)
            boxes = self.clean_yolo_boxes(raw_boxes)
            return boxes, conf, [{
                "conf": conf,
                "raw_count": len(raw_boxes),
                "count": len(boxes),
                "merged": len(boxes)
            }]

        candidate_confs = [conf, 0.03, 0.05, 0.10]

        confs = []
        for c in candidate_confs:
            c = max(0.03, min(float(c), 0.99))
            if c not in confs:
                confs.append(c)

        merged = []
        attempts = []

        for c in confs:
            raw_boxes = self.detect_yolo_boxes(bgr, c)
            clean_boxes = self.clean_yolo_boxes(raw_boxes)

            added = 0
            for b in clean_boxes:
                # Add only new tile-like boxes, not duplicates.
                if all(self.box_iou(b, existing) <= 0.35 for existing in merged):
                    merged.append(b)
                    added += 1

            merged = self.clean_yolo_boxes(merged)
            merged = sorted(merged, key=lambda b: (b["x1"] + b["x2"]) / 2)

            attempts.append({
                "conf": c,
                "raw_count": len(raw_boxes),
                "count": len(clean_boxes),
                "added": added,
                "merged": len(merged)
            })

        if expected_count > 0 and trim_to_expected and len(merged) > expected_count:
            before = len(merged)
            merged = sorted(merged, key=lambda b: b.get("score", 0.0), reverse=True)[:expected_count]
            merged = sorted(merged, key=lambda b: (b["x1"] + b["x2"]) / 2)
            attempts.append({
                "conf": "trim",
                "raw_count": before,
                "count": len(merged),
                "merged": len(merged)
            })

        return merged, conf, attempts

    def detect_yolo_boxes(self, bgr, conf: float):
        results = self.yolo_model.predict(source=bgr, conf=conf, verbose=False)

        boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    boxes.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score
                    })

        boxes = sorted(boxes, key=lambda b: (b["x1"] + b["x2"]) / 2)
        return boxes

    def normalize_tile_for_ai(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        mx = max(1, int(w * 0.04))
        my = max(1, int(h * 0.04))

        if h > 2 * my and w > 2 * mx:
            gray = gray[my:h-my, mx:w-mx]

        gray = cv2.resize(gray, (64, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        return gray

    def extract_ai_features(self, bgr_img):
        gray = self.normalize_tile_for_ai(bgr_img)
        hog_feat = self.hog.compute(gray).flatten()
        small = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA).flatten() / 255.0
        feat = np.concatenate([hog_feat, small])
        return feat.astype(np.float32)

    def predict_ai_raw(self, crop_bgr):
        feat = self.extract_ai_features(crop_bgr)
        X = np.array([feat], dtype=np.float32)

        try:
            pred = self.ai_model.predict(X)[0]
            scores = self.ai_model.decision_function(X)

            if scores.ndim == 1:
                classes = list(self.ai_model.named_steps["clf"].classes_)
                vals = scores.tolist()
                vals = [-vals[0], vals[0]]
            else:
                classes = list(self.ai_model.named_steps["clf"].classes_)
                vals = scores[0].tolist()

            ranked = sorted(zip(classes, vals), key=lambda x: x[1], reverse=True)
            top3 = ranked[:3]
            best_score = top3[0][1] if top3 else 0.0

            return pred, float(best_score), [(n, float(s)) for n, s in top3]
        except Exception:
            pred = self.ai_model.predict(X)[0]
            return pred, 0.0, [(pred, 0.0)]

    def predict_ai(self, crop_bgr, orientation: str = "off", side_rotation: bool = True):
        orientation = (orientation or "off").lower()

        candidates = []

        if orientation == "force180":
            candidates.append(("rotated180", cv2.rotate(crop_bgr, cv2.ROTATE_180)))
        else:
            candidates.append(("normal", crop_bgr))

        if side_rotation:
            try:
                candidates.append(("rotated90", cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)))
                candidates.append(("rotated270", cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)))
            except Exception:
                pass

        best_label = "unknown"
        best_score = -999999.0
        best_top3 = []
        best_orientation = "normal"

        for name, cand in candidates:
            try:
                label, score, top3 = self.predict_ai_raw(cand)
                if score > best_score:
                    best_label = label
                    best_score = score
                    best_top3 = top3
                    best_orientation = name
            except Exception:
                pass

        return best_label, best_score, best_top3, best_orientation

    def draw_text_cv2(self, image, text, pos):
        try:
            from PIL import Image, ImageDraw, ImageFont
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)

            font_candidates = [
                r"C:\Windows\Fonts\msjh.ttc",
                r"C:\Windows\Fonts\mingliu.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
            ]

            font = None
            for fp in font_candidates:
                if Path(fp).exists():
                    font = ImageFont.truetype(fp, 22)
                    break
            if font is None:
                font = ImageFont.load_default()

            x, y = pos
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
            draw.text((x, y), text, font=font, fill=(0, 0, 255))
            image[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(image, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


# ------------------------------------------------------------
# V17 Rule-set scoring helpers
# ------------------------------------------------------------
def score_by_rule(
    tiles,
    rule_set="hk",
    self_draw=False,
    concealed=False,
    flower_count=0,
    seat_wind="none",
    round_wind="east",
    extra_options=None,
):
    rule_set = (rule_set or "hk").lower()

    if rule_set in ["tw", "taiwan", "taiwanese"]:
        return score_tw_simple(
            tiles,
            self_draw=self_draw,
            concealed=concealed,
            flower_count=flower_count,
            seat_wind=seat_wind,
            round_wind=round_wind,
            extra_options=extra_options
        )

    return score_hk_simple(
        tiles,
        self_draw=self_draw,
        concealed=concealed,
        flower_count=flower_count,
        seat_wind=seat_wind,
        round_wind=round_wind
    )


def add_fan_once(fan_items, total_holder, key_set, name, fan):
    if name in key_set:
        return
    fan_items.append({"name": name, "fan": fan})
    total_holder[0] += fan
    key_set.add(name)


def is_terminal_tile(t):
    suit, num = tile_suit_num(t)
    return suit in ["wan", "tong", "sok"] and num in [1, 9]


def is_honor_tile(t):
    return t in ["east", "south", "west", "north", "red", "green", "white"]


def is_simple_tile(t):
    suit, num = tile_suit_num(t)
    return suit in ["wan", "tong", "sok"] and num is not None and 2 <= num <= 8


def detect_extra_patterns(counts, main_tiles, decomposition=None):
    """
    Detect extra patterns from closed visible tiles only.
    Some real-world rules involving open/closed meld state are intentionally not included yet.
    """
    patterns = []

    if not main_tiles:
        return patterns

    all_honor = all(is_honor_tile(t) for t in main_tiles)
    all_terminal = all(is_terminal_tile(t) for t in main_tiles)
    all_terminal_or_honor = all(is_terminal_tile(t) or is_honor_tile(t) for t in main_tiles)
    all_simple = all(is_simple_tile(t) for t in main_tiles)

    if all_honor:
        patterns.append(("字一色", 10))
    elif all_terminal:
        patterns.append(("清老頭", 10))
    elif all_terminal_or_honor:
        patterns.append(("混老頭", 5))

    if all_simple:
        patterns.append(("斷么九", 1))

    # Big / small four winds
    wind_pungs = sum(1 for w in ["east", "south", "west", "north"] if counts.get(w, 0) >= 3)
    wind_pairs = sum(1 for w in ["east", "south", "west", "north"] if counts.get(w, 0) == 2)

    if wind_pungs == 4:
        patterns.append(("大四喜", 13))
    elif wind_pungs == 3 and wind_pairs >= 1:
        patterns.append(("小四喜", 8))

    # Three same-number pungs across three suits
    for n in range(1, 10):
        if all(counts.get(f"{n}_{s}", 0) >= 3 for s in ["wan", "tong", "sok"]):
            patterns.append((f"三色同刻 {n}", 3))

    # Nine gates: very simple closed-hand visible pattern check
    # 1112345678999 + any same suit extra tile, no honors.
    if len(main_tiles) == 14:
        suit_result = detect_suit_pattern(main_tiles)
        if suit_result == "pure":
            suits = {tile_suit_num(t)[0] for t in main_tiles if tile_suit_num(t)[0] in ["wan", "tong", "sok"]}
            if len(suits) == 1:
                s = list(suits)[0]
                need = {1: 3, 9: 3}
                ok = True
                for n in range(1, 10):
                    c = counts.get(f"{n}_{s}", 0)
                    min_need = need.get(n, 1)
                    if c < min_need:
                        ok = False
                        break
                if ok:
                    patterns.append(("九蓮寶燈", 10))

    # Decomposition-based patterns
    if decomposition:
        melds = decomposition.get("melds", [])

        chow_sets = []
        pung_tiles = []

        for kind, tiles in melds:
            if kind == "chow":
                suit, num = tile_suit_num(tiles[0])
                chow_sets.append((suit, num))
            elif kind == "pung":
                pung_tiles.append(tiles[0])

        # 一氣通貫: 123,456,789 in same suit
        for s in ["wan", "tong", "sok"]:
            if (s, 1) in chow_sets and (s, 4) in chow_sets and (s, 7) in chow_sets:
                patterns.append(("一氣通貫", 3))

        # 三色同順: same chow number in all 3 suits
        for n in range(1, 8):
            if all((s, n) in chow_sets for s in ["wan", "tong", "sok"]):
                patterns.append((f"三色同順 {n}-{n+1}-{n+2}", 3))

    return patterns



def add_tw_item(fan_items, total_holder, existing, name, fan):
    """
    V19 Taiwan / 番數 table helper.
    """
    if name in existing:
        return
    fan_items.append({"name": name, "fan": int(fan)})
    total_holder[0] += int(fan)
    existing.add(name)


def count_meld_patterns_from_decomposition(decomposition):
    chow_counts = {}
    pung_counts = {}

    if not decomposition:
        return chow_counts, pung_counts

    for kind, tiles in decomposition.get("melds", []):
        if kind == "chow":
            suit, num = tile_suit_num(tiles[0])
            if suit in ["wan", "tong", "sok"] and num is not None:
                chow_counts[(suit, num)] = chow_counts.get((suit, num), 0) + 1
        elif kind == "pung":
            t = tiles[0]
            suit, num = tile_suit_num(t)
            if suit in ["wan", "tong", "sok"] and num is not None:
                pung_counts[(suit, num)] = pung_counts.get((suit, num), 0) + 1

    return chow_counts, pung_counts



def collect_meld_decompositions(counts, melds=None, limit=80):
    if melds is None:
        melds = []

    if not counts:
        return [melds]

    if len(melds) >= 6:
        return []

    tile = sorted(counts.keys(), key=tile_sort_key)[0]
    results = []

    if counts.get(tile, 0) >= 3:
        new_counts = dict(counts)
        new_counts[tile] -= 3
        if new_counts[tile] == 0:
            del new_counts[tile]
        results.extend(collect_meld_decompositions(new_counts, melds + [("pung", [tile, tile, tile])], limit))
        if len(results) >= limit:
            return results[:limit]

    suit, num = tile_suit_num(tile)
    if suit in ["wan", "tong", "sok"] and num is not None and num <= 7:
        t2 = f"{num+1}_{suit}"
        t3 = f"{num+2}_{suit}"
        if counts.get(t2, 0) > 0 and counts.get(t3, 0) > 0:
            new_counts = dict(counts)
            for t in [tile, t2, t3]:
                new_counts[t] -= 1
                if new_counts[t] == 0:
                    del new_counts[t]
            results.extend(collect_meld_decompositions(new_counts, melds + [("chow", [tile, t2, t3])], limit))
            if len(results) >= limit:
                return results[:limit]

    return results[:limit]


def collect_standard_decompositions(counts, limit=120):
    results = []
    total = sum(counts.values())
    if total % 3 != 2:
        return results

    for pair_tile, c in sorted(counts.items(), key=lambda x: tile_sort_key(x[0])):
        if c >= 2:
            remaining = dict(counts)
            remaining[pair_tile] -= 2
            if remaining[pair_tile] == 0:
                del remaining[pair_tile]

            for melds in collect_meld_decompositions(remaining, [], limit):
                results.append({"pair": pair_tile, "melds": melds})
                if len(results) >= limit:
                    return results

    return results


def detect_four_return_patterns(counts):
    """
    V34:
    Detect and count multiple 四歸 patterns.

    四歸一 = 5番 each:
      same tile appears 4 times and is used as pung + chow, e.g. 111 + 123.

    四歸二 = 10番 each:
      same tile appears 4 times and two are used as the pair/eye.

    四歸四 = 20番 each:
      same tile appears 4 times and all four are used in chows.

    Returns grouped display items:
      四歸一
      四歸一 x 2
      四歸二 x 2
      etc.
    """
    decomps = collect_standard_decompositions(counts)

    # per physical tile value, keep its best 四歸 type across all decompositions
    best_by_tile = {}
    rank = {"四歸一": 1, "四歸二": 2, "四歸四": 3}
    fan_value = {"四歸一": 5, "四歸二": 10, "四歸四": 20}

    for decomp in decomps:
        pair_tile = decomp.get("pair")
        melds = decomp.get("melds", [])

        for t, c in counts.items():
            if c < 4:
                continue

            pung_uses = 0
            chow_uses = 0

            for kind, tiles in melds:
                if kind == "pung" and tiles and tiles[0] == t:
                    pung_uses += 3
                elif kind == "chow":
                    chow_uses += sum(1 for x in tiles if x == t)

            pattern = None
            if pair_tile == t:
                pattern = "四歸二"
            elif chow_uses >= 4 and pung_uses == 0:
                pattern = "四歸四"
            elif pung_uses >= 3 and chow_uses >= 1:
                pattern = "四歸一"

            if pattern:
                old = best_by_tile.get(t)
                if old is None or rank[pattern] > rank[old]:
                    best_by_tile[t] = pattern

    # Count how many tiles fall under each 四歸 type.
    grouped = {}
    for tile, pattern in best_by_tile.items():
        grouped[pattern] = grouped.get(pattern, 0) + 1

    output = []
    for pattern in ["四歸一", "四歸二", "四歸四"]:
        count = grouped.get(pattern, 0)
        if count <= 0:
            continue

        if count == 1:
            output.append((pattern, fan_value[pattern]))
        else:
            output.append((f"{pattern} x {count}", fan_value[pattern] * count))

    return output




def detect_tw_thirteen_orphans(counts, main_tiles):
    required = [
        "1_wan", "9_wan", "1_tong", "9_tong", "1_sok", "9_sok",
        "east", "south", "west", "north", "red", "green", "white"
    ]
    return (
        len(main_tiles) >= 14
        and all(counts.get(t, 0) >= 1 for t in required)
        and any(counts.get(t, 0) >= 2 for t in required)
    )


def is_nonconnected_three_numbers(nums):
    nums = sorted(nums)
    if len(nums) != 3 or len(set(nums)) != 3:
        return False
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(nums[i] - nums[j]) <= 2:
                return False
    return True


def detect_sixteen_no_connection(counts, main_tiles):
    """
    十六不搭 = 40.

    Practical detector:
    - all seven honor kinds 東南西北中發白 are present
    - at least one pair exists
    - each number suit ideally has 3 isolated numbers
    - tolerant mode allows at least two valid isolated suit-groups and enough total number tiles
      because recognition/correction examples may have one duplicated/misaligned suit tile.
    """
    honors = ["east", "south", "west", "north", "red", "green", "white"]
    if not all(counts.get(t, 0) >= 1 for t in honors):
        return False

    if not any(v >= 2 for v in counts.values()):
        return False

    isolated_suit_groups = 0
    total_number_tiles = 0

    for suit in ["wan", "tong", "sok"]:
        nums = []
        for n in range(1, 10):
            nums.extend([n] * counts.get(f"{n}_{suit}", 0))

        total_number_tiles += len(nums)
        unique_nums = sorted(set(nums))

        ok = False
        for a in range(len(unique_nums)):
            for b in range(a + 1, len(unique_nums)):
                for c in range(b + 1, len(unique_nums)):
                    if is_nonconnected_three_numbers([unique_nums[a], unique_nums[b], unique_nums[c]]):
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break

        if ok:
            isolated_suit_groups += 1

    if isolated_suit_groups >= 3:
        return True

    if isolated_suit_groups >= 2 and total_number_tiles >= 8:
        return True

    return False


def count_flower_sets(flower_tiles):
    season_set = {"flower_spring", "flower_summer", "flower_autumn", "flower_winter"}
    plant_set = {"flower_plum", "flower_orchid", "flower_chrysanthemum", "flower_bamboo"}
    flower_set = set(flower_tiles)
    count = 0
    if season_set.issubset(flower_set):
        count += 1
    if plant_set.issubset(flower_set):
        count += 1
    return count

def add_tw_detectable_patterns(result, counts, main_tiles, flower_tiles, decomposition, extra_options=None):
    """
    V19:
    Applies values from the user-provided 番數 table for rules we can detect from final tiles.
    Manual/open-state rules are handled through Taiwan toggle buttons.
    """
    fan_items = result.setdefault("fan_items", [])
    existing = {item["name"] for item in fan_items}
    total_holder = [int(result.get("total_fan", 0))]

    if extra_options is None:
        extra_options = {}
    if not isinstance(extra_options, dict):
        extra_options = {}

    if not main_tiles:
        result["total_fan"] = total_holder[0]
        return result

    suit_result = detect_suit_pattern(main_tiles)
    has_honor = any(is_honor_tile(t) for t in main_tiles)

    suits = set()
    for t in main_tiles:
        suit, num = tile_suit_num(t)
        if suit in ["wan", "tong", "sok"]:
            suits.add(suit)

    # Color / suit patterns from table
    if suit_result == "pure":
        add_tw_item(fan_items, total_holder, existing, "清一色", 80)
    elif suit_result == "mixed":
        add_tw_item(fan_items, total_holder, existing, "混一色", 30)

    if len(suits) == 2:
        add_tw_item(fan_items, total_holder, existing, "缺一門", 5)

    # No honors / no flowers
    if not has_honor:
        add_tw_item(fan_items, total_holder, existing, "無字", 1)
        if not flower_tiles:
            add_tw_item(fan_items, total_holder, existing, "無字花", 5)

    # 斷么 / 混么 / 清么
    if all(is_simple_tile(t) for t in main_tiles):
        add_tw_item(fan_items, total_holder, existing, "斷么", 5)

    # 混么 / 清么:
    # 清么 = all terminals only, no honor tiles.
    # 混么 = terminals + honor tiles. It should include 番子 and should not stack with 清么.
    all_terminal_or_honor = all(is_terminal_tile(t) or is_honor_tile(t) for t in main_tiles)
    has_honor_tile = any(is_honor_tile(t) for t in main_tiles)
    all_terminal_only = all(is_terminal_tile(t) for t in main_tiles)

    if all_terminal_only and not has_honor_tile:
        add_tw_item(fan_items, total_holder, existing, "清么", 80)
    elif all_terminal_or_honor and has_honor_tile:
        add_tw_item(fan_items, total_holder, existing, "混么", 30)

    # Dragons
    dragon_pungs = sum(1 for d in ["red", "green", "white"] if counts.get(d, 0) >= 3)
    dragon_pairs = sum(1 for d in ["red", "green", "white"] if counts.get(d, 0) == 2)

    if dragon_pungs == 3:
        add_tw_item(fan_items, total_holder, existing, "大三元", 40)
    elif dragon_pungs == 2 and dragon_pairs >= 1:
        add_tw_item(fan_items, total_holder, existing, "小三元", 20)

    for d in ["red", "green", "white"]:
        if counts.get(d, 0) >= 3:
            add_tw_item(fan_items, total_holder, existing, f"碰出{DISPLAY_NAMES.get(d, d)}", 2)

    # Winds
    wind_pungs = sum(1 for w in ["east", "south", "west", "north"] if counts.get(w, 0) >= 3)
    wind_pairs = sum(1 for w in ["east", "south", "west", "north"] if counts.get(w, 0) == 2)

    if wind_pungs == 4:
        add_tw_item(fan_items, total_holder, existing, "大四喜", 80)
    elif wind_pungs == 3 and wind_pairs >= 1:
        add_tw_item(fan_items, total_holder, existing, "小四喜", 60)
    elif wind_pungs == 3:
        add_tw_item(fan_items, total_holder, existing, "大三風", 30)
    elif wind_pungs == 2 and wind_pairs >= 1:
        add_tw_item(fan_items, total_holder, existing, "小三風", 15)

    if decomposition:
        melds = decomposition.get("melds", [])
        all_pungs = bool(melds) and all(m[0] == "pung" for m in melds)
        all_chows = bool(melds) and all(m[0] == "chow" for m in melds)

        if all_pungs:
            add_tw_item(fan_items, total_holder, existing, "對對胡", 30)

            if bool(extra_options.get("concealed_self_draw", False)) or (
                bool(extra_options.get("self_draw", False)) and bool(extra_options.get("concealed", False))
            ):
                add_tw_item(fan_items, total_holder, existing, "間間胡", 100)

        if all_chows:
            add_tw_item(fan_items, total_holder, existing, "平胡", 3)

        pair = decomposition.get("pair")
        suit, num = tile_suit_num(pair) if pair else (None, None)
        if num in [2, 5, 8]:
            add_tw_item(fan_items, total_holder, existing, "將眼", 1)

        chow_counts, pung_counts = count_meld_patterns_from_decomposition(decomposition)

        # 四歸 rules.
        for name, fan in detect_four_return_patterns(counts):
            add_tw_item(fan_items, total_holder, existing, name, fan)


        # 老少 is counted later by direct count detection so multiple groups can be counted.

        # 兄弟刻子:
        # 二兄弟: two pungs with same number in different suits = 3番
        # 小三兄弟: two pungs + same-number pair as eye = 10番
        # 大三兄弟: three pungs with same number in three suits = 15番
        for n in range(1, 10):
            pung_suits = [s for s in ["wan", "tong", "sok"] if counts.get(f"{n}_{s}", 0) >= 3]
            pair_suits = [s for s in ["wan", "tong", "sok"] if counts.get(f"{n}_{s}", 0) == 2]

            if len(pung_suits) >= 3:
                add_tw_item(fan_items, total_holder, existing, "大三兄弟", 15)
                break
            elif len(pung_suits) >= 2 and pair_suits:
                add_tw_item(fan_items, total_holder, existing, "小三兄弟", 10)
                break
            elif len(pung_suits) >= 2:
                add_tw_item(fan_items, total_holder, existing, "二兄弟", 3)
                break

        # 姊妹刻子:
        # 小三姊妹: same suit, two consecutive-number pungs + one connected pair = 8番
        # 大三姊妹: same suit, three consecutive-number pungs = 15番
        sister_found = False
        for s in ["wan", "tong", "sok"]:
            for n in range(1, 8):
                c1 = counts.get(f"{n}_{s}", 0)
                c2 = counts.get(f"{n+1}_{s}", 0)
                c3 = counts.get(f"{n+2}_{s}", 0)

                pung_count = sum(1 for c in [c1, c2, c3] if c >= 3)
                pair_count = sum(1 for c in [c1, c2, c3] if c == 2)

                if pung_count == 3:
                    add_tw_item(fan_items, total_holder, existing, "大三姊妹", 15)
                    sister_found = True
                    break
                elif pung_count == 2 and pair_count >= 1:
                    add_tw_item(fan_items, total_holder, existing, "小三姊妹", 8)
                    sister_found = True
                    break

            if sister_found:
                break

        # Related same-sequence rules:
        # 四同順 / 五同順 suppress 三相逢 / 一般高 / 三般高 / 四般高.
        # User table:
        # 四同順: 四個數字一樣順子，不論款式，不再計三相逢/一般高/兩般高/三般高
        # 五同順: 五個數字一樣順子，不論款式，不再計三相逢/一般高/兩般高/三般高
        max_same_number_chow = 0
        for n in range(1, 8):
            total_same_number_chow = sum(chow_counts.get((s, n), 0) for s in ["wan", "tong", "sok"])
            max_same_number_chow = max(max_same_number_chow, total_same_number_chow)

        if max_same_number_chow >= 5:
            add_tw_item(fan_items, total_holder, existing, "五同順", 40)
        elif max_same_number_chow >= 4:
            add_tw_item(fan_items, total_holder, existing, "四同順", 20)
        else:
            # Only count these lower rules if 四同順 / 五同順 is not present.
            max_same_chow = max(chow_counts.values()) if chow_counts else 0
            if max_same_chow >= 4:
                add_tw_item(fan_items, total_holder, existing, "四般高", 30)
            elif max_same_chow >= 3:
                add_tw_item(fan_items, total_holder, existing, "三般高", 15)
            elif max_same_chow >= 2:
                add_tw_item(fan_items, total_holder, existing, "一般高", 3)

            # 二相逢 / 三相逢: same-number chow in different suits
            max_same_num_suits = 0
            for n in range(1, 8):
                suits_for_n = [s for s in ["wan", "tong", "sok"] if chow_counts.get((s, n), 0) > 0]
                max_same_num_suits = max(max_same_num_suits, len(suits_for_n))
            if max_same_num_suits >= 3:
                add_tw_item(fan_items, total_holder, existing, "三相逢", 10)
            elif max_same_num_suits >= 2:
                add_tw_item(fan_items, total_holder, existing, "二相逢", 2)


        # 龍 / 雜龍:
        # Same suit has 1-9:
        #   暗龍 button OFF -> 明龍 10番
        #   暗龍 button ON  -> 暗龍 20番
        #
        # Mixed suits have 1-9:
        #   暗龍 button OFF -> 明雜龍 8番
        #   暗龍 button ON  -> 暗雜龍 15番
        #
        # The final photo can detect the 1-9 shape, but cannot know open/closed.
        dark_dragon_selected = bool(extra_options.get("dark_dragon", False))

        pure_dragon_found = False
        for s in ["wan", "tong", "sok"]:
            if all(counts.get(f"{n}_{s}", 0) > 0 for n in range(1, 10)):
                pure_dragon_found = True
                if dark_dragon_selected:
                    add_tw_item(fan_items, total_holder, existing, "暗龍", 20)
                else:
                    add_tw_item(fan_items, total_holder, existing, "明龍", 10)
                break

        # Only check 雜龍 if it is not already a pure same-suit 龍.
        if not pure_dragon_found:
            dragon_number_suits = []
            mixed_dragon_ok = True
            for n in range(1, 10):
                suits_for_n = [s for s in ["wan", "tong", "sok"] if counts.get(f"{n}_{s}", 0) > 0]
                if not suits_for_n:
                    mixed_dragon_ok = False
                    break
                dragon_number_suits.append(suits_for_n[0])

            if mixed_dragon_ok and len(set(dragon_number_suits)) >= 2:
                if dark_dragon_selected:
                    add_tw_item(fan_items, total_holder, existing, "暗雜龍", 15)
                else:
                    add_tw_item(fan_items, total_holder, existing, "明雜龍", 8)

        # 五門齊
        has_wind = any(counts.get(w, 0) > 0 for w in ["east", "south", "west", "north"])
        has_dragon = any(counts.get(d, 0) > 0 for d in ["red", "green", "white"])
        if has_wind and has_dragon and all(s in suits for s in ["wan", "tong", "sok"]):
            add_tw_item(fan_items, total_holder, existing, "五門齊", 10)

    # 老少 direct count detection:
    # Same suit 123 + 789 OR 111 + 999.
    # Count multiple 老少 groups separately: 老少 x 2 = 4番.
    lao_shao_count = 0
    for s in ["wan", "tong", "sok"]:
        has_123_789_by_count = all(counts.get(f"{n}_{s}", 0) >= 1 for n in [1, 2, 3, 7, 8, 9])
        has_111_999_by_count = counts.get(f"1_{s}", 0) >= 3 and counts.get(f"9_{s}", 0) >= 3

        # If both forms happen in the same suit, count it once for that suit.
        if has_123_789_by_count or has_111_999_by_count:
            lao_shao_count += 1

    if lao_shao_count > 0:
        if lao_shao_count == 1:
            add_tw_item(fan_items, total_holder, existing, "老少", 2)
        else:
            add_tw_item(fan_items, total_holder, existing, f"老少 x {lao_shao_count}", 2 * lao_shao_count)

    # 明摃 / 暗摃.
    if not bool(extra_options.get("closed_kong", False)):
        kong_count = sum(1 for v in counts.values() if v >= 4)
        if kong_count == 1:
            add_tw_item(fan_items, total_holder, existing, "明摃", 1)
        elif kong_count > 1:
            add_tw_item(fan_items, total_holder, existing, f"明摃 x {kong_count}", kong_count)

    # Special hands.
    if detect_tw_thirteen_orphans(counts, main_tiles):
        add_tw_item(fan_items, total_holder, existing, "十三么", 80)

    if detect_sixteen_no_connection(counts, main_tiles):
        add_tw_item(fan_items, total_holder, existing, "十六不搭", 40)

    # 嚦咕嚦咕: eight pairs style in 17-tile Mahjong.
    if len(main_tiles) == 17:
        pairs = sum(v // 2 for v in counts.values())
        if pairs >= 8:
            add_tw_item(fan_items, total_holder, existing, "嚦咕嚦咕", 40)

    result["total_fan"] = total_holder[0]
    return result



def normalize_lao_shao_fan(result):
    """
    V28 safety fix:
    Ensure 老少 x N always has fan = 2 * N.
    Then recalculate total_fan from all fan_items.
    """
    try:
        for item in result.get("fan_items", []):
            name = str(item.get("name", ""))
            m = re.search(r"老少\s*x\s*(\d+)", name)
            if m:
                count = int(m.group(1))
                item["fan"] = 2 * count
            elif name == "老少":
                item["fan"] = 2

        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_flower_sets(result):
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}
        if ("一台花" in names) or ("兩台花/花胡" in names):
            result["fan_items"] = [
                item for item in items
                if not (
                    str(item.get("name", "")).startswith("正花 ")
                    or str(item.get("name", "")).startswith("爛花 ")
                    or str(item.get("name", "")) == "無花"
                )
            ]
        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_all_self_draw_pung(result):
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}
        if "間間胡" in names:
            suppressed = {"對對胡", "門清自摸", "自摸", "門清"}
            result["fan_items"] = [item for item in items if str(item.get("name", "")) not in suppressed]
        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_kong_items(result):
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}
        if "暗摃" in names:
            result["fan_items"] = [
                item for item in items
                if not str(item.get("name", "")).startswith("明摃")
            ]
        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_yao_patterns(result):
    """
    V40 safety fix:
    清么 and 混么 are large/top terminal patterns.

    If 清么 exists, suppress smaller structure patterns naturally included:
      混么
      對對胡
      二兄弟 / 小三兄弟 / 大三兄弟
      老少 / 老少 x N
      斷么

    If 混么 exists, suppress 對對胡 because 混么 naturally forms pungs/pair
    from terminals + honors and should not stack with 對對胡.
    """
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}

        if "清么" in names:
            suppressed_exact = {
                "混么",
                "對對胡",
                "二兄弟",
                "小三兄弟",
                "大三兄弟",
                "老少",
                "斷么",
            }

            result["fan_items"] = [
                item for item in items
                if (
                    str(item.get("name", "")) not in suppressed_exact
                    and not str(item.get("name", "")).startswith("老少 x")
                )
            ]

        elif "混么" in names:
            result["fan_items"] = [
                item for item in items
                if str(item.get("name", "")) != "對對胡"
            ]

        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_sequence_exclusions(result):
    """
    V30 safety fix:
    If 四同順 or 五同順 exists, remove lower related sequence items:
    一般高 / 三般高 / 四般高 / 二相逢 / 三相逢.
    """
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}
        has_big_same_chow = ("四同順" in names) or ("五同順" in names)

        if has_big_same_chow:
            suppressed = {"一般高", "三般高", "四般高", "二相逢", "三相逢"}
            result["fan_items"] = [
                item for item in items
                if str(item.get("name", "")) not in suppressed
            ]

        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_brother_pungs(result):
    """
    V31 safety fix:
    大三兄弟 suppresses 小三兄弟 / 二兄弟.
    小三兄弟 suppresses 二兄弟.
    """
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}

        if "大三兄弟" in names:
            suppressed = {"小三兄弟", "二兄弟"}
        elif "小三兄弟" in names:
            suppressed = {"二兄弟"}
        else:
            suppressed = set()

        if suppressed:
            result["fan_items"] = [
                item for item in items
                if str(item.get("name", "")) not in suppressed
            ]

        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def normalize_sister_pungs(result):
    """
    V32 safety fix:
    大三姊妹 suppresses 小三姊妹.
    """
    try:
        items = result.get("fan_items", [])
        names = {str(item.get("name", "")) for item in items}

        if "大三姊妹" in names:
            result["fan_items"] = [
                item for item in items
                if str(item.get("name", "")) != "小三姊妹"
            ]

        result["total_fan"] = sum(int(item.get("fan", 0)) for item in result.get("fan_items", []))
    except Exception:
        pass
    return result


def score_tw_simple(
    tiles,
    self_draw=False,
    concealed=False,
    flower_count=0,
    seat_wind="none",
    round_wind="east",
    extra_options=None,
):
    """
    V19 Taiwan / copied 番數 table scoring mode.

    Uses the user's provided 番數 values where possible.
    Some rules need manual buttons because the final tile photo does not tell open/closed/winning method details.
    """
    fan_items = []
    notes = []

    tiles = [t for t in tiles if t and t != "unknown"]
    flower_tiles = [t for t in tiles if t.startswith("flower_")]
    main_tiles = [t for t in tiles if not t.startswith("flower_")]

    counts = {}
    for t in main_tiles:
        counts[t] = counts.get(t, 0) + 1

    total_holder = [0]
    existing = set()

    if isinstance(extra_options, str):
        try:
            extra_options = json.loads(extra_options)
        except Exception:
            extra_options = {}
    if not isinstance(extra_options, dict):
        extra_options = {}

    # Flower rules from provided table.
    flower_by_seat = {
        "east": {"flower_spring", "flower_plum"},
        "south": {"flower_summer", "flower_orchid"},
        "west": {"flower_autumn", "flower_chrysanthemum"},
        "north": {"flower_winter", "flower_bamboo"},
    }
    seat_flower_set = flower_by_seat.get(seat_wind, set())
    matching_flowers = [t for t in flower_tiles if t in seat_flower_set]
    non_matching_flowers = [t for t in flower_tiles if t not in seat_flower_set]

    if not flower_tiles:
        add_tw_item(fan_items, total_holder, existing, "無花", 1)
    for t in matching_flowers:
        add_tw_item(fan_items, total_holder, existing, f"正花 {DISPLAY_NAMES.get(t, t)}", 2)
    for t in non_matching_flowers:
        add_tw_item(fan_items, total_holder, existing, f"爛花 {DISPLAY_NAMES.get(t, t)}", 1)

    # 一台花 / 兩台花.
    flower_set_count = count_flower_sets(flower_tiles)
    if flower_set_count >= 2:
        add_tw_item(fan_items, total_holder, existing, "兩台花/花胡", 30)
    elif flower_set_count == 1:
        add_tw_item(fan_items, total_holder, existing, "一台花", 10)

    # Manual option values from table.
    manual_table = {
        "ready": ("聽牌/叮", 5),
        "pair_wait": ("對碰", 1),
        "fake_single_wait": ("假獨", 1),
        "single_wait": ("獨獨", 2),
        "concealed": ("門清", 3),
        "self_draw": ("自摸", 1),
        "concealed_self_draw": ("門清自摸", 5),
        "last_tile": ("海底撈月", 20),
        "flower_self_draw": ("花上食胡", 1),
        "kong_draw": ("摃上食胡", 1),
        "rob_kong": ("搶摃食胡", 1),
        "double_kong_draw": ("摃上摃食胡", 30),
        "rob_double_kong": ("搶摃上摃食胡", 30),
        "all_from_others": ("全求人", 15),
        "half_from_others": ("半求人", 8),
        "within_7": ("七只內", 20),
        "within_10": ("十只內", 10),
        "heaven_win": ("天胡", 100),
        "earth_win": ("地胡", 80),
        "human_win": ("人胡", 80),
        "closed_kong": ("暗摃", 2),
        "two_concealed_pungs": ("二暗刻", 3),
        "three_concealed_pungs": ("三暗刻", 10),
        "four_concealed_pungs": ("四暗刻", 30),
        "five_concealed_pungs": ("五暗刻", 80),
        "all_self_draw_pung": ("間間胡", 100),
        "open_mixed_dragon": ("明雜龍", 8),
        "dark_mixed_dragon": ("暗雜龍", 15),
        "heaven_ready": ("天聽/地聽", 2),
    }

    # V29 combination rule:
    # 門清 = 3, 自摸 = 1, but 門清 + 自摸 should count as 門清自摸 = 5 only.
    # Do not count 門清 and 自摸 separately when both are selected.
    if self_draw:
        extra_options["self_draw"] = True
    if concealed:
        extra_options["concealed"] = True

    if bool(extra_options.get("self_draw", False)) and bool(extra_options.get("concealed", False)):
        extra_options["concealed_self_draw"] = True
        extra_options["self_draw"] = False
        extra_options["concealed"] = False

    for key, (name, fan) in manual_table.items():
        if bool(extra_options.get(key, False)):
            add_tw_item(fan_items, total_holder, existing, name, fan)

    standard_win, decomposition = find_standard_win_decomposition(counts)
    is_win = bool(len(main_tiles) == 17 and standard_win)

    if len(main_tiles) == 17 and standard_win:
        notes.append("17-tile winning structure detected: 5 melds + 1 pair.")
    elif len(main_tiles) != 17:
        notes.append(f"Taiwan mode normally expects 17 main tiles. Current main tile count: {len(main_tiles)}.")
    else:
        notes.append("17 tiles detected, but winning structure not confirmed.")

    result = {
        "is_win": is_win,
        "fan_items": fan_items,
        "total_fan": total_holder[0],
        "notes": notes,
        "rule_set": "tw",
        "unit": "番",
    }

    result = add_tw_detectable_patterns(result, counts, main_tiles, flower_tiles, decomposition if standard_win else None, extra_options=extra_options)

    # 雞胡: only when it is a winning hand and the total is very small.
    if is_win and result["total_fan"] <= 1:
        holder = [result["total_fan"]]
        existing2 = {item["name"] for item in result["fan_items"]}
        add_tw_item(result["fan_items"], holder, existing2, "雞胡", 10)
        result["total_fan"] = holder[0]

    result = normalize_lao_shao_fan(result)
    result = normalize_flower_sets(result)
    result = normalize_yao_patterns(result)
    result = normalize_sequence_exclusions(result)
    result = normalize_brother_pungs(result)
    result = normalize_sister_pungs(result)
    result = normalize_all_self_draw_pung(result)
    result = normalize_kong_items(result)
    return result


# ------------------------------------------------------------
# Simple Hong Kong-style scoring functions
# ------------------------------------------------------------
def score_hk_simple(
    tiles,
    self_draw=False,
    concealed=False,
    flower_count=0,
    seat_wind="none",
    round_wind="east",
):
    fan_items = []
    notes = []

    tiles = [t for t in tiles if t and t != "unknown"]

    flower_tiles = [t for t in tiles if t.startswith("flower_")]
    main_tiles = [t for t in tiles if not t.startswith("flower_")]

    # V6 flower rule:
    # Flowers only give fan when they match Seat Wind.
    # East:  春 / 梅
    # South: 夏 / 蘭
    # West:  秋 / 菊
    # North: 冬 / 竹
    flower_by_seat = {
        "east": {"flower_spring", "flower_plum"},
        "south": {"flower_summer", "flower_orchid"},
        "west": {"flower_autumn", "flower_chrysanthemum"},
        "north": {"flower_winter", "flower_bamboo"},
    }

    seat_flower_set = flower_by_seat.get(seat_wind, set())
    matching_flower_tiles = [t for t in flower_tiles if t in seat_flower_set]
    non_matching_flower_tiles = [t for t in flower_tiles if t not in seat_flower_set]

    try:
        manual_flower_count = int(flower_count or 0)
    except Exception:
        manual_flower_count = 0

    counts = {}
    for t in main_tiles:
        counts[t] = counts.get(t, 0) + 1

    total_fan = 0

    if self_draw:
        fan_items.append({"name": "自摸", "fan": 1})
        total_fan += 1

    if concealed:
        fan_items.append({"name": "門清", "fan": 1})
        total_fan += 1

    if matching_flower_tiles:
        display_flowers = " ".join(DISPLAY_NAMES.get(t, t) for t in matching_flower_tiles)
        fan_items.append({"name": f"正花 {display_flowers}", "fan": len(matching_flower_tiles)})
        total_fan += len(matching_flower_tiles)

    if flower_tiles and non_matching_flower_tiles:
        display_non = " ".join(DISPLAY_NAMES.get(t, t) for t in non_matching_flower_tiles)
        notes.append(f"Non-matching flowers not counted: {display_non}")

    if manual_flower_count > 0 and not flower_tiles:
        notes.append("Flower Count is used for expected box count only. Flower fan needs actual flower tile labels or manual correction.")

    seven_pairs = is_seven_pairs(counts)
    thirteen_orphans = is_thirteen_orphans(counts)

    if thirteen_orphans:
        fan_items.append({"name": "十三么", "fan": 8})
        total_fan += 8
        return {
            "is_win": True,
            "fan_items": fan_items,
            "total_fan": total_fan,
            "notes": notes + ["十三么 detected as special hand."]
        }

    if seven_pairs:
        fan_items.append({"name": "七對子", "fan": 4})
        total_fan += 4

    standard_win, decomposition = find_standard_win_decomposition(counts)
    is_win = bool(seven_pairs or standard_win)

    if not is_win:
        notes.append("Cannot confirm standard winning structure yet. Score may be incomplete.")
        notes.append("If the hand includes open melds, later version should enter melds manually.")

    suit_result = detect_suit_pattern(main_tiles)
    if suit_result == "pure":
        fan_items.append({"name": "清一色", "fan": 7})
        total_fan += 7
    elif suit_result == "mixed":
        fan_items.append({"name": "混一色", "fan": 3})
        total_fan += 3

    dragon_pungs = 0
    dragon_pairs = 0
    for d in ["red", "green", "white"]:
        if counts.get(d, 0) >= 3:
            dragon_pungs += 1
        elif counts.get(d, 0) == 2:
            dragon_pairs += 1

    if dragon_pungs == 3:
        fan_items.append({"name": "大三元", "fan": 8})
        total_fan += 8
    elif dragon_pungs == 2 and dragon_pairs >= 1:
        fan_items.append({"name": "小三元", "fan": 5})
        total_fan += 5
    else:
        for d in ["red", "green", "white"]:
            if counts.get(d, 0) >= 3:
                fan_items.append({"name": f"番牌刻子 {DISPLAY_NAMES.get(d, d)}", "fan": 1})
                total_fan += 1

    wind_label = {"east": "東", "south": "南", "west": "西", "north": "北"}

    if seat_wind in ["east", "south", "west", "north"] and counts.get(seat_wind, 0) >= 3:
        fan_items.append({"name": f"門風刻 {wind_label.get(seat_wind, seat_wind)}", "fan": 1})
        total_fan += 1

    if round_wind in ["east", "south", "west", "north"] and counts.get(round_wind, 0) >= 3:
        fan_items.append({"name": f"圈風刻 {wind_label.get(round_wind, round_wind)}", "fan": 1})
        total_fan += 1

    if standard_win and decomposition:
        melds = decomposition["melds"]

        all_pungs = all(m[0] == "pung" for m in melds)
        all_chows = all(m[0] == "chow" for m in melds)

        if all_pungs:
            fan_items.append({"name": "對對糊 / 碰碰糊", "fan": 3})
            total_fan += 3

        if all_chows:
            fan_items.append({"name": "平糊", "fan": 1})
            total_fan += 1

    # V17 extra detectable Hong Kong-style patterns.
    existing_names = {item["name"] for item in fan_items}
    holder = [total_fan]
    for name, fan in detect_extra_patterns(counts, main_tiles, decomposition if standard_win else None):
        add_fan_once(fan_items, holder, existing_names, name, fan)
    total_fan = holder[0]

    return {
        "is_win": is_win,
        "fan_items": fan_items,
        "total_fan": total_fan,
        "notes": notes
    }


def is_seven_pairs(counts):
    if sum(counts.values()) != 14:
        return False
    return len(counts) == 7 and all(v == 2 for v in counts.values())


def is_thirteen_orphans(counts):
    if sum(counts.values()) != 14:
        return False

    required = set([
        "1_wan", "9_wan",
        "1_tong", "9_tong",
        "1_sok", "9_sok",
        "east", "south", "west", "north",
        "red", "green", "white",
    ])

    if not required.issubset(set(counts.keys())):
        return False

    return any(counts.get(t, 0) >= 2 for t in required)


def detect_suit_pattern(tiles):
    suits = set()
    has_honor = False

    for t in tiles:
        if t.endswith("_wan"):
            suits.add("wan")
        elif t.endswith("_tong"):
            suits.add("tong")
        elif t.endswith("_sok"):
            suits.add("sok")
        elif t in ["east", "south", "west", "north", "red", "green", "white"]:
            has_honor = True

    if len(suits) == 1 and not has_honor:
        return "pure"
    if len(suits) == 1 and has_honor:
        return "mixed"
    return None


def find_standard_win_decomposition(counts):
    total = sum(counts.values())
    if total % 3 != 2:
        return False, None

    for pair_tile, c in list(counts.items()):
        if c >= 2:
            remaining = dict(counts)
            remaining[pair_tile] -= 2
            if remaining[pair_tile] == 0:
                del remaining[pair_tile]

            success, melds = can_form_melds(remaining, [])
            if success:
                return True, {
                    "pair": pair_tile,
                    "melds": melds
                }

    return False, None


def can_form_melds(counts, melds):
    if not counts:
        return True, melds

    tile = sorted(counts.keys(), key=tile_sort_key)[0]

    if counts.get(tile, 0) >= 3:
        new_counts = dict(counts)
        new_counts[tile] -= 3
        if new_counts[tile] == 0:
            del new_counts[tile]

        ok, result = can_form_melds(new_counts, melds + [("pung", [tile, tile, tile])])
        if ok:
            return True, result

    suit, num = tile_suit_num(tile)
    if suit in ["wan", "tong", "sok"] and num is not None and num <= 7:
        t2 = f"{num+1}_{suit}"
        t3 = f"{num+2}_{suit}"

        if counts.get(t2, 0) > 0 and counts.get(t3, 0) > 0:
            new_counts = dict(counts)
            for t in [tile, t2, t3]:
                new_counts[t] -= 1
                if new_counts[t] == 0:
                    del new_counts[t]

            ok, result = can_form_melds(new_counts, melds + [("chow", [tile, t2, t3])])
            if ok:
                return True, result

    return False, None


def tile_suit_num(tile):
    parts = tile.split("_")
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1], int(parts[0])
    return None, None


def tile_sort_key(tile):
    suit_order = {
        "wan": 0,
        "tong": 1,
        "sok": 2,
        "east": 3,
        "south": 4,
        "west": 5,
        "north": 6,
        "red": 7,
        "green": 8,
        "white": 9,
    }

    suit, num = tile_suit_num(tile)
    if suit is not None:
        return (suit_order.get(suit, 99), num)

    return (suit_order.get(tile, 99), 0)


engine = MahjongEngine()
app = FastAPI(title=APP_TITLE)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/", response_class=HTMLResponse)
def index():
    if WEB_HTML_FILE.exists():
        return WEB_HTML_FILE.read_text(encoding="utf-8")

    return """
    <html>
      <body>
        <h1>Mahjong Server V99</h1>
        <p>mahjong_web_V99.html not found. Please put it beside mahjong_server_V99.py.</p>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "ok": True,
        "yolo_model_exists": YOLO_MODEL_FILE.exists(),
        "ai_model_exists": AI_MODEL_FILE.exists(),
        "yolo_model": str(YOLO_MODEL_FILE),
        "ai_model": str(AI_MODEL_FILE),
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    conf: float = Form(0.10),
    orientation: str = Form("off"),
    side_rotation: bool = Form(True),
    self_draw: bool = Form(False),
    concealed: bool = Form(False),
    flower_count: int = Form(0),
    seat_wind: str = Form("none"),
    round_wind: str = Form("east"),
    rule_set: str = Form("hk"),
    extra_options: str = Form("{}"),
    expected_count: int = Form(0),
    auto_retry: bool = Form(True),
    trim_to_expected: bool = Form(True),
):
    run_dir = OUTPUT_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        suffix = ".jpg"

    input_file = run_dir / f"input{suffix}"

    try:
        with input_file.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = engine.recognize_image(
            input_file,
            conf=conf,
            orientation=orientation,
            side_rotation=side_rotation,
            self_draw=self_draw,
            concealed=concealed,
            flower_count=flower_count,
            seat_wind=seat_wind,
            round_wind=round_wind,
            rule_set=rule_set,
            extra_options=extra_options,
            expected_count=expected_count,
            auto_retry=auto_retry,
            trim_to_expected=trim_to_expected,
        )

        return JSONResponse(result)

    except Exception as e:
        err = traceback.format_exc()
        (run_dir / "error.txt").write_text(err, encoding="utf-8")
        return JSONResponse(
            {
                "ok": False,
                "error": str(e),
                "traceback": err,
            },
            status_code=500
        )


@app.post("/score")
async def score_only(request: Request):
    """
    V5:
    Recalculate fan from manually edited tile labels in the HTML page.
    No image recognition is run here.
    """
    try:
        data = await request.json()

        tiles = data.get("tiles", [])
        if not isinstance(tiles, list):
            tiles = []

        self_draw = bool(data.get("self_draw", False))
        concealed = bool(data.get("concealed", False))

        try:
            flower_count = int(data.get("flower_count", 0) or 0)
        except Exception:
            flower_count = 0

        seat_wind = str(data.get("seat_wind", "none") or "none")
        round_wind = str(data.get("round_wind", "east") or "east")
        rule_set = str(data.get("rule_set", "hk") or "hk")
        extra_options = data.get("extra_options", {})

        score_result = score_by_rule(
            tiles,
            rule_set=rule_set,
            self_draw=self_draw,
            concealed=concealed,
            flower_count=flower_count,
            seat_wind=seat_wind,
            round_wind=round_wind,
            extra_options=extra_options
        )

        return JSONResponse({
            "ok": True,
            "tile_count": len([t for t in tiles if t]),
            "recognized_labels": tiles,
            "recognized_row": [DISPLAY_NAMES.get(t, t) for t in tiles],
            "score": score_result,
        })

    except Exception as e:
        return JSONResponse(
            {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500
        )



if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print(APP_TITLE)
    print(f"Folder: {BASE_DIR}")
    print(f"YOLO model: {YOLO_MODEL_FILE}")
    print(f"AI model: {AI_MODEL_FILE}")
    print("=" * 70)
    print("Open on this PC: http://127.0.0.1:8000")
    print("For phone: open http://YOUR_PC_IP:8000 on same Wi-Fi")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
