import streamlit as st
import pandas as pd
import re
import os
import json
import base64
import html
from datetime import datetime, timedelta
import requests
from io import BytesIO, StringIO

try:
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    LAYOUTLMV3_AVAILABLE = True
except Exception:
    LAYOUTLMV3_AVAILABLE = False
    LayoutLMv3Processor = None
    LayoutLMv3ForTokenClassification = None
    torch = None

# Microsoft Graph API endpoints
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_OVERRIDES_PATH = os.path.join(PROJECT_ROOT, "output", "user_label_overrides.json")

# Import authentication module
from auth import (
    get_auth_url,
    get_token_from_code,
    refresh_access_token,
    get_auth_config_issues,
    is_auth_configured,
)

# Import BERT classifier (optional - falls back to rules if not available)
try:
    from bert_classifier import HybridClassifier, TRANSFORMERS_AVAILABLE
    BERT_AVAILABLE = TRANSFORMERS_AVAILABLE
except ImportError:
    BERT_AVAILABLE = False
    HybridClassifier = None

# OCR extraction dependencies (optional)
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PDF extraction dependency (optional)
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

SUPPORTED_ATTACHMENT_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.pdf'
}

if OCR_AVAILABLE:
    tesseract_exe = os.path.join(PROJECT_ROOT, "tessaret", "tesseract.exe")
    if os.path.exists(tesseract_exe):
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe


# ============== PO Classification Functions ==============

# Negative keywords to filter out non-PO emails
NEGATIVE_KEYWORDS = [
    'newsletter', 'unsubscribe', 'meeting invite', 'calendar invite',
    'out of office', 'automatic reply', 'auto-reply', 'linkedin', 
    'facebook', 'promotional', 'advertisement', 'sale offer',
    'webinar', 'survey', 'feedback request', 'password reset',
    'verify your email', 'account notification', 'social media',
    'job alert', 'daily digest', 'weekly summary'
]

PO_KEYWORDS = [
    'purchase order', 'po#', 'po number', 'p.o.', 'p.o', 'po:',
    'order confirmation', 'order acknowledgment', 'order acknowledgement',
    'order placed', 'new order', 'order details', 'order number',
    'procurement', 'requisition', 'indent', 'supply order',
    'packing', 'shipment', 'delivery', 'dispatch', 'trims',
    'please confirm', 'kindly confirm', 'attached po', 'attached purchase order',
    'mel2025po', 'mel2024po', 'mel2026po', 'price sticker', 'carton sticker'
]

PO_NUMBER_PATTERNS = [
    r'\bP[O0]\s*#?\s*[:\-]?\s*([A-Z0-9-]*\d[A-Z0-9-]*)\b',
    r'MEL\d{4}PO\d+',
    r'[A-Z]{2,4}\d{4}PO\d+',
    r'\bPurchase\s*Order\s*#?\s*[:\-]?\s*([A-Z0-9-]*\d[A-Z0-9-]*)\b',
]

MASTER_COLUMNS = [
    'Type',
    'Contract No',
    'Item Category',
    '5lb',
    'First Size',
    'Up To 1Mth',
    'Up To 3Mth',
    '3-6 Mths',
    '6-9 Mths',
    '9-12 Mths',
    '12-18 Mths',
    '1.5-2 Yrs',
    'Total',
]

MASTER_TEXT_COLUMNS = {'Type', 'Contract No', 'Item Category'}
MASTER_NUMERIC_COLUMNS = [col for col in MASTER_COLUMNS if col not in MASTER_TEXT_COLUMNS]

MASTER_SIZE_ALIASES = {
    '5lb': {'5lb', '5 lb'},
    'First Size': {'first size'},
    'Up To 1Mth': {'up to 1mth', 'up to 1 mth', 'upto 1mth', 'up to 1 month'},
    'Up To 3Mth': {'up to 3mth', 'up to 3 mth', 'upto 3mth', 'up to 3 month'},
    '3-6 Mths': {'3-6 mths', '3 6 mths', '3-6 months', '3 to 6 mths'},
    '6-9 Mths': {'6-9 mths', '6 9 mths', '6-9 months', '6 to 9 mths'},
    '9-12 Mths': {'9-12 mths', '9 12 mths', '9-12 months', '9 to 12 mths'},
    '12-18 Mths': {'12-18 mths', '12 18 mths', '12-18 months', '12 to 18 mths'},
    '1.5-2 Yrs': {'1.5-2 yrs', '15-2 yrs', '1.5 2 yrs', '1.5-2 years'},
    'Total': {'total'},
}

_LAYOUTLMV3_CACHE = {
    'loaded': False,
    'processor': None,
    'model': None,
}


def _empty_master_row():
    row = {col: '' for col in MASTER_TEXT_COLUMNS}
    for col in MASTER_NUMERIC_COLUMNS:
        row[col] = None
    return row


def _canonical_master_size_key(raw_key):
    normalized = re.sub(r'\s+', ' ', str(raw_key or '').strip().lower())
    normalized = normalized.replace('.', '')
    normalized = normalized.replace('_', ' ')
    normalized = normalized.replace('months', 'mths').replace('month', 'mth')
    normalized = normalized.replace('years', 'yrs').replace('year', 'yrs')
    for canonical, aliases in MASTER_SIZE_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _get_layoutlmv3_components(debug=False):
    if _LAYOUTLMV3_CACHE['loaded']:
        return _LAYOUTLMV3_CACHE['processor'], _LAYOUTLMV3_CACHE['model']

    _LAYOUTLMV3_CACHE['loaded'] = True
    if not LAYOUTLMV3_AVAILABLE:
        log_body_debug(debug, "layoutlmv3_unavailable transformers_or_torch_missing")
        return None, None

    model_path = os.getenv('LAYOUTLMV3_MODEL_PATH', '').strip()
    if not model_path:
        log_body_debug(debug, "layoutlmv3_disabled set LAYOUTLMV3_MODEL_PATH for fine-tuned model")
        return None, None

    try:
        processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        model.eval()
        _LAYOUTLMV3_CACHE['processor'] = processor
        _LAYOUTLMV3_CACHE['model'] = model
        log_body_debug(debug, f"layoutlmv3_loaded path={model_path}")
    except Exception as exc:
        log_body_debug(debug, f"layoutlmv3_load_failed error={exc}")
        _LAYOUTLMV3_CACHE['processor'] = None
        _LAYOUTLMV3_CACHE['model'] = None

    return _LAYOUTLMV3_CACHE['processor'], _LAYOUTLMV3_CACHE['model']


def _layoutlmv3_confirms_item_label(label_text, debug=False):
    """Use a fine-tuned LayoutLMv3 model to confirm whether a row label is an item category."""
    processor, model = _get_layoutlmv3_components(debug=debug)
    if processor is None or model is None or torch is None:
        return None

    words = [w for w in re.split(r'\s+', str(label_text or '').strip()) if w]
    if not words:
        return None

    # Synthetic bounding boxes keep token order; useful when only row label text is available.
    boxes = []
    left = 50
    for _ in words:
        right = min(left + 110, 980)
        boxes.append([left, 400, right, 520])
        left = min(right + 10, 980)

    try:
        from PIL import Image as PILImage
        dummy_image = PILImage.new('RGB', (1000, 1000), color='white')
        encoded = processor(
            images=dummy_image,
            words=words,
            boxes=boxes,
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predicted = torch.argmax(logits, dim=-1)[0].tolist()

        id2label = getattr(model.config, 'id2label', {}) or {}
        labels = [str(id2label.get(idx, '')).upper() for idx in predicted]
        return any('ITEM' in lbl or 'CATEGORY' in lbl for lbl in labels)
    except Exception as exc:
        log_body_debug(debug, f"layoutlmv3_infer_failed error={exc}")
        return None


def log_attachment_debug(enabled, message):
    """Print attachment extraction debug details to the terminal when enabled."""
    if enabled:
        print(f"[ATTACH_DEBUG] {message}")


def log_body_debug(enabled, message):
    """Print email body extraction debug details to the terminal when enabled."""
    if enabled:
        print(f"[BODY_DEBUG] {message}")


def load_user_label_overrides(path=LABEL_OVERRIDES_PATH):
    """Load persisted PO/non-PO label overrides keyed by message id."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return {}
        normalized = {}
        for key, value in payload.items():
            if key:
                normalized[str(key)] = bool(value)
        return normalized
    except Exception:
        return {}


def save_user_label_overrides(overrides, path=LABEL_OVERRIDES_PATH):
    """Persist PO/non-PO label overrides for future runs."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clean = {str(k): bool(v) for k, v in (overrides or {}).items() if k}
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(clean, handle, indent=2)
        return True
    except Exception:
        return False


def clean_email_body_text(body_content):
    """Convert HTML email body into readable plain text while preserving line breaks."""
    if not body_content:
        return ""

    text = str(body_content)
    # Preserve common block-level separators before stripping tags.
    text = re.sub(r'(?i)<br\s*/?>', '\n', text)
    text = re.sub(r'(?i)</(p|div|li|tr|h1|h2|h3|h4|h5|h6)>', '\n', text)
    text = re.sub(r'(?i)<li[^>]*>', '- ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_html_tables_fallback(html_content):
    """Extract HTML tables with a lightweight parser when pandas.read_html engines are unavailable."""
    if not html_content:
        return []

    html_text = str(html_content)
    table_blocks = re.findall(r'(?is)<table\b[^>]*>.*?</table>', html_text)
    extracted = []

    def _cell_text(cell_html):
        value = re.sub(r'(?i)<br\s*/?>', ' ', str(cell_html))
        value = re.sub(r'<[^>]+>', ' ', value)
        value = html.unescape(value)
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    def _looks_like_header(row_values):
        joined = ' '.join(v.lower() for v in row_values if v)
        hints = ['type', 'contract', 'item', 'size', 'mth', 'yrs', 'total', 'sticker', 'ticket', 'pos']
        return any(h in joined for h in hints)

    for table_html in table_blocks:
        row_blocks = re.findall(r'(?is)<tr\b[^>]*>.*?</tr>', table_html)
        rows = []
        for row_html in row_blocks:
            cell_blocks = re.findall(r'(?is)<t[hd]\b[^>]*>.*?</t[hd]>', row_html)
            if not cell_blocks:
                continue
            row_values = [_cell_text(cell) for cell in cell_blocks]
            if any(val for val in row_values):
                rows.append(row_values)

        if not rows:
            continue

        max_cols = max(len(r) for r in rows)
        normalized_rows = [r + [''] * (max_cols - len(r)) for r in rows]

        if len(normalized_rows) >= 2 and _looks_like_header(normalized_rows[0]):
            df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
        else:
            df = pd.DataFrame(normalized_rows)

        extracted.append(df)

    return extracted


def calculate_po_score(subject, body, attachments=None):
    """Calculate PO classification score with improved filtering."""
    score = 0
    matched_keywords = []
    matched_patterns = []
    
    text = f"{subject} {body}".lower()
    subject_lower = subject.lower()
    
    # First check negative keywords - these indicate non-PO emails
    for neg_keyword in NEGATIVE_KEYWORDS:
        if neg_keyword in text:
            score -= 5  # Strong penalty for non-PO indicators
    
    for keyword in PO_KEYWORDS:
        if keyword.lower() in text:
            if keyword.lower() in subject_lower:
                score += 3
            else:
                score += 1
            matched_keywords.append(keyword)
    
    for pattern in PO_NUMBER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            score += 5
            if isinstance(matches[0], str):
                matched_patterns.extend(matches)
    
    if attachments:
        for att in attachments:
            att_lower = att.lower()
            if 'po' in att_lower or 'purchase' in att_lower or 'order' in att_lower:
                score += 2
    
    return score, matched_keywords, matched_patterns


def extract_po_numbers(text):
    """
    Extract PO numbers from text, prioritizing the "Purchase Order Number" label.
    Returns a list with a single, most likely correct PO number.
    Ensures only valid PO format is extracted without trailing text.
    """
    if not text:
        return []

    raw_text = str(text)
    upper_text = raw_text.upper()
    po_number = None
    
    # PRIORITY 1: Extract from "Purchase Order Number" label (most reliable)
    # Capture only the PO number part, stopping at first non-alphanumeric character
    pon_pattern = r'Purchase\s*Order\s*Number[:\s\-]*([A-Z0-9-]+?)(?:\s|$|[^A-Z0-9-])'
    pon_match = re.search(pon_pattern, raw_text, re.IGNORECASE)
    if pon_match:
        candidate = pon_match.group(1).strip()
        # Validate it has the right format (contains 'PO' or is MEL format)
        if 'PO' in candidate.upper() or re.match(r'MEL\d{4}PO\d+', candidate, re.IGNORECASE):
            # Extract only the valid MEL PO format from the candidate
            valid_po = re.search(r'MEL\d{4}PO\d+', candidate, re.IGNORECASE)
            if valid_po:
                return [valid_po.group(0)]
            return [candidate]
    
    # PRIORITY 2: Extract MEL format PO (e.g., MEL2026PO14536)
    # Use word boundary to ensure we stop at the end of the number
    mel_pattern = r'\bMEL\d{4}PO\d+\b'
    mel_match = re.search(mel_pattern, upper_text)
    if mel_match:
        po_number = mel_match.group(0)
        return [po_number]
    
    # Also try without word boundary (in case of text concatenation)
    mel_pattern_loose = r'MEL\d{4}PO\d+'
    mel_match = re.search(mel_pattern_loose, upper_text)
    if mel_match:
        po_number = mel_match.group(0)
        return [po_number]
    
    # PRIORITY 3: Extract from generic MEL format with OCR fuzzy matching
    mel_spaced_pattern = r'M\s*E\s*L\s*([0-9OIL]{4})\s*P\s*[O0]\s*([0-9OIL]{4,}?)(?:\s|$|[^0-9OIL])'
    mel_spaced_match = re.search(mel_spaced_pattern, upper_text, re.IGNORECASE)
    if mel_spaced_match:
        normalized_year = mel_spaced_match.group(1).replace('O', '0').replace('I', '1').replace('L', '1')
        normalized_number = mel_spaced_match.group(2).replace('O', '0').replace('I', '1').replace('L', '1')
        po_number = f"MEL{normalized_year}PO{normalized_number}"
        return [po_number]
    
    # PRIORITY 4: Extract from collapsed text (no spaces/punctuation)
    collapsed_text = re.sub(r'[^A-Z0-9]', '', upper_text)
    collapsed_text = collapsed_text.replace('P0', 'PO')
    collapsed_match = re.search(r'(MEL[0-9OIL]{4}PO[0-9OIL]{4,})', collapsed_text, re.IGNORECASE)
    if collapsed_match:
        match = collapsed_match.group(1)
        normalized = re.sub(r'(?<=\d)[O](?=\d)', '0', match.upper())
        normalized = re.sub(r'(?<=PO)[OIL]+', lambda m: m.group(0).replace('O', '0').replace('I', '1').replace('L', '1'), normalized)
        normalized = normalized.replace('P0', 'PO')
        return [normalized]
    
    # No valid PO number found
    return []


def extract_item_codes(text):
    """Extract item codes from text."""
    patterns = [
        r'[A-Z]{2,4}\d+[A-Z]?\d*-[A-Z]?\d+',
    ]
    
    items = []
    for pattern in patterns:
        items.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(items))


def extract_items_from_email_body(text, debug=False, message_id=None, raw_html=None):
    """Extract item details from plain email body text."""
    if not text and not raw_html:
        log_body_debug(debug, f"message_id={message_id} body_empty")
        return []

    def _to_float(value):
        try:
            return float(str(value).replace(',', '').strip())
        except Exception:
            return None

    normalized = str(text or '').replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    lines = []
    for ln in normalized.splitlines():
        cleaned_line = re.sub(r'\s+', ' ', ln).strip()
        cleaned_line = re.sub(r'^[\-\*\u2022•]+\s*', '', cleaned_line)
        if cleaned_line:
            lines.append(cleaned_line)

    header_block = {
        'item', 'item description', 'product', 'quantity', 'unit', 'unit price', 'line total', 'amount', 'rate'
    }
    parsed_items = []
    seen = set()
    debug_stats = {
        'html_tables_found': 0,
        'tables_skipped_not_item': 0,
        'rows_seen': 0,
        'rows_skipped_empty_or_helper': 0,
        'rows_skipped_no_numeric_cells': 0,
        'rows_skipped_no_total': 0,
        'rows_skipped_non_item': 0,
        'rows_skipped_layoutlmv3_reject': 0,
        'rows_skipped_duplicate': 0,
        'rows_extracted_from_html': 0,
        'rows_extracted_from_text': 0,
        'rows_extracted_from_fallback': 0,
    }
    log_body_debug(
        debug,
        f"message_id={message_id} body_debug_start plain_text_chars={len(normalized)} raw_html_chars={len(str(raw_html or ''))}"
    )

    # First pass: parse true HTML tables before flattened text parsing.
    def _flatten_columns(columns):
        flattened = []
        for col in columns:
            if isinstance(col, tuple):
                parts = [str(p).strip() for p in col if str(p).strip() and str(p).strip().lower() != 'nan']
                flattened.append(' '.join(parts) if parts else '')
            else:
                flattened.append(str(col).strip())
        return [re.sub(r'\s+', ' ', c) for c in flattened]

    def _normalize_header(name):
        lowered = str(name or '').strip().lower()
        lowered = lowered.replace('.', '').replace('_', ' ')
        lowered = re.sub(r'\s+', ' ', lowered)
        return lowered

    def _is_size_like_header(name):
        h = _normalize_header(name)
        if not h:
            return False
        return any(token in h for token in ['mth', 'mths', 'yrs', 'year', 'size', 'lb', 'total'])

    def _safe_text(value):
        if value is None:
            return ''
        as_text = str(value).strip()
        if as_text.lower() == 'nan':
            return ''
        return re.sub(r'\s+', ' ', as_text)

    def _find_contract_no(table_values):
        flattened_text = ' '.join(_safe_text(v) for v in table_values)
        match = re.search(r'\b(?:VA|VJ|VQ|VB)\d{6,}\b', flattened_text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def _find_po_type(table_values):
        flattened_text = ' '.join(_safe_text(v).lower() for v in table_values)
        if 'online' in flattened_text:
            return 'Online'
        if 'retail' in flattened_text:
            return 'Retail'
        return None

    def _extract_scalar_text(value):
        """Return normalized text for scalar, Series, or DataFrame cell selections."""
        if isinstance(value, pd.DataFrame):
            parts = value.astype(str).values.flatten().tolist()
            text = ' '.join(_safe_text(v) for v in parts)
        elif isinstance(value, pd.Series):
            parts = value.astype(str).tolist()
            text = ' '.join(_safe_text(v) for v in parts)
        elif isinstance(value, (list, tuple, set)):
            text = ' '.join(_safe_text(v) for v in value)
        else:
            text = _safe_text(value)

        text = re.sub(r'\s+Name\s*:\s*\d+\s*,\s*dtype\s*:\s*\w+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*dtype\s*:\s*\w+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _derive_item_category(row, non_size_cols, po_type, contract_no):
        """Derive clean item category text from non-size columns in a row."""
        raw_parts = [_extract_scalar_text(row.get(col, '')) for col in non_size_cols]
        raw_parts = [p for p in raw_parts if p and p.lower() not in {'nan', 'none'}]
        if not raw_parts:
            return ''

        def _extract_known_category(text):
            value = str(text or '')
            value = re.sub(r'\b(?:Type|Contract\s*No|Contract\s*Number)\b', ' ', value, flags=re.IGNORECASE)
            value = re.sub(r'\b(?:Online|Retail)\b', ' ', value, flags=re.IGNORECASE)
            value = re.sub(r'\b(?:VA|VJ|VQ)\d{6,}\b', ' ', value, flags=re.IGNORECASE)
            value = re.sub(r'\s+', ' ', value).strip(' -:;,.')
            lowered = value.lower()

            category_patterns = [
                (r'^\s*7\s*%\s*$', '7%'),
                (r'\bbase\s*(?:qty|quantity|number)?\b', 'Base Qty'),
                (r'\bprice\s*ticket\b', 'PRICE TICKET'),
                (r'\bcarto+n\s*sticker\b|\bcarton\s*sticker\b|\bcarton\s*stk\b|\bcartoon\s*stk\b', 'Carton stk'),
                (r'\blaminat(?:ing|e)\s*sticker\b|\blaminating\s*stk\b|\blaminate\s*stk\b|\blaminating\b', 'Laminating Stk'),
                (r'\bpos\b', 'POS'),
            ]

            matches = []
            for pattern, canonical in category_patterns:
                if re.search(pattern, lowered, flags=re.IGNORECASE):
                    matches.append(canonical)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                # Headers can contain mixed labels (e.g. "POS & CARTON STICKER"); treat as non-row noise.
                return ''
            return ''

        blocked_exact = {
            'type', 'contract no', 'contract number', 'item category', 'dir',
            'online', 'retail'
        }
        known_keywords = ['pos', 'carton stk', 'cartoon sticker', 'carton sticker', 'laminating stk', 'laminating', 'price ticket']

        candidates = []
        contract_pattern = re.compile(r'\b(?:VA|VJ|VQ|VB)\d{6,}\b', re.IGNORECASE)
        for part in raw_parts:
            cleaned = re.sub(r'\s+', ' ', part).strip(' -:;,.')
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in blocked_exact:
                continue
            if po_type and lowered == po_type.lower():
                continue
            if contract_no and cleaned.upper() == str(contract_no).upper():
                continue
            if contract_pattern.fullmatch(cleaned):
                continue
            candidates.append(cleaned)

        if not candidates:
            return _extract_scalar_text(raw_parts[-1])

        for candidate in candidates:
            known = _extract_known_category(candidate)
            if known:
                return known

        for candidate in reversed(candidates):
            lowered = candidate.lower()
            if any(keyword in lowered for keyword in known_keywords):
                known = _extract_known_category(candidate)
                if known:
                    return known

        # If no known category was found, treat as irrelevant for this matrix extraction path.
        return ''

    def _candidate_size_score(values):
        score = 0
        for value in values:
            token = _safe_text(value)
            if not token:
                continue
            if _canonical_master_size_key(token):
                score += 2
            elif _is_size_like_header(token):
                score += 1
        return score

    def _infer_header_row_and_reframe(df):
        """Find a likely header row in the first few rows and rebuild the DataFrame."""
        if df is None or df.empty:
            return df, None

        rows = df.astype(str).fillna('').values.tolist()
        lookahead = min(len(rows), 8)
        best_idx = None
        best_score = -1

        for idx in range(lookahead):
            score = _candidate_size_score(rows[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None or best_score < 2 or best_idx >= len(rows) - 1:
            return df, None

        header_row = [_safe_text(v) for v in rows[best_idx]]
        data_rows = rows[best_idx + 1:]
        reframed = pd.DataFrame(data_rows, columns=header_row)
        return reframed, best_idx

    if raw_html:
        try:
            html_tables = pd.read_html(StringIO(str(raw_html)))
        except Exception as exc:
            log_body_debug(debug, f"message_id={message_id} body_html_parse_failed error={exc}")
            html_tables = extract_html_tables_fallback(raw_html)
            log_body_debug(
                debug,
                f"message_id={message_id} body_html_fallback_tables_found={len(html_tables)}"
            )

        if html_tables:
            log_body_debug(debug, f"message_id={message_id} body_html_tables_found={len(html_tables)}")
        else:
            log_body_debug(debug, f"message_id={message_id} body_html_tables_found=0")

        debug_stats['html_tables_found'] = len(html_tables)

        for table_idx, df in enumerate(html_tables, start=1):
            if df is None or df.empty:
                log_body_debug(debug, f"message_id={message_id} body_table_{table_idx} skipped=empty_dataframe")
                continue

            raw_table_values = df.astype(str).fillna('').values.flatten().tolist()

            working_df = df.copy()
            working_df.columns = _flatten_columns(working_df.columns)
            log_body_debug(
                debug,
                f"message_id={message_id} body_table_{table_idx} shape={working_df.shape} columns={working_df.columns.tolist()}"
            )
            normalized_cols = [_normalize_header(c) for c in working_df.columns]
            size_cols = [
                col for col, normalized_col in zip(working_df.columns, normalized_cols)
                if _is_size_like_header(normalized_col)
            ]

            # Skip non-item tables.
            if len(size_cols) < 1:
                inferred_df, inferred_idx = _infer_header_row_and_reframe(working_df)
                if inferred_df is not None and inferred_idx is not None:
                    working_df = inferred_df
                    working_df.columns = _flatten_columns(working_df.columns)
                    normalized_cols = [_normalize_header(c) for c in working_df.columns]
                    size_cols = [
                        col for col, normalized_col in zip(working_df.columns, normalized_cols)
                        if _is_size_like_header(normalized_col)
                    ]
                    log_body_debug(
                        debug,
                        f"message_id={message_id} body_table_{table_idx} header_inferred row_index={inferred_idx} columns={working_df.columns.tolist()} size_cols={size_cols}"
                    )

            if len(size_cols) < 1:
                debug_stats['tables_skipped_not_item'] += 1
                sample_rows = working_df.head(3).astype(str).values.tolist()
                log_body_debug(
                    debug,
                    f"message_id={message_id} body_table_{table_idx} skipped=not_item_table size_cols={size_cols} sample_rows={sample_rows}"
                )
                continue

            table_values = raw_table_values + working_df.astype(str).fillna('').values.flatten().tolist()
            contract_no = _find_contract_no(table_values)
            po_type = _find_po_type(table_values)

            row_label_col = None
            for col in working_df.columns:
                if col in size_cols:
                    continue
                candidate_sample = working_df[col].head(5)
                if isinstance(candidate_sample, pd.DataFrame):
                    candidate_values = candidate_sample.astype(str).values.flatten().tolist()
                else:
                    candidate_values = candidate_sample.astype(str).tolist()
                candidate_text = ' '.join(_safe_text(v) for v in candidate_values)
                if re.search(r'[A-Za-z]{3,}', candidate_text):
                    row_label_col = col
                    break

            if row_label_col is None:
                row_label_col = working_df.columns[0]

            non_size_cols = [col for col in working_df.columns if col not in size_cols]

            # Some email tables have a final total column without a proper header.
            implicit_total_col = None
            for col in reversed(non_size_cols):
                if col == row_label_col:
                    continue
                col_name = _normalize_header(col)
                series = working_df[col]
                if isinstance(series, pd.DataFrame):
                    values = series.astype(str).values.flatten().tolist()
                else:
                    values = series.astype(str).tolist()

                numeric_hits = sum(1 for v in values if _to_float(v) is not None)
                non_empty = sum(1 for v in values if _safe_text(v))
                if non_empty == 0:
                    continue

                # Prefer blank/unnamed headers with mostly numeric values.
                if (
                    col_name in {'', 'nan'}
                    or col_name.startswith('unnamed:')
                    or col_name in {'total value', 'value'}
                ) and (numeric_hits / max(non_empty, 1) >= 0.6):
                    implicit_total_col = col
                    break

            for _, row in working_df.iterrows():
                debug_stats['rows_seen'] += 1
                row_context_text = ' '.join(_extract_scalar_text(row.get(col, '')) for col in non_size_cols).strip()
                label = _derive_item_category(row, non_size_cols, po_type, contract_no)
                if not label:
                    if re.fullmatch(r'\s*7\s*%\s*', row_context_text, flags=re.IGNORECASE):
                        label = '7%'
                    elif row_context_text == '':
                        label = 'Base Qty'
                    else:
                        debug_stats['rows_skipped_non_item'] += 1
                        continue
                label_lower = label.lower()
                if not label or label_lower in {'type', 'contract no', 'dir'}:
                    debug_stats['rows_skipped_empty_or_helper'] += 1
                    continue
                if any(token in label_lower for token in ['price ticket', 'cartoon sticker', 'laminating', 'carton stk', 'laminating stk', 'pos', '7%', 'base qty']):
                    is_known_item = True
                else:
                    is_known_item = False

                size_breakdown = {}
                for col in size_cols:
                    numeric_val = _to_float(row.get(col, None))
                    if numeric_val is None:
                        continue
                    size_breakdown[_normalize_header(col)] = numeric_val

                if len(size_breakdown) < 1:
                    debug_stats['rows_skipped_no_numeric_cells'] += 1
                    continue

                non_total_values = [
                    value for col_name, value in size_breakdown.items()
                    if 'total' not in col_name
                ]
                total_candidates = [
                    value for col_name, value in size_breakdown.items()
                    if 'total' in col_name
                ]

                explicit_total = total_candidates[0] if total_candidates else None
                implicit_total_value = _to_float(row.get(implicit_total_col, None)) if implicit_total_col is not None else None
                total_value = explicit_total if explicit_total is not None else implicit_total_value

                if not is_known_item:
                    # Filter out helper rows like percentage/tax rows that are not item rows.
                    if label_lower in {'%', 'first size'}:
                        debug_stats['rows_skipped_non_item'] += 1
                        continue
                    if len(re.findall(r'[A-Za-z]{2,}', label)) == 0:
                        debug_stats['rows_skipped_non_item'] += 1
                        continue

                    layoutlmv3_decision = _layoutlmv3_confirms_item_label(label, debug=debug)
                    if layoutlmv3_decision is False:
                        debug_stats['rows_skipped_layoutlmv3_reject'] += 1
                        continue

                product_name = label

                master_row = _empty_master_row()
                master_row['Type'] = po_type or ''
                master_row['Contract No'] = contract_no or ''
                master_row['Item Category'] = label

                for raw_key, raw_value in size_breakdown.items():
                    canonical_key = _canonical_master_size_key(raw_key)
                    if canonical_key:
                        master_row[canonical_key] = raw_value

                if total_value is not None:
                    master_row['Total'] = total_value

                dedupe_total = total_value if total_value is not None else (round(sum(non_total_values), 4) if non_total_values else 0.0)
                key = (
                    product_name.lower(),
                    contract_no or '',
                    round(float(dedupe_total), 4),
                )
                if key in seen:
                    debug_stats['rows_skipped_duplicate'] += 1
                    continue
                seen.add(key)

                parsed_items.append({
                    'product_name': product_name,
                    'quantity': None,
                    'price': None,
                    'amount': total_value,
                    'unit': 'Nos',
                    'contract_no': contract_no,
                    'type': po_type,
                    'item_category': label,
                    'master_columns': master_row,
                    'size_breakdown': json.dumps(size_breakdown, ensure_ascii=True),
                    'source': f'body_table_{table_idx}',
                })
                debug_stats['rows_extracted_from_html'] += 1

    log_body_debug(debug, f"message_id={message_id} body_items_after_html_tables={len(parsed_items)}")

    dash_pattern = re.compile(
        r'^(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]+?)\s*-\s*(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s*-\s*(?:USD\s*)?(?P<rate>\d+(?:\.\d+)?)$',
        re.IGNORECASE,
    )
    table_with_amount_pattern = re.compile(
        r'^(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]+?)\s+(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s+(?P<rate>\d+(?:\.\d+)?)\s*(?:USD)?\s+(?P<amount>\d+(?:\.\d+)?)$',
        re.IGNORECASE,
    )
    table_no_amount_pattern = re.compile(
        r'^(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]+?)\s+(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s+(?:USD\s*)?(?P<rate>\d+(?:\.\d+)?)$',
        re.IGNORECASE,
    )
    compact_dash_pattern = re.compile(
        r'^(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]+?)\s*-\s*(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s*-\s*(?P<currency>USD)?\s*(?P<rate>\d+(?:\.\d+)?)\s*(?:USD)?$',
        re.IGNORECASE,
    )
    labeled_dash_pattern = re.compile(
        r'^(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]+?)\s*-\s*Quantity\s*:\s*(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s*-\s*(?:Unit\s*)?(?:Rate|Price)\s*:\s*(?P<rate>\d+(?:\.\d+)?)\s*(?:USD)?(?:\s*-\s*(?:Line\s*Total|Amount)\s*:\s*(?P<amount>\d+(?:\.\d+)?))?$',
        re.IGNORECASE,
    )

    # Fallback for HTML tables flattened into one long line.
    full_text_table_pattern = re.compile(
        r'(?P<name>[A-Za-z][A-Za-z0-9&/\-\s]{2,40}?)\s+(?P<qty>\d[\d,]*)\s*(?P<unit>pcs?|pieces?)\s+(?P<rate>\d+(?:\.\d+)?)\s*(?:USD)?\s+(?P<amount>\d+(?:\.\d+)?)',
        re.IGNORECASE,
    )

    log_body_debug(debug, f"message_id={message_id} body_lines={len(lines)}")
    candidate_lines = []
    for line in lines:
        lower = line.lower()
        if lower in header_block:
            continue
        if any(h in lower for h in ['itemized order breakdown', 'description value', 'net amount', 'subtotal', 'grand total']):
            continue

        if any(tok in lower for tok in ['sticker', 'ticket', 'pcs', 'pieces', 'usd', 'unit price', 'line total']):
            candidate_lines.append(line)

        match = (
            dash_pattern.match(line)
            or compact_dash_pattern.match(line)
            or labeled_dash_pattern.match(line)
            or table_with_amount_pattern.match(line)
            or table_no_amount_pattern.match(line)
        )
        if not match:
            continue

        name = re.sub(r'\s+', ' ', match.group('name')).strip(' -:;,.')
        qty = _to_float(match.group('qty'))
        rate = _to_float(match.group('rate'))
        amount = _to_float(match.groupdict().get('amount'))
        if qty is None or rate is None:
            continue
        if amount is None:
            amount = round(qty * rate, 4)

        key = (name.lower(), round(qty, 4), round(rate, 6), round(amount, 4))
        if key in seen:
            continue
        seen.add(key)

        parsed_items.append({
            'product_name': name,
            'quantity': qty,
            'price': rate,
            'amount': amount,
            'unit': match.group('unit').capitalize(),
            'source': 'body',
        })
        debug_stats['rows_extracted_from_text'] += 1

    log_body_debug(debug, f"message_id={message_id} body_items_extracted={len(parsed_items)}")

    # Second pass: parse from whole text when body collapsed to one line/table-like stream.
    if not parsed_items:
        collapsed_text = re.sub(r'\s+', ' ', normalized)
        for match in full_text_table_pattern.finditer(collapsed_text):
            name = re.sub(r'\s+', ' ', match.group('name')).strip(' -:;,.')
            lower_name = name.lower()
            if any(bad in lower_name for bad in ['itemized order breakdown', 'item description', 'quantity', 'unit price', 'line total']):
                continue

            qty = _to_float(match.group('qty'))
            rate = _to_float(match.group('rate'))
            amount = _to_float(match.group('amount'))
            if qty is None or rate is None or amount is None:
                continue

            key = (name.lower(), round(qty, 4), round(rate, 6), round(amount, 4))
            if key in seen:
                continue
            seen.add(key)
            parsed_items.append({
                'product_name': name,
                'quantity': qty,
                'price': rate,
                'amount': amount,
                'unit': match.group('unit').capitalize(),
                'source': 'body',
            })
            debug_stats['rows_extracted_from_fallback'] += 1

        if parsed_items:
            log_body_debug(debug, f"message_id={message_id} body_items_extracted_fallback={len(parsed_items)}")

    if debug and parsed_items:
        for idx, item in enumerate(parsed_items, start=1):
            log_body_debug(
                debug,
                f"message_id={message_id} body_item_{idx} item_name={item.get('product_name')} quantity={item.get('quantity')} rate={item.get('price')} total_amount={item.get('amount')}"
            )
    elif debug and candidate_lines:
        log_body_debug(debug, f"message_id={message_id} body_candidate_lines_count={len(candidate_lines)}")
        for idx, line in enumerate(candidate_lines[:20], start=1):
            log_body_debug(debug, f"message_id={message_id} body_candidate_line_{idx}={line}")

    log_body_debug(debug, f"message_id={message_id} body_debug_summary={debug_stats}")
    if debug and not parsed_items:
        preview = normalized[:300].replace('\n', ' ')
        log_body_debug(debug, f"message_id={message_id} body_no_items_extracted plain_text_preview={preview}")

    return parsed_items


def extract_supplier_name_from_email_body(text):
    """Extract supplier/vendor name from plain email body text."""
    if not text:
        return None

    normalized_text = re.sub(r'\r\n?', '\n', str(text))

    # Label-based patterns are most reliable in email bodies.
    patterns = [
        r'(?im)^\s*(?:supplier|vendor|seller|manufacturer|from company|company|supplier name|vendor name)\s*[:\-]\s*([^\n]{2,120})',
        r'(?im)^\s*dear\s+([A-Za-z][A-Za-z0-9&.,\'()\-\s]{1,80})\s*[,!]',
    ]

    def _clean_supplier(candidate):
        candidate = re.sub(r'\s+', ' ', candidate).strip(" \t\r\n-:;,.")
        candidate = re.sub(r'\b(?:thanks|thank you|regards|best regards|sincerely)\b.*$', '', candidate, flags=re.IGNORECASE).strip()
        # Skip obvious non-name fragments.
        if len(candidate) < 3:
            return None
        if any(token in candidate.lower() for token in ['@', 'http://', 'https://']):
            return None
        return candidate[:120]

    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if match:
            cleaned = _clean_supplier(match.group(1))
            if cleaned:
                return cleaned

    # Fallback: look for signatures like "ACME LTD" in the last lines.
    tail_lines = [ln.strip() for ln in normalized_text.split('\n')[-8:] if ln.strip()]
    company_suffix_pattern = r'\b(?:LTD|LIMITED|LLC|INC|CORP|CO\.?|PVT\.?\s*LTD)\b'
    for line in tail_lines:
        if re.search(company_suffix_pattern, line, re.IGNORECASE):
            cleaned = _clean_supplier(line)
            if cleaned:
                return cleaned

    return None


def get_confidence_level(score):
    """Get confidence level from score."""
    if score >= 15:
        return "HIGH", "🟢"
    elif score >= 10:
        return "MEDIUM", "🟡"
    elif score >= 5:
        return "LOW", "🟠"
    else:
        return "NOT_PO", "⚪"


# ============== Attachment OCR + PO Parsing Functions ==============

def normalize_date(date_str):
    """Normalize various date formats to YYYY-MM-DD format."""
    if not date_str:
        return None

    date_str = date_str.strip()
    date_formats = [
        '%d/%m/%y', '%d-%m-%y',
        '%d/%m/%Y', '%d-%m-%Y',
        '%m/%d/%y', '%m-%d-%y',
        '%m/%d/%Y', '%m-%d-%Y',
        '%y/%m/%d', '%y-%m-%d',
        '%Y/%m/%d', '%Y-%m-%d',
        '%b %d, %Y', '%B %d, %Y',
        '%b %d %Y', '%B %d %Y',
        '%d %b %Y', '%d %B %Y',
        '%d %b, %Y', '%d %B, %Y',
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            continue

    return date_str


def _run_ocr_on_images(original_img, processed_img):
    """Run multiple OCR passes and combine their output for better PO detection."""
    ocr_outputs = []
    ocr_configs = [
        ('--oem 3 --psm 6', processed_img),
        ('--oem 3 --psm 11', processed_img),
        ('--oem 3 --psm 3', original_img),
    ]

    for config, image in ocr_configs:
        text = pytesseract.image_to_string(image, config=config)
        if text and text.strip():
            ocr_outputs.append(text)

    return "\n".join(ocr_outputs)


def parse_purchase_order(text):
    """Parse extracted OCR text and identify common PO fields."""
    po_data = {
        'po_number': None,
        'date': None,
        'vendor_name': None,
        'total_amount': None,
        'items': [],
        'raw_text': text
    }

    lines = text.split('\n')

    po_patterns = [
        r'Works?\s*Order\s*(?:Number|No\.?|#)?\s*[:\s]*([A-Z0-9-]+)',
        r'(?:REQUISITION|Requisition)\s*No[.,]?\s*([0-9]+)',
        r'No[.,]\s*([0-9]+)',
        r'PO[W#:.\s]+\s*([0-9]+)',
        r'P\.?O\.?\s*#?\s*[:\s]*([0-9]+)',
        r'P\.?O\.?\s*(?:Number|No\.?|#)?\s*[:\s]*([A-Z0-9-]+)',
        r'Purchase\s*Order\s*(?:Number|No\.?|#)?\s*[:\s]*([A-Z0-9-]+)',
        r'Order\s*(?:Number|No\.?|#)?\s*[:\s]*([A-Z0-9-]+)',
    ]

    date_patterns = [
        r'[Dd]ate\s*[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
        r'([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})',
        r'([0-9]{1,2}-[0-9]{1,2}-[0-9]{2,4})',
        r'([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})',
    ]

    amount_patterns = [
        r'[Pp]rice\s*[:\s]*([0-9,\s]+)',
        r'[Ff]\s*([0-9]{2,3},\s*[0-9]{3})',
        r'(\d{2,3},\s*\d{3})',
        r'Total\s*[:\s]*\$?([0-9,\s]+\.?[0-9]*)',
        r'Grand\s*Total\s*[:\s]*\$?([0-9,\s]+\.?[0-9]*)',
        r'Amount\s*Due\s*[:\s]*\$?([0-9,\s]+\.?[0-9]*)',
        r'SUBTOTAL\s*[:\s]*\$?([0-9,\s]+\.?[0-9]*)',
        r'\$\s*([0-9,]+\.[0-9]{2})',
    ]

    vendor_patterns = [
        r'Customer\s*:\s*([A-Z][A-Z\s]+\s*\([A-Z\s]+\)\s*(?:LTD|Ltd))',
        r'([A-Z][A-Z\s]+\s*\([A-Z\s]+\)\s*(?:LTD|Ltd))',
        r'(?:Supplier|Vendor|Customer)\s*[:\s]+([A-Za-z][A-Za-z0-9\s&.,\'\(\)-]+?(?:LTD|LLC|Inc|Ltd|Corp|Co|PVT)?)',
    ]

    # PRIORITY 1: Extract from "Purchase Order Number" label (most reliable from attachments)
    pon_pattern = r'Purchase\s*Order\s*Number[:\s\-]*([A-Z0-9-]+)'
    pon_match = re.search(pon_pattern, text, re.IGNORECASE)
    if pon_match:
        candidate = pon_match.group(1).strip()
        # Validate it has the right format
        if 'PO' in candidate.upper() or re.match(r'MEL\d{4}PO\d+', candidate, re.IGNORECASE):
            po_data['po_number'] = candidate

    # PRIORITY 2: Use extract_po_numbers() if not found from label
    if not po_data['po_number']:
        extracted_po_numbers = extract_po_numbers(text)
        if extracted_po_numbers:
            po_data['po_number'] = extracted_po_numbers[0]

    # PRIORITY 3: Try other patterns if still not found
    if not po_data['po_number']:
        for pattern in po_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                po_num = match.group(1).strip()
                if re.search(r'\d', po_num) and po_num.upper() not in ['BOX', 'DATE', 'ORDER']:
                    po_data['po_number'] = po_num
                    break

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            po_data['date'] = normalize_date(match.group(1).strip())
            break

    for pattern in vendor_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vendor = re.sub(r'\s+', ' ', match.group(1).strip())
            vendor = re.sub(r'[\s,.:]+$', '', vendor)
            if len(vendor) > 2 and re.search(r'[A-Za-z]{2,}', vendor):
                po_data['vendor_name'] = vendor
                break

    amounts = []
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                clean_amount = str(m).replace(',', '').replace(' ', '')
                amounts.append(float(clean_amount))
            except Exception:
                pass

    if amounts:
        po_data['total_amount'] = max(amounts)

    def _to_float(value):
        try:
            return float(str(value).replace(',', '').strip())
        except Exception:
            return None

    header_tokens = {'item', 'supplier colour', 'colour', 'size', 'quantity', 'unit', 'rate', 'amount', 'amount (usd)'}
    table_header_pattern = re.compile(r'(?i)\bitem\b.*\b(?:quantity|qty)\b')
    table_end_pattern = re.compile(r'(?i)\b(?:subtotal|grand\s*total|amount\s*due|total\s*usd|tax|vat)\b')

    def _sanitize_item_name(left):
        value = left
        if '|' in value:
            value = value.split('|', 1)[0].strip()
        if ';' in value:
            value = value.split(';')[-1].strip()

        # Keep the last clean "Item : Description" segment if OCR merged multiple lines.
        pair_matches = re.findall(
            r'([A-Za-z][A-Za-z0-9&/\-\s]{1,40}:\s*[A-Za-z][A-Za-z0-9&/\-\s]{1,80})',
            value,
        )
        if pair_matches:
            value = pair_matches[-1]

        value = re.sub(r'\s+', ' ', value).strip(" \t\r\n-:;,.|")

        # Remove common noisy prefixes that appear before the real item text.
        value = re.sub(r'(?i)^.*?\b(?:stroke\s*number|supplier\s*fi\s*number)\b', '', value).strip()

        tokens = value.split()
        compact_tokens = []
        for tok in tokens:
            if compact_tokens and compact_tokens[-1].lower() == tok.lower():
                continue
            compact_tokens.append(tok)

        while len(compact_tokens) > 2 and compact_tokens[-1].isupper() and len(compact_tokens[-1]) <= 3:
            compact_tokens.pop()

        return ' '.join(compact_tokens).strip()

    def _canonical_item_name(name):
        """Return a stable item identity for deduping OCR variants."""
        value = re.sub(r'\s+', ' ', str(name or '')).strip().lower()
        if ':' in value:
            value = value.split(':', 1)[0].strip()
        value = re.sub(r'[^a-z0-9]+', ' ', value)
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    def _normalize_num_token(token):
        cleaned = str(token).strip()
        cleaned = re.sub(r'(?<=\d)[Oo](?=\d)', '0', cleaned)
        cleaned = re.sub(r'(?<=\d)[Il](?=\d)', '1', cleaned)
        cleaned = re.sub(r'[^0-9,\.]', '', cleaned)
        return cleaned

    def _extract_item_from_line(candidate_line):
        tokens = [tok for tok in re.split(r'\s+', candidate_line.strip()) if tok]
        if len(tokens) < 5:
            return None

        numeric_positions = []
        for idx, tok in enumerate(tokens):
            val = _to_float(_normalize_num_token(tok))
            if val is not None:
                numeric_positions.append((idx, val))

        if len(numeric_positions) < 2:
            return None

        if len(numeric_positions) >= 3:
            qty_idx, qty_val = numeric_positions[-3]
            rate_idx, rate_val = numeric_positions[-2]
            amount_idx, amount_val = numeric_positions[-1]
            if not (qty_idx < rate_idx < amount_idx):
                return None
        else:
            # OCR can miss/scramble rate token (e.g., "0.0215" -> "oz").
            qty_idx, qty_val = numeric_positions[-2]
            amount_idx, amount_val = numeric_positions[-1]
            if not (qty_idx < amount_idx) or qty_val == 0:
                return None
            rate_idx = amount_idx
            rate_val = amount_val / qty_val

        left = ' '.join(tokens[:qty_idx]).strip()
        left = re.sub(r'^\d+[\).:\-]?\s*', '', left)
        left = re.sub(r'\s+', ' ', left).strip()
        if len(left) < 3:
            return None

        has_row_marker = any(mark in left for mark in [':', '|', ';'])
        has_unit = bool(unit) if 'unit' in locals() else False
        if not has_row_marker and not has_unit:
            return None

        left_lower = left.lower()
        forbidden_left_tokens = [
            'supplier address', 'delivery address', 'reference number', 'purchase order number',
            'stroke number', 'supplier pi number', 'supplier fi number', 'vat number', 'svat number'
        ]
        if any(tok in left_lower for tok in forbidden_left_tokens):
            return None
        if any(tok in left_lower for tok in header_tokens):
            return None
        if any(x in left_lower for x in ['total', 'subtotal', 'payment', 'balance']):
            return None

        unit = None
        if qty_idx + 1 < rate_idx:
            unit_candidate = tokens[qty_idx + 1].lower().strip('.,:;')
            if unit_candidate in {'piece', 'pieces', 'preces', 'peices', 'pcs', 'nos', 'no', 'sets', 'set'}:
                unit = unit_candidate.capitalize()

        has_item_hint = any(k in left_lower for k in ['sticker', 'ticket', 'laminating', 'label', 'carton', 'price'])
        if not unit and not has_item_hint:
            return None

        left = _sanitize_item_name(left)
        if len(left) < 3:
            return None
        if ':' not in left:
            return None

        return {
            'product_name': left,
            'variant': None,
            'description': left,
            'quantity': qty_val,
            'unit': unit,
            'price': rate_val,
            'amount': amount_val,
        }

    pending_left = None
    seen_item_keys = set()
    in_item_table = False
    carry_block_tokens = [
        'supplier address', 'delivery address', 'reference number', 'purchase order number',
        'stroke number', 'supplier pi number', 'supplier fi number', 'vat', 'svat'
    ]

    for line in lines:
        line = line.strip()
        if not line:
            if in_item_table:
                pending_left = None
            continue

        line_lower = line.lower()
        if table_header_pattern.search(line):
            in_item_table = True
            pending_left = None
            continue

        if not in_item_table:
            looks_like_item_line = (':' in line or '|' in line) and re.search(r'\d', line)
            if not looks_like_item_line:
                continue

        if table_end_pattern.search(line):
            in_item_table = False
            pending_left = None
            continue

        if any(tok in line_lower for tok in header_tokens):
            continue
        if any(x in line_lower for x in ['total', 'subtotal', 'payment', 'balance']):
            continue

        # Merge split OCR rows: first line has item text, next line has numeric columns.
        if pending_left:
            merged_line = f"{pending_left} {line}".strip()
            pending_left = None
        else:
            merged_line = line

        if not re.search(r'\d', merged_line):
            # Keep likely item text for next line if this line looks like first column content.
            if ':' in merged_line or len(merged_line.split()) >= 2:
                pending_left = merged_line
            continue

        parsed_item = _extract_item_from_line(merged_line)
        if parsed_item:
            normalized_name = _canonical_item_name(parsed_item.get('description', ''))
            item_key = (
                normalized_name,
                round(float(parsed_item.get('quantity') or 0), 4),
                round(float(parsed_item.get('amount') or 0), 4),
            )
            if item_key not in seen_item_keys:
                seen_item_keys.add(item_key)
                po_data['items'].append(parsed_item)
            continue

        merged_lower = merged_line.lower()
        numeric_count = len(re.findall(r'\d[\d,]*\.?\d*', merged_line))
        blocked_for_carry = any(tok in merged_lower for tok in carry_block_tokens)

        # Carry only true text fragments for the next OCR line; avoid carrying noisy lines.
        if not blocked_for_carry and numeric_count <= 1 and (':' in merged_line or len(merged_line.split()) >= 2):
            pending_left = merged_line
        else:
            pending_left = None

    return po_data


def _is_supported_attachment(name, content_type=None):
    """Check whether an attachment file type can be processed by OCR or text extraction."""
    ext = os.path.splitext((name or '').lower())[1]
    if ext in SUPPORTED_ATTACHMENT_EXTENSIONS:
        return True
    if content_type and content_type.lower().startswith('image/'):
        return True
    if ext == '.pdf' or (content_type and 'pdf' in content_type.lower()):
        return PDF_AVAILABLE
    return False


def _enhance_image_from_bytes(attachment_bytes):
    """Load and enhance image bytes for OCR."""
    pil_image = Image.open(BytesIO(attachment_bytes))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]
    if width < 1000:
        scale = 1000 / max(width, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, binary


def _extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF using PyMuPDF, with OCR fallback for scanned pages."""
    combined_text = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        page_text = page.get_text("text")
        if len(page_text.strip()) > 50:  # text-based page
            combined_text.append(page_text)
        elif OCR_AVAILABLE:  # scanned page — render and OCR
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            original_img, processed_img = _enhance_image_from_bytes(img_bytes)
            ocr_text = _run_ocr_on_images(original_img, processed_img)
            combined_text.append(ocr_text)
    doc.close()
    return "\n".join(combined_text)


def extract_po_from_attachment(attachment, debug=False):
    """Extract PO fields from one attachment (image via OCR or PDF via text extraction)."""
    attachment_bytes = attachment.get('bytes')
    if not attachment_bytes:
        return None

    name = attachment.get('name', '')
    ext = os.path.splitext(name.lower())[1]
    content_type = attachment.get('contentType', '')
    is_pdf = ext == '.pdf' or 'pdf' in content_type.lower()

    try:
        if is_pdf and PDF_AVAILABLE:
            combined_text = _extract_text_from_pdf(attachment_bytes)
        elif OCR_AVAILABLE:
            original_img, processed_img = _enhance_image_from_bytes(attachment_bytes)
            combined_text = _run_ocr_on_images(original_img, processed_img)
        else:
            return None

        searchable_text = f"{name}\n{combined_text}"
        po_data = parse_purchase_order(searchable_text)
        po_candidates = extract_po_numbers(searchable_text)
        if po_candidates and not po_data.get('po_number'):
            po_data['po_number'] = po_candidates[0]
        po_data['po_candidates'] = po_candidates
        po_data['source_file'] = name
        po_data['text_length'] = len(combined_text.strip())
        po_data['raw_text'] = combined_text
        po_data['extraction_status'] = 'processed'

        extracted_items = po_data.get('items', [])
        log_attachment_debug(
            debug,
            f"source_file={name} extracted_items_count={len(extracted_items)}"
        )
        if extracted_items:
            for idx, item in enumerate(extracted_items, start=1):
                log_attachment_debug(
                    debug,
                    "source_file={name} item_{idx} item_name={item_name} quantity={qty} rate={rate} total_amount={amount}".format(
                        name=name,
                        idx=idx,
                        item_name=item.get('product_name') or item.get('description', ''),
                        qty=item.get('quantity', ''),
                        rate=item.get('price', ''),
                        amount=item.get('amount', ''),
                    )
                )
        else:
            log_attachment_debug(debug, f"source_file={name} no_item_rows_extracted")
            candidate_lines = []
            for raw_line in combined_text.splitlines():
                line = re.sub(r'\s+', ' ', raw_line).strip()
                if not line:
                    continue
                if re.search(r'\d', line) and (
                    ':' in line or 'piece' in line.lower() or 'qty' in line.lower() or 'amount' in line.lower()
                ):
                    candidate_lines.append(line)
                if len(candidate_lines) >= 20:
                    break
            if candidate_lines:
                log_attachment_debug(debug, f"source_file={name} item_candidate_lines_count={len(candidate_lines)}")
                for idx, line in enumerate(candidate_lines, start=1):
                    log_attachment_debug(debug, f"source_file={name} candidate_line_{idx}={line}")

        return po_data
    except Exception as e:
        return {
            'po_number': None,
            'date': None,
            'vendor_name': None,
            'total_amount': None,
            'items': [],
            'raw_text': '',
            'po_candidates': extract_po_numbers(name),
            'source_file': name,
            'text_length': 0,
            'extraction_status': 'error',
            'error': str(e),
        }


# ============== Microsoft Graph API Functions ==============

def get_emails(access_token, folder="inbox", top=50, filter_unread=False):
    """Fetch emails from Outlook using Microsoft Graph API."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Build query parameters
    params = {
        '$top': top,
        '$select': 'id,subject,from,receivedDateTime,body,hasAttachments,isRead',
        '$orderby': 'receivedDateTime DESC'
    }
    
    if filter_unread:
        params['$filter'] = 'isRead eq false'
    
    url = f"{GRAPH_API_ENDPOINT}/me/mailFolders/{folder}/messages"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('value', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching emails: {str(e)}")
        return []


def get_email_attachments(access_token, message_id):
    """Get attachment names for an email."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    url = f"{GRAPH_API_ENDPOINT}/me/messages/{message_id}/attachments"
    
    try:
        response = requests.get(url, headers=headers, params={'$select': 'name'})
        response.raise_for_status()
        attachments = response.json().get('value', [])
        return [att.get('name', '') for att in attachments]
    except:
        return []


def get_email_attachments_with_content(access_token, message_id, max_attachments=8, max_size_mb=10, debug=False):
    """Get processable attachment metadata and content bytes from an email."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    url = f"{GRAPH_API_ENDPOINT}/me/messages/{message_id}/attachments"

    try:
        response = requests.get(url, headers=headers, params={'$select': 'id,name,size,contentType'})
        response.raise_for_status()
        all_attachments = response.json().get('value', [])
        log_attachment_debug(debug, f"message_id={message_id} listed_attachments={len(all_attachments)}")
    except requests.exceptions.RequestException as exc:
        response = getattr(exc, 'response', None)
        if response is not None:
            log_attachment_debug(
                debug,
                f"message_id={message_id} failed_to_list_attachments status={response.status_code} body={response.text[:500]}"
            )
        else:
            log_attachment_debug(debug, f"message_id={message_id} failed_to_list_attachments error={exc}")
        return []
    except Exception as exc:
        log_attachment_debug(debug, f"message_id={message_id} failed_to_list_attachments error={exc}")
        return []

    extracted_inputs = []
    for att in all_attachments[:max_attachments]:
        name = att.get('name', '')
        content_type = att.get('contentType', '')
        size = int(att.get('size') or 0)
        if size > max_size_mb * 1024 * 1024:
            log_attachment_debug(debug, f"message_id={message_id} skipped_oversize name={name} size={size}")
            continue
        if not _is_supported_attachment(name, content_type):
            log_attachment_debug(debug, f"message_id={message_id} skipped_unsupported name={name} content_type={content_type}")
            continue

        attachment_id = att.get('id')
        if not attachment_id:
            log_attachment_debug(debug, f"message_id={message_id} skipped_missing_attachment_id name={name}")
            continue

        detail_url = f"{GRAPH_API_ENDPOINT}/me/messages/{message_id}/attachments/{attachment_id}"
        try:
            detail = requests.get(detail_url, headers=headers)
            detail.raise_for_status()
            payload = detail.json()
            odata_type = payload.get('@odata.type')
            if odata_type and odata_type != '#microsoft.graph.fileAttachment':
                log_attachment_debug(debug, f"message_id={message_id} skipped_non_file_attachment name={name} odata_type={odata_type}")
                continue

            content_bytes = payload.get('contentBytes')
            if not content_bytes:
                log_attachment_debug(
                    debug,
                    f"message_id={message_id} missing_content_bytes name={name} odata_type={odata_type}"
                )
                continue

            decoded_bytes = base64.b64decode(content_bytes)
            log_attachment_debug(
                debug,
                f"message_id={message_id} fetched_attachment_bytes name={payload.get('name', name)} bytes={len(decoded_bytes)} content_type={payload.get('contentType', content_type)}"
            )

            extracted_inputs.append({
                'name': payload.get('name', name),
                'contentType': payload.get('contentType', content_type),
                'bytes': decoded_bytes
            })
        except requests.exceptions.RequestException as exc:
            response = getattr(exc, 'response', None)
            if response is not None:
                log_attachment_debug(
                    debug,
                    f"message_id={message_id} failed_to_fetch_attachment_bytes name={name} status={response.status_code} body={response.text[:500]}"
                )
            else:
                log_attachment_debug(debug, f"message_id={message_id} failed_to_fetch_attachment_bytes name={name} error={exc}")
            continue
        except Exception as exc:
            log_attachment_debug(debug, f"message_id={message_id} failed_to_fetch_attachment_bytes name={name} error={exc}")
            continue

    log_attachment_debug(debug, f"message_id={message_id} extracted_inputs={len(extracted_inputs)}")
    return extracted_inputs


def extract_po_data_from_attachments(access_token, message_id, debug=False):
    """Extract PO data from all supported attachments in one email."""
    if not OCR_AVAILABLE and not PDF_AVAILABLE:
        log_attachment_debug(debug, f"message_id={message_id} extraction_unavailable ocr={OCR_AVAILABLE} pdf={PDF_AVAILABLE}")
        return []

    attachments = get_email_attachments_with_content(access_token, message_id, debug=debug)
    results = []
    for attachment in attachments:
        log_attachment_debug(
            debug,
            f"message_id={message_id} processing_attachment name={attachment.get('name', '')} content_type={attachment.get('contentType', '')}"
        )
        po_data = extract_po_from_attachment(attachment, debug=debug)
        if po_data:
            if not po_data.get('po_number') and po_data.get('po_candidates'):
                po_data['po_number'] = po_data['po_candidates'][0]
            log_attachment_debug(
                debug,
                f"message_id={message_id} processed_attachment name={po_data.get('source_file', '')} status={po_data.get('extraction_status')} po_number={po_data.get('po_number')} po_candidates={po_data.get('po_candidates', [])} text_length={po_data.get('text_length')} error={po_data.get('error', '')}"
            )
            results.append(po_data)
        else:
            log_attachment_debug(debug, f"message_id={message_id} attachment_returned_none name={attachment.get('name', '')}")

    log_attachment_debug(debug, f"message_id={message_id} attachment_results={len(results)}")
    return results


def get_user_profile(access_token):
    """Get user profile information."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{GRAPH_API_ENDPOINT}/me", headers=headers)
        response.raise_for_status()
        return response.json()
    except:
        return None


def search_emails(access_token, query, top=50):
    """Search emails with a query string."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    params = {
        '$search': f'"{query}"',
        '$top': top,
        '$select': 'id,subject,from,receivedDateTime,body,hasAttachments'
    }
    
    url = f"{GRAPH_API_ENDPOINT}/me/messages"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('value', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching emails: {str(e)}")
        return []


# ============== Streamlit UI ==============

def main():
    st.set_page_config(
        page_title="Outlook PO Email Reader",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Outlook PO Email Reader")
    st.markdown("Classify and extract Purchase Order details from your Outlook emails")
    
    # Initialize session state
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'token_expires' not in st.session_state:
        st.session_state.token_expires = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'emails' not in st.session_state:
        st.session_state.emails = []
    if 'classified_emails' not in st.session_state:
        st.session_state.classified_emails = []
    if 'user_label_overrides' not in st.session_state:
        st.session_state.user_label_overrides = load_user_label_overrides()
    
    # Auto-capture authorization code from URL (after Microsoft redirect)
    query_params = st.query_params
    if 'code' in query_params and st.session_state.access_token is None:
        auth_code = query_params['code']
        with st.spinner("Logging in with Microsoft..."):
            token_data = get_token_from_code(auth_code)
            if token_data and 'access_token' in token_data:
                st.session_state.access_token = token_data['access_token']
                st.session_state.refresh_token = token_data.get('refresh_token')
                expires_in = token_data.get('expires_in', 3600)
                st.session_state.token_expires = datetime.now() + timedelta(seconds=expires_in)
                
                # Get user info
                user = get_user_profile(st.session_state.access_token)
                if user:
                    st.session_state.user_info = user
                
                # Clear URL parameters and refresh
                st.query_params.clear()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Login failed. Please check error details below and try again.")
                # Show debug info
                if token_data:
                    error_desc = token_data.get('error_description', 'Unknown error')
                    st.code(f"Error: {token_data.get('error', 'Unknown')}\nDetails: {error_desc}", language=None)
                st.query_params.clear()
    
    # Sidebar for authentication
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if st.session_state.access_token is None:
            st.warning("Not logged in")

            auth_issues = get_auth_config_issues()
            auth_ready = is_auth_configured()
            if not auth_ready:
                st.error(
                    "OAuth setup incomplete. Set environment variables: "
                    + ", ".join(auth_issues)
                )
                st.code(
                    "$env:AZURE_CLIENT_ID=\"<your-client-id>\"\n"
                    "$env:AZURE_CLIENT_SECRET=\"<your-client-secret>\"",
                    language="powershell"
                )
            
            st.markdown("### Login with Microsoft")
            st.markdown("Click the button below to login with your Microsoft account.")
            
            auth_url = get_auth_url()
            st.link_button("🔐 Login with Microsoft", auth_url, width='stretch', disabled=not auth_ready)
            
            st.markdown("---")
            st.caption("Or paste authorization code manually:")
            
            auth_code = st.text_input("Authorization Code:", type="password", label_visibility="collapsed", placeholder="Paste code here...")
            
            if st.button("🔓 Login", width='stretch', disabled=(not auth_code) or (not auth_ready)):
                with st.spinner("Authenticating..."):
                    token_data = get_token_from_code(auth_code)
                    if token_data and 'access_token' in token_data:
                        st.session_state.access_token = token_data['access_token']
                        st.session_state.refresh_token = token_data.get('refresh_token')
                        expires_in = token_data.get('expires_in', 3600)
                        st.session_state.token_expires = datetime.now() + timedelta(seconds=expires_in)
                        
                        # Get user info
                        user = get_user_profile(st.session_state.access_token)
                        if user:
                            st.session_state.user_info = user
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed. Please try again.")
        
        else:
            # Logged in state
            if st.session_state.user_info:
                st.success(f"Logged in as:")
                st.markdown(f"**{st.session_state.user_info.get('displayName', 'User')}**")
                st.caption(st.session_state.user_info.get('mail', ''))
            
            if st.session_state.token_expires:
                time_left = st.session_state.token_expires - datetime.now()
                if time_left.total_seconds() > 0:
                    st.caption(f"Token expires in: {int(time_left.total_seconds() / 60)} min")
                else:
                    st.warning("Token expired")
            
            if st.button("🚪 Logout", width='stretch'):
                st.session_state.access_token = None
                st.session_state.user_info = None
                st.session_state.emails = []
                st.session_state.classified_emails = []
                st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        email_count = st.slider("Emails to fetch:", 10, 100, 50)
        classification_threshold = st.slider("Classification threshold:", 1, 20, 8)
        filter_unread = st.checkbox("Only unread emails")
        extract_from_attachments = st.checkbox(
            "Extract PO data from PDF/image attachments",
            value=False,
            help="Reads supported PDF and image attachments, then uses text extraction and Tesseract OCR to capture PO numbers and related fields."
        )
        debug_attachment_extraction = st.checkbox(
            "Debug extraction in terminal (attachments + body)",
            value=False,
            help="Prints attachment/body extraction diagnostics, including table detection and row skip reasons for body parsing."
        )

        if extract_from_attachments and not OCR_AVAILABLE and not PDF_AVAILABLE:
            st.warning("Attachment extraction dependencies missing. Install requirements to enable OCR/PDF extraction.")
        
        st.markdown("---")
        st.header("🤖 Classifier")
        
        # Check if BERT is available
        if BERT_AVAILABLE:
            use_bert = st.checkbox("Use BERT classifier", value=True, help="More accurate but slower")
            if use_bert:
                st.caption("✅ BERT enabled")
            else:
                st.caption("📜 Using rule-based")
        else:
            use_bert = False
            st.caption("📜 Rule-based (install transformers for BERT)")
        
        # Store in session state
        if 'use_bert' not in st.session_state:
            st.session_state.use_bert = use_bert
        st.session_state.use_bert = use_bert
    
    # Main content area
    if st.session_state.access_token is None:
        st.info("👈 Please login using the sidebar to access your Outlook emails")
        
        # Show demo/instructions
        st.markdown("---")
        st.markdown("### How to use this app:")
        st.markdown("""
        1. **Login** with your Microsoft account using the sidebar
        2. **Fetch** your emails from Outlook
        3. **Classify** emails automatically to identify Purchase Orders
        4. **Extract** key details like PO numbers, item codes, dates
        5. **Export** results to CSV for further processing
        """)
        
        st.markdown("### Features:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📬 Email Reading")
            st.markdown("- Fetch recent emails\n- Search by keyword\n- Filter unread only")
        with col2:
            st.markdown("#### 🏷️ Classification")
            st.markdown("- PO keyword detection\n- Pattern matching\n- Confidence scoring")
        with col3:
            st.markdown("#### 📊 Extraction")
            st.markdown("- PO numbers\n- Item codes\n- Amounts & dates")
    
    else:
        # Logged in - show email tools
        
        # Tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs(["📬 Fetch Emails", "🔍 Search Emails", "📊 Results", "🤖 Train Model"])
        
        with tab1:
            st.header("Fetch Recent Emails")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                folder = st.selectbox("Select folder:", ["inbox", "sentitems", "drafts"])
            with col2:
                st.write("")
                st.write("")
                fetch_btn = st.button("📥 Fetch Emails", width='stretch', type="primary")
            
            if fetch_btn:
                with st.spinner(f"Fetching {email_count} emails from {folder}..."):
                    emails = get_emails(
                        st.session_state.access_token,
                        folder=folder,
                        top=email_count,
                        filter_unread=filter_unread
                    )
                    
                    if emails:
                        st.session_state.emails = emails
                        st.success(f"Fetched {len(emails)} emails")
                    else:
                        st.warning("No emails found")
            
            # Show fetched emails
            if st.session_state.emails:
                st.markdown("---")
                st.subheader(f"📧 {len(st.session_state.emails)} Emails Loaded")
                
                if st.button("🏷️ Classify All Emails", type="primary", width='stretch'):
                    with st.spinner("Classifying emails..."):
                        classified = []
                        progress = st.progress(0)
                        
                        # Initialize classifier
                        bert_classifier = None
                        if st.session_state.get('use_bert') and BERT_AVAILABLE:
                            try:
                                model_path = "models/po_classifier"
                                if os.path.exists(model_path):
                                    bert_classifier = HybridClassifier(model_path)
                                    st.info("Using trained BERT classifier")
                                else:
                                    st.warning("No trained BERT model found. Using rule-based classifier.")
                            except Exception as e:
                                st.warning(f"BERT unavailable, using rules: {e}")
                        
                        for idx, email in enumerate(st.session_state.emails):
                            subject = email.get('subject', '')
                            message_id = email.get('id', '')
                            body_content = email.get('body', {}).get('content', '')
                            body_text = clean_email_body_text(body_content)
                            
                            # Get attachments if any
                            attachments = []
                            attachment_po_data = []
                            if email.get('hasAttachments'):
                                attachments = get_email_attachments(
                                    st.session_state.access_token,
                                    email.get('id')
                                )
                                if extract_from_attachments and (OCR_AVAILABLE or PDF_AVAILABLE):
                                    attachment_po_data = extract_po_data_from_attachments(
                                        st.session_state.access_token,
                                        email.get('id'),
                                        debug=debug_attachment_extraction
                                    )

                            attachment_po_numbers = sorted({
                                po_number
                                for parsed_attachment in attachment_po_data
                                for po_number in ([parsed_attachment.get('po_number')] if parsed_attachment.get('po_number') else []) + parsed_attachment.get('po_candidates', [])
                                if po_number
                            })
                            
                            # Use BERT or rule-based classification
                            if bert_classifier:
                                # Always compute rule signal as a safety net for obvious PO formats.
                                rule_score, keywords, patterns = calculate_po_score(subject, body_text, attachments)
                                result = bert_classifier.classify(subject, body_text, attachments)
                                po_score = float(result.get('po_score', result.get('score', 0.0)))
                                bert_score = int(po_score * 20)
                                score = max(bert_score, rule_score)
                                confidence, _ = get_confidence_level(score)
                                is_po = (po_score >= 0.6) or (rule_score >= classification_threshold)
                                method = 'BERT+RULES'
                            else:
                                score, keywords, patterns = calculate_po_score(subject, body_text, attachments)
                                confidence, icon = get_confidence_level(score)
                                is_po = score >= classification_threshold
                                method = 'RULES'

                            if attachment_po_numbers:
                                score += 6
                                is_po = True
                                confidence, _ = get_confidence_level(score)
                                method = f"{method}+ATTACH_EXTRACT"

                            # Apply manual user labels (if available) as final override.
                            manual_label = st.session_state.user_label_overrides.get(message_id)
                            if manual_label is not None:
                                is_po = bool(manual_label)
                                method = f"{method}+USER_LABEL"
                                if is_po:
                                    confidence = 'HIGH'
                                    icon = '🟢'
                                else:
                                    confidence = 'NOT_PO'
                                    icon = '⚪'

                            merged_po_numbers = sorted(set(
                                extract_po_numbers(f"{subject} {body_text}") + attachment_po_numbers
                            ))
                            supplier_name = extract_supplier_name_from_email_body(body_text)
                            if supplier_name in {'Supplier', 'Supplier Team', 'Vendor', 'Vendor Team'}:
                                supplier_name = None
                            body_items = extract_items_from_email_body(
                                body_text,
                                debug=debug_attachment_extraction,
                                message_id=message_id,
                                raw_html=body_content,
                            )
                            
                            _, icon = get_confidence_level(score)
                            if manual_label is not None:
                                if is_po:
                                    icon = '🟢'
                                else:
                                    icon = '⚪'
                            
                            classified.append({
                                'id': email.get('id'),
                                'subject': subject,
                                'from': email.get('from', {}).get('emailAddress', {}).get('address', ''),
                                'from_name': email.get('from', {}).get('emailAddress', {}).get('name', ''),
                                'date': email.get('receivedDateTime', ''),
                                'body': body_text[:500],
                                'has_attachments': email.get('hasAttachments', False),
                                'attachments': attachments,
                                'score': score,
                                'confidence': confidence,
                                'icon': icon,
                                'is_po': is_po,
                                'method': method,
                                'po_numbers': merged_po_numbers,
                                'item_codes': extract_item_codes(f"{subject} {body_text}"),
                                'attachment_po_data': attachment_po_data,
                                'attachment_po_numbers': attachment_po_numbers,
                                'body_items': body_items,
                                'supplier_name': supplier_name,
                                'matched_keywords': keywords,
                                'matched_patterns': patterns,
                                'user_label': manual_label,
                            })
                            
                            progress.progress((idx + 1) / len(st.session_state.emails))
                        
                        st.session_state.classified_emails = classified
                        po_count = sum(1 for e in classified if e['is_po'])
                        st.success(f"Classified {len(classified)} emails. Found {po_count} potential PO emails!")
                
                # Display email list
                for email in st.session_state.emails[:10]:  # Show first 10
                    with st.expander(f"📧 {email.get('subject', 'No Subject')[:60]}..."):
                        st.markdown(f"**From:** {email.get('from', {}).get('emailAddress', {}).get('address', 'Unknown')}")
                        st.markdown(f"**Date:** {email.get('receivedDateTime', 'Unknown')}")
                        if email.get('hasAttachments'):
                            st.markdown("📎 Has attachments")
                
                if len(st.session_state.emails) > 10:
                    st.caption(f"... and {len(st.session_state.emails) - 10} more emails")
        
        with tab2:
            st.header("Search Emails")
            
            search_query = st.text_input("Search query:", placeholder="e.g., purchase order, MEL2025PO")
            
            if st.button("🔍 Search", disabled=not search_query):
                with st.spinner(f"Searching for '{search_query}'..."):
                    results = search_emails(st.session_state.access_token, search_query, top=email_count)
                    
                    if results:
                        st.session_state.emails = results
                        st.success(f"Found {len(results)} emails matching '{search_query}'")
                    else:
                        st.warning("No emails found matching your search")
        
        with tab3:
            st.header("Classification Results")
            
            if not st.session_state.classified_emails:
                st.info("No classified emails yet. Fetch emails and click 'Classify All Emails'")
            else:
                # Summary metrics
                total = len(st.session_state.classified_emails)
                po_emails = [e for e in st.session_state.classified_emails if e['is_po']]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Emails", total)
                with col2:
                    st.metric("PO Emails", len(po_emails))
                with col3:
                    high_conf = sum(1 for e in po_emails if e['confidence'] == 'HIGH')
                    st.metric("High Confidence", high_conf)
                with col4:
                    unique_pos = set()
                    for e in po_emails:
                        unique_pos.update(e['po_numbers'])
                    st.metric("Unique POs", len(unique_pos))
                
                st.markdown("---")
                
                # Filter options
                show_only_po = st.checkbox("Show only PO emails", value=True)
                
                display_emails = po_emails if show_only_po else st.session_state.classified_emails
                
                # Display classified emails
                for email in display_emails:
                    confidence_color = {
                        'HIGH': 'green',
                        'MEDIUM': 'orange', 
                        'LOW': 'red',
                        'NOT_PO': 'gray'
                    }.get(email['confidence'], 'gray')
                    
                    with st.expander(f"{email['icon']} {email['subject'][:50]}... | Score: {email['score']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Subject:** {email['subject']}")
                            st.markdown(f"**From:** {email['from_name']} ({email['from']})")
                            st.markdown(f"**Date:** {email['date']}")
                            
                            if email['po_numbers']:
                                st.markdown(f"**PO Numbers:** `{', '.join(email['po_numbers'])}`")
                            if email.get('attachment_po_numbers'):
                                st.markdown(f"**PO Numbers (Attachment OCR/PDF):** `{', '.join(email['attachment_po_numbers'])}`")
                        
                        with col2:
                            st.markdown(f"**Confidence:** :{confidence_color}[{email['confidence']}]")
                            st.markdown(f"**Score:** {email['score']}")
                            if email['has_attachments']:
                                st.markdown("📎 **Attachments:**")
                                for att in email['attachments'][:3]:
                                    st.caption(f"  - {att}")
                        
                        st.markdown("**Body Preview:**")
                        st.text(email['body'][:300] + "..." if len(email['body']) > 300 else email['body'])

                        if email.get('body_items'):
                            st.markdown("**Body Item Extraction:**")
                            body_rows = []
                            for item in email['body_items'][:20]:
                                master = item.get('master_columns') or _empty_master_row()
                                body_rows.append({
                                    'Type': master.get('Type', item.get('type', '')),
                                    'Contract No': master.get('Contract No', item.get('contract_no', '')),
                                    'Item Category': master.get('Item Category', item.get('item_category', item.get('product_name', ''))),
                                    '5lb': master.get('5lb', None),
                                    'First Size': master.get('First Size', None),
                                    'Up To 1Mth': master.get('Up To 1Mth', None),
                                    'Up To 3Mth': master.get('Up To 3Mth', None),
                                    '3-6 Mths': master.get('3-6 Mths', None),
                                    '6-9 Mths': master.get('6-9 Mths', None),
                                    '9-12 Mths': master.get('9-12 Mths', None),
                                    '12-18 Mths': master.get('12-18 Mths', None),
                                    '1.5-2 Yrs': master.get('1.5-2 Yrs', None),
                                    'Total': master.get('Total', item.get('amount', None)),
                                    'Source': item.get('source', ''),
                                })
                            body_df = pd.DataFrame(body_rows)
                            for col in MASTER_NUMERIC_COLUMNS:
                                if col in body_df.columns:
                                    body_df[col] = pd.to_numeric(body_df[col], errors='coerce').round().astype('Int64')
                            st.dataframe(body_df, width='stretch')

                        if email.get('attachment_po_data'):
                            st.markdown("**Attachment Extraction:**")
                            for po in email['attachment_po_data']:
                                details = []
                                if po.get('extraction_status'):
                                    details.append(f"Status: {po['extraction_status']}")
                                if po.get('po_number'):
                                    details.append(f"PO: {po['po_number']}")
                                elif po.get('po_candidates'):
                                    details.append(f"PO candidates: {', '.join(po['po_candidates'][:3])}")
                                if po.get('date'):
                                    details.append(f"Date: {po['date']}")
                                if po.get('vendor_name'):
                                    details.append(f"Vendor: {po['vendor_name']}")
                                if po.get('text_length') is not None:
                                    details.append(f"Text chars: {po['text_length']}")
                                if po.get('items'):
                                    qty_total = sum(float(item.get('quantity', 0) or 0) for item in po['items'])
                                    details.append(f"Items: {len(po['items'])}")
                                    details.append(f"Qty: {qty_total:g}")
                                if po.get('error'):
                                    details.append(f"Error: {po['error']}")
                                st.caption(f"• {po.get('source_file', 'attachment')} | {' | '.join(details) if details else 'No structured fields found'}")
                                if po.get('items'):
                                    item_rows = []
                                    for item in po['items'][:10]:
                                        item_rows.append({
                                            'Item Name': item.get('product_name') or item.get('description', ''),
                                        })
                                    st.dataframe(pd.DataFrame(item_rows), width='stretch')
                
                # Export functionality
                st.markdown("---")
                st.subheader("📥 Export Results")
                
                if st.button("Export to CSV", type="primary"):
                    export_data = []
                    for email in po_emails:
                        base_row = {
                            'Subject': email['subject'],
                            'From': email['from'],
                            'From Name': email['from_name'],
                            'Date': email['date'],
                            'PO Numbers': '; '.join(email['po_numbers']),
                            'Confidence': email['confidence'],
                            'Score': email['score'],
                            'Has Attachments': email['has_attachments'],
                            'Attachments': '; '.join(email['attachments']),
                            'Attachment PO Numbers': '; '.join(email.get('attachment_po_numbers', [])),
                            'Attachment Extraction Count': len(email.get('attachment_po_data', [])),
                            'Body Preview': email['body'][:200],
                        }

                        all_items = []
                        for item in email.get('body_items', []):
                            master = item.get('master_columns') or _empty_master_row()
                            all_items.append({
                                'Item Name': str(item.get('product_name') or item.get('description', '')),
                                'Type': str(master.get('Type', item.get('type', ''))),
                                'Contract No': str(master.get('Contract No', item.get('contract_no', ''))),
                                'Item Category': str(master.get('Item Category', item.get('item_category', item.get('product_name', '')))),
                                '5lb': master.get('5lb'),
                                'First Size': master.get('First Size'),
                                'Up To 1Mth': master.get('Up To 1Mth'),
                                'Up To 3Mth': master.get('Up To 3Mth'),
                                '3-6 Mths': master.get('3-6 Mths'),
                                '6-9 Mths': master.get('6-9 Mths'),
                                '9-12 Mths': master.get('9-12 Mths'),
                                '12-18 Mths': master.get('12-18 Mths'),
                                '1.5-2 Yrs': master.get('1.5-2 Yrs'),
                                'Total': master.get('Total'),
                                'Item Source': 'Body',
                            })

                        for po in email.get('attachment_po_data', []):
                            for item in po.get('items', []):
                                all_items.append({
                                    'Item Name': str(item.get('product_name') or item.get('description', '')),
                                    'Type': '',
                                    'Contract No': '',
                                    'Item Category': '',
                                    '5lb': '',
                                    'First Size': '',
                                    'Up To 1Mth': '',
                                    'Up To 3Mth': '',
                                    '3-6 Mths': '',
                                    '6-9 Mths': '',
                                    '9-12 Mths': '',
                                    '12-18 Mths': '',
                                    '1.5-2 Yrs': '',
                                    'Total': '',
                                    'Item Source': f"Attachment:{po.get('source_file', 'file')}",
                                })

                        if all_items:
                            for item_row in all_items:
                                export_data.append({**base_row, **item_row})
                        else:
                            export_data.append({
                                **base_row,
                                'Item Name': '',
                                'Type': '',
                                'Contract No': '',
                                'Item Category': '',
                                '5lb': '',
                                'First Size': '',
                                'Up To 1Mth': '',
                                'Up To 3Mth': '',
                                '3-6 Mths': '',
                                '6-9 Mths': '',
                                '9-12 Mths': '',
                                '12-18 Mths': '',
                                '1.5-2 Yrs': '',
                                'Total': '',
                                'Item Source': '',
                            })
                    
                    df = pd.DataFrame(export_data)
                    for col in MASTER_NUMERIC_COLUMNS:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"po_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.dataframe(df, width='stretch')
        
        with tab4:
            st.header("🤖 Train BERT Classifier")
            
            if not BERT_AVAILABLE:
                st.error("BERT not available. Install with: `pip install transformers torch`")
                st.code("pip install transformers torch", language="bash")
            else:
                st.markdown("""
                Train a custom BERT model on your labeled emails for better accuracy.
                
                **How it works:**
                1. Label some emails as PO or Not PO
                2. Train the model on your labeled data
                3. The model learns your specific PO patterns
                4. Use the trained model for classification
                """)
                
                # Initialize training data in session state
                if 'training_data' not in st.session_state:
                    st.session_state.training_data = []
                
                st.markdown("---")
                st.subheader("📝 Step 1: Label Emails")
                
                if st.session_state.classified_emails:
                    st.markdown("Review classified emails and correct any mistakes:")
                    label_limit = min(50, len(st.session_state.classified_emails))
                    st.caption(f"Showing latest {label_limit} emails for labeling")
                    
                    # Let user label emails
                    for idx, email in enumerate(st.session_state.classified_emails[:label_limit]):
                        col1, col2, col3 = st.columns([4, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{email['subject'][:50]}...**")
                            st.caption(f"From: {email['from']}")
                        
                        with col2:
                            current_label = email.get('user_label', email['is_po'])
                            is_po = st.checkbox("PO", value=current_label, key=f"label_{idx}")
                            email['user_label'] = is_po
                        
                        with col3:
                            st.caption(f"Score: {email['score']}")
                    
                    if st.button("✅ Save Labels", type="primary"):
                        # Add to training data
                        for email in st.session_state.classified_emails:
                            if 'user_label' in email:
                                if email.get('id'):
                                    st.session_state.user_label_overrides[email['id']] = bool(email['user_label'])
                                email['is_po'] = bool(email['user_label'])
                                if email['is_po']:
                                    email['confidence'] = 'HIGH'
                                    email['icon'] = '🟢'
                                else:
                                    email['confidence'] = 'NOT_PO'
                                    email['icon'] = '⚪'
                                st.session_state.training_data.append({
                                    'subject': email['subject'],
                                    'body': email['body'],
                                    'is_po': email['user_label']
                                })
                        persisted = save_user_label_overrides(st.session_state.user_label_overrides)
                        if persisted:
                            st.success(
                                f"Saved {len(st.session_state.training_data)} labeled examples and "
                                f"{len(st.session_state.user_label_overrides)} persistent label overrides"
                            )
                        else:
                            st.warning(
                                "Labels were saved for current session, but writing persistent overrides failed."
                            )
                else:
                    st.info("Fetch and classify emails first, then come back here to label them")
                
                st.markdown("---")
                st.subheader("📊 Step 2: Training Data")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Labeled Examples", len(st.session_state.training_data))
                with col2:
                    po_count = sum(1 for d in st.session_state.training_data if d['is_po'])
                    st.metric("PO / Not PO", f"{po_count} / {len(st.session_state.training_data) - po_count}")
                
                # Upload training data
                st.markdown("**Or upload training data (JSON):**")
                uploaded_file = st.file_uploader("Upload training data", type=['json'])
                
                if uploaded_file:
                    try:
                        data = json.load(uploaded_file)
                        if 'examples' in data:
                            st.session_state.training_data.extend(data['examples'])
                        elif isinstance(data, list):
                            st.session_state.training_data.extend(data)
                        st.success(f"Loaded {len(data.get('examples', data))} examples")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                
                # Download template
                if st.button("📥 Download Template"):
                    template = {
                        "instructions": "Add examples. Set is_po to true for PO emails, false otherwise.",
                        "examples": [
                            {"subject": "PO#12345 - Order Confirmation", "body": "Please find attached...", "is_po": True},
                            {"subject": "Weekly Newsletter", "body": "Check out our updates...", "is_po": False}
                        ]
                    }
                    st.download_button(
                        "📄 Download Template JSON",
                        json.dumps(template, indent=2),
                        "training_template.json",
                        "application/json"
                    )
                
                st.markdown("---")
                st.subheader("🚀 Step 3: Train Model")
                
                min_samples = 20
                can_train = len(st.session_state.training_data) >= min_samples
                
                if not can_train:
                    st.warning(f"Need at least {min_samples} labeled examples ({len(st.session_state.training_data)} so far)")
                
                epochs = st.slider("Training epochs:", 1, 5, 3)
                
                if st.button("🎯 Train Model", type="primary", disabled=not can_train):
                    with st.spinner("Training BERT model... This may take a few minutes..."):
                        try:
                            classifier = HybridClassifier()
                            
                            # Train
                            classifier.train_bert(
                                st.session_state.training_data,
                                output_path="models/po_classifier"
                            )
                            
                            st.success("✅ Model trained and saved!")
                            st.info("The app will now use your trained model for classification.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
