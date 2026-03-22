import streamlit as st
import pandas as pd
import re
import os
import json
import base64
from datetime import datetime, timedelta
import requests
from io import BytesIO

# test change

# Microsoft Graph API endpoints
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import authentication module
from auth import get_auth_url, get_token_from_code, refresh_access_token

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


def log_attachment_debug(enabled, message):
    """Print attachment extraction debug details to the terminal when enabled."""
    if enabled:
        print(f"[ATTACH_DEBUG] {message}")


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

    combined_item_pattern = r'^(\d+)\.?\s+(.+?)\s+(\d+)\s+\$\s*([\d,]+\.?\d*)\s+\$\s*([\d,]+\.?\d*)$'
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(combined_item_pattern, line)
        if match:
            qty = int(match.group(3))
            description = match.group(2).strip()
            rate = float(match.group(4).replace(',', ''))
            amount = float(match.group(5).replace(',', ''))
            if not any(x in description.lower() for x in ['total', 'subtotal', 'payment', 'balance']):
                po_data['items'].append({
                    'quantity': qty,
                    'description': description,
                    'price': rate,
                    'amount': amount
                })

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


def extract_po_from_attachment(attachment):
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
        po_data = extract_po_from_attachment(attachment)
        if po_data:
            if not po_data.get('po_number') and po_data.get('po_candidates'):
                po_data['po_number'] = po_data['po_candidates'][0]
            log_attachment_debug(
                debug,
                f"message_id={message_id} processed_attachment name={po_data.get('source_file', '')} status={po_data.get('extraction_status')} po_number={po_data.get('po_number')} po_candidates={po_data.get('po_candidates', [])} text_length={po_data.get('text_length')}"
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
            
            st.markdown("### Login with Microsoft")
            st.markdown("Click the button below to login with your Microsoft account.")
            
            auth_url = get_auth_url()
            st.link_button("🔐 Login with Microsoft", auth_url, width='stretch')
            
            st.markdown("---")
            st.caption("Or paste authorization code manually:")
            
            auth_code = st.text_input("Authorization Code:", type="password", label_visibility="collapsed", placeholder="Paste code here...")
            
            if st.button("🔓 Login", width='stretch', disabled=not auth_code):
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
            "Debug attachment extraction in terminal",
            value=False,
            help="Prints attachment listing, byte download, OCR/PDF extraction status, and PO candidates to the terminal."
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
                            body_content = email.get('body', {}).get('content', '')
                            # Strip HTML tags
                            body_text = re.sub(r'<[^>]+>', '', body_content)
                            
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

                            merged_po_numbers = sorted(set(
                                extract_po_numbers(f"{subject} {body_text}") + attachment_po_numbers
                            ))
                            
                            _, icon = get_confidence_level(score)
                            
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
                                'matched_keywords': keywords,
                                'matched_patterns': patterns,
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
                            if email['item_codes']:
                                st.markdown(f"**Item Codes:** `{', '.join(email['item_codes'])}`")
                        
                        with col2:
                            st.markdown(f"**Confidence:** :{confidence_color}[{email['confidence']}]")
                            st.markdown(f"**Score:** {email['score']}")
                            if email['has_attachments']:
                                st.markdown("📎 **Attachments:**")
                                for att in email['attachments'][:3]:
                                    st.caption(f"  - {att}")
                        
                        st.markdown("**Body Preview:**")
                        st.text(email['body'][:300] + "..." if len(email['body']) > 300 else email['body'])

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
                                    qty_total = sum(item.get('quantity', 0) for item in po['items'])
                                    details.append(f"Qty: {qty_total}")
                                if po.get('error'):
                                    details.append(f"Error: {po['error']}")
                                st.caption(f"• {po.get('source_file', 'attachment')} | {' | '.join(details) if details else 'No structured fields found'}")
                
                # Export functionality
                st.markdown("---")
                st.subheader("📥 Export Results")
                
                if st.button("Export to CSV", type="primary"):
                    export_data = []
                    for email in po_emails:
                        export_data.append({
                            'Subject': email['subject'],
                            'From': email['from'],
                            'From Name': email['from_name'],
                            'Date': email['date'],
                            'PO Numbers': '; '.join(email['po_numbers']),
                            'Item Codes': '; '.join(email['item_codes']),
                            'Confidence': email['confidence'],
                            'Score': email['score'],
                            'Has Attachments': email['has_attachments'],
                            'Attachments': '; '.join(email['attachments']),
                            'Attachment PO Numbers': '; '.join(email.get('attachment_po_numbers', [])),
                            'Attachment Extraction Count': len(email.get('attachment_po_data', [])),
                            'Body Preview': email['body'][:200]
                        })
                    
                    df = pd.DataFrame(export_data)
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
                                st.session_state.training_data.append({
                                    'subject': email['subject'],
                                    'body': email['body'],
                                    'is_po': email['user_label']
                                })
                        st.success(f"Saved {len(st.session_state.training_data)} labeled examples")
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
