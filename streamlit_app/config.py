"""
Configuration file for Outlook PO Email Reader

Update the values below with your Azure AD app credentials.
You can also set these as environment variables instead.
"""

import os


def _load_env_file(env_path):
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

# ============== Azure AD App Configuration ==============
# Register your app at https://portal.azure.com

# Application (client) ID from Azure AD app registration
# Set via environment variable AZURE_CLIENT_ID
CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")

# Client secret value (create in Certificates & secrets)
# IMPORTANT: Never commit secrets to version control!
# Set via environment variable AZURE_CLIENT_SECRET
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")

# Redirect URI (must match the one in Azure AD app registration)
REDIRECT_URI = os.getenv("AZURE_REDIRECT_URI", "http://localhost:8502")

# Tenant ID options:
# - "common" = Any Microsoft account (personal or work/school) - RECOMMENDED
# - "consumers" = Personal Microsoft accounts only (outlook.com, hotmail.com, live.com)
# - "organizations" = Work/school accounts only
# - Your tenant ID = Specific organization only
#
# IMPORTANT: Your Azure AD app must be registered with:
# "Accounts in any organizational directory and personal Microsoft accounts"
TENANT_ID = os.getenv("AZURE_TENANT_ID", "common")  # Supports both personal and work/school accounts


# ============== App Settings ==============

# Default number of emails to fetch
DEFAULT_EMAIL_COUNT = 50

# Default classification threshold (emails with score >= this are classified as PO)
DEFAULT_CLASSIFICATION_THRESHOLD = 5

# Output folder for exported files
OUTPUT_FOLDER = "output"


# ============== PO Detection Keywords ==============
# Add or remove keywords to customize PO detection

PO_KEYWORDS = [
    # PO identifiers
    'purchase order', 'po#', 'po number', 'p.o.', 'p.o', 'po:',
    
    # Order-related terms
    'order confirmation', 'order acknowledgment', 'order acknowledgement',
    'order placed', 'new order', 'order details', 'order number',
    
    # Procurement terms
    'procurement', 'requisition', 'indent', 'supply order',
    
    # Commercial terms
    'quotation', 'quote', 'invoice', 'proforma', 'pro-forma',
    
    # Processing terms
    'packing', 'shipment', 'delivery', 'dispatch', 'trims',
    
    # Action terms
    'please confirm', 'kindly confirm', 'attached po', 'attached purchase order',
    
    # Custom PO formats (add your organization's formats here)
    'mel2025po', 'mel2024po', 'mel2026po'
]

# Regex patterns for detecting PO numbers
PO_NUMBER_PATTERNS = [
    r'\bP[O0]\s*#?\s*[:\-]?\s*([A-Z0-9-]*\d[A-Z0-9-]*)\b',            # PO#12345, PO: 12345 (must include a digit)
    r'MEL\d{4}PO\d+',                       # MEL2025PO12232
    r'[A-Z]{2,4}\d{4}PO\d+',                # XXX2024PO12345
    r'\bPurchase\s*Order\s*#?\s*[:\-]?\s*([A-Z0-9-]*\d[A-Z0-9-]*)\b', # Purchase Order #12345 (must include a digit)
]
