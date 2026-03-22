"""
Microsoft OAuth2 Authentication Module for Outlook Email Access
Uses Microsoft Graph API with OAuth2 authorization code flow
"""
import os
import urllib.parse
import requests
from typing import Optional, Dict

# ============== Configuration ==============
# To use this app, you need to register an app in Azure AD:
# 1. Go to https://portal.azure.com
# 2. Navigate to Azure Active Directory > App registrations
# 3. Click "New registration"
# 4. Name your app (e.g., "Outlook PO Reader")
# 5. Set redirect URI to: http://localhost:8501 (for Streamlit)
# 6. After creation, note the Application (client) ID
# 7. Go to "Certificates & secrets" and create a new client secret
# 8. Go to "API permissions" and add:
#    - Microsoft Graph > Delegated permissions > Mail.Read
#    - Microsoft Graph > Delegated permissions > User.Read
# 9. Click "Grant admin consent" if you have admin access

# Load from environment variables or config file
try:
    from config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TENANT_ID
except ImportError:
    CLIENT_ID = os.getenv("OUTLOOK_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
    CLIENT_SECRET = os.getenv("OUTLOOK_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
    REDIRECT_URI = os.getenv("OUTLOOK_REDIRECT_URI", "http://localhost:8501")
    TENANT_ID = os.getenv("OUTLOOK_TENANT_ID", "common")  # Use "common" for all account types

# Microsoft OAuth2 endpoints
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
AUTH_ENDPOINT = f"{AUTHORITY}/oauth2/v2.0/authorize"
TOKEN_ENDPOINT = f"{AUTHORITY}/oauth2/v2.0/token"

# Required scopes for reading emails
SCOPES = [
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/User.Read",
    "offline_access"  # For refresh token
]


def get_auth_config_issues() -> list:
    """Return a list of missing OAuth configuration fields."""
    issues = []
    if not CLIENT_ID or CLIENT_ID.strip() in {"", "YOUR_CLIENT_ID_HERE"}:
        issues.append("AZURE_CLIENT_ID")
    if not CLIENT_SECRET or CLIENT_SECRET.strip() in {"", "YOUR_CLIENT_SECRET_HERE"}:
        issues.append("AZURE_CLIENT_SECRET")
    if not REDIRECT_URI or not REDIRECT_URI.strip():
        issues.append("REDIRECT_URI")
    return issues


def is_auth_configured() -> bool:
    """Check whether required OAuth config values are present."""
    return len(get_auth_config_issues()) == 0


def get_auth_url() -> str:
    """
    Generate the Microsoft OAuth2 authorization URL.
    User should open this URL in browser to login.
    
    Returns:
        Authorization URL string
    """
    if not is_auth_configured():
        return ""

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "response_mode": "query",
        "scope": " ".join(SCOPES),
        "state": "outlook_po_reader"  # CSRF protection token
    }
    
    auth_url = f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"
    return auth_url


def get_token_from_code(auth_code: str) -> Optional[Dict]:
    """
    Exchange authorization code for access token.
    
    Args:
        auth_code: The authorization code from the redirect URL
    
    Returns:
        Token data dict with access_token, refresh_token, expires_in
        or None if failed
    """
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "scope": " ".join(SCOPES)
    }
    
    try:
        response = requests.post(TOKEN_ENDPOINT, data=data)
        result = response.json()
        if response.status_code == 200:
            return result
        else:
            # Return error info for debugging
            print(f"Token exchange error: {result}")
            return result  # Contains 'error' and 'error_description'
    except requests.exceptions.RequestException as e:
        print(f"Token exchange error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
            try:
                return e.response.json()
            except:
                pass
        return {'error': 'request_failed', 'error_description': str(e)}


def refresh_access_token(refresh_token: str) -> Optional[Dict]:
    """
    Refresh an expired access token using the refresh token.
    
    Args:
        refresh_token: The refresh token from initial authentication
    
    Returns:
        New token data dict or None if failed
    """
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": " ".join(SCOPES)
    }
    
    try:
        response = requests.post(TOKEN_ENDPOINT, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Token refresh error: {str(e)}")
        return None


def validate_token(access_token: str) -> bool:
    """
    Validate if an access token is still valid.
    
    Args:
        access_token: The access token to validate
    
    Returns:
        True if valid, False otherwise
    """
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        response = requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers=headers
        )
        return response.status_code == 200
    except:
        return False


# ============== Interactive Device Flow (Alternative) ==============
# This flow is useful for apps that can't open a browser

def get_device_code() -> Optional[Dict]:
    """
    Start device code flow for authentication.
    Useful for headless environments.
    
    Returns:
        Device code response with user_code and verification_uri
    """
    data = {
        "client_id": CLIENT_ID,
        "scope": " ".join(SCOPES)
    }
    
    device_code_endpoint = f"{AUTHORITY}/oauth2/v2.0/devicecode"
    
    try:
        response = requests.post(device_code_endpoint, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Device code error: {str(e)}")
        return None


def poll_for_token(device_code: str, interval: int = 5, timeout: int = 300) -> Optional[Dict]:
    """
    Poll for token after user completes device code authentication.
    
    Args:
        device_code: The device code from get_device_code()
        interval: Polling interval in seconds
        timeout: Maximum time to wait in seconds
    
    Returns:
        Token data dict or None if failed/timeout
    """
    import time
    
    data = {
        "client_id": CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
    }
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.post(TOKEN_ENDPOINT, data=data)
            result = response.json()
            
            if "access_token" in result:
                return result
            
            error = result.get("error")
            if error == "authorization_pending":
                time.sleep(interval)
                continue
            elif error == "slow_down":
                time.sleep(interval + 5)
                continue
            else:
                print(f"Token polling error: {error}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Polling error: {str(e)}")
            return None
    
    print("Token polling timeout")
    return None


# ============== Setup Instructions ==============

def print_setup_instructions():
    """Print instructions for setting up Azure AD app."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    MICROSOFT AZURE AD SETUP INSTRUCTIONS                  ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  1. Go to https://portal.azure.com                                       ║
    ║                                                                          ║
    ║  2. Navigate to: Azure Active Directory > App registrations              ║
    ║                                                                          ║
    ║  3. Click "New registration"                                             ║
    ║     - Name: "Outlook PO Reader"                                          ║
    ║     - Supported account types: Choose based on your needs                ║
    ║       * "Personal Microsoft accounts only" for personal Outlook.com      ║
    ║       * "Accounts in any organizational directory" for work/school       ║
    ║     - Redirect URI: Web > http://localhost:8501                          ║
    ║                                                                          ║
    ║  4. After creation, copy the "Application (client) ID"                   ║
    ║                                                                          ║
    ║  5. Go to "Certificates & secrets" > "New client secret"                 ║
    ║     - Copy the secret value (shown only once!)                           ║
    ║                                                                          ║
    ║  6. Go to "API permissions" > "Add a permission" > "Microsoft Graph"     ║
    ║     - Add Delegated permissions:                                         ║
    ║       * Mail.Read                                                        ║
    ║       * User.Read                                                        ║
    ║                                                                          ║
    ║  7. Set environment variables or update config.py:                       ║
    ║     - OUTLOOK_CLIENT_ID=your_client_id                                   ║
    ║     - OUTLOOK_CLIENT_SECRET=your_client_secret                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    # Print setup instructions when run directly
    print_setup_instructions()
    
    print("\nCurrent Configuration:")
    print(f"  CLIENT_ID: {'(set)' if CLIENT_ID != 'YOUR_CLIENT_ID_HERE' else '(not set)'}")
    print(f"  CLIENT_SECRET: {'(set)' if CLIENT_SECRET != 'YOUR_CLIENT_SECRET_HERE' else '(not set)'}")
    print(f"  REDIRECT_URI: {REDIRECT_URI}")
    print(f"  TENANT_ID: {TENANT_ID}")
    
    if CLIENT_ID != 'YOUR_CLIENT_ID_HERE':
        print("\nAuthorization URL:")
        print(get_auth_url())
