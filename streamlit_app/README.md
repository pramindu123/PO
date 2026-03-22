# Outlook PO Email Reader - Streamlit App

A Streamlit web application that connects to your Outlook inbox via Microsoft Graph API to classify and extract Purchase Order (PO) details from emails.

## Features

- 🔐 **Microsoft OAuth Login** - Secure authentication with your Microsoft account
- 📬 **Email Fetching** - Fetch emails from inbox, sent items, or drafts
- 🔍 **Email Search** - Search emails by keyword
- 🏷️ **PO Classification** - Automatically identify PO-related emails using keyword and pattern matching
- 📊 **Detail Extraction** - Extract PO numbers, item codes, and other key information
- 🖼️ **Attachment OCR Extraction** - Extract PO data (especially PO number) from PDF and image attachments using text extraction and Tesseract OCR
- 📥 **CSV Export** - Export classified emails to CSV

## Setup Instructions

### 1. Register an Azure AD Application

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** > **App registrations**
3. Click **New registration**
4. Configure your app:
   - **Name**: `Outlook PO Reader` (or any name you prefer)
   - **Supported account types**: Choose based on your needs:
     - "Personal Microsoft accounts only" - for personal Outlook.com accounts
     - "Accounts in any organizational directory and personal Microsoft accounts" - for both work and personal
   - **Redirect URI**: 
     - Platform: **Web**
     - URI: `http://localhost:8501`
5. Click **Register**

### 2. Configure API Permissions

1. In your app registration, go to **API permissions**
2. Click **Add a permission** > **Microsoft Graph** > **Delegated permissions**
3. Add these permissions:
   - `Mail.Read` - Read user mail
   - `User.Read` - Sign in and read user profile
4. Click **Add permissions**
5. (Optional) If you have admin access, click **Grant admin consent**

### 3. Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. Add a description and choose expiry
4. Click **Add**
5. **IMPORTANT**: Copy the secret **Value** immediately (it won't be shown again!)

### 4. Configure the Application

Option A: **Environment Variables** (Recommended for security)
```bash
set OUTLOOK_CLIENT_ID=your_client_id_here
set OUTLOOK_CLIENT_SECRET=your_client_secret_here
set OUTLOOK_REDIRECT_URI=http://localhost:8501
```

Option B: **Edit config.py**
Update the values in `config.py`:
```python
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.1 OCR Requirement (for attachment extraction)

This feature uses Tesseract OCR. The app auto-detects `../tessaret/tesseract.exe` in this workspace.
If you use another location, install Tesseract and make sure it is available on PATH.

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Login**: Click "Get Login URL" in the sidebar, open the URL in your browser
2. **Authorize**: Login with your Microsoft account and grant permissions
3. **Get Code**: After authorization, you'll be redirected to localhost with a `code` parameter in the URL
4. **Enter Code**: Copy the code value and paste it into the app
5. **Fetch Emails**: Select a folder and click "Fetch Emails"
6. **Optional**: Enable **Extract PO data from PDF/image attachments** in the sidebar
7. **Classify**: Click "Classify All Emails" to identify PO emails and parse supported PDF/image attachments
8. **Export**: View results and export to CSV

## Troubleshooting

### "Token exchange error"
- Make sure your Client ID and Client Secret are correct
- Verify the redirect URI matches exactly (http://localhost:8501)
- Check that you copied the complete authorization code

### "No emails found"
- Ensure you granted Mail.Read permission
- Try fetching from a different folder (inbox, sentitems)

### "Using the Trainer with PyTorch requires accelerate>=1.1.0"
- Install/upgrade dependencies: `pip install -r requirements.txt`
- Or install directly: `pip install --upgrade "accelerate>=1.1.0"`

### "AADSTS error codes"
- AADSTS50011: Redirect URI mismatch - update in Azure AD
- AADSTS7000215: Invalid client secret - create a new one
- AADSTS65001: Consent required - grant admin consent or login as user to consent

## File Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── auth.py             # Microsoft OAuth2 authentication
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Security Notes

- Never commit your Client Secret to version control
- Use environment variables for credentials in production
- The access token expires after 1 hour
- Refresh tokens allow getting new access tokens without re-login

## License

MIT License
