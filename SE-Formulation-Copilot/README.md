# SE Formulation CoPilot

A comprehensive repository for Systems Engineering (SE) Formulation CoPilot tools and utilities.

## 📋 Overview

This repository contains various tools and applications for SE formulation processes, including Streamlit applications, document processing utilities, and MCP (Model Context Protocol) integrations.

## 🚀 Quick Start

### Prerequisites
- Python 3.12.3
- Virtual environment (recommended)
- LibreOffice (for document conversion - see installation instructions below)

### LibreOffice Installation

The application requires LibreOffice for converting legacy Office documents (.ppt, .doc, .xls) to modern formats. Install LibreOffice using your platform's package manager:

#### macOS
```bash
brew install --cask libreoffice
```

#### Ubuntu/Debian Linux
```bash
sudo apt update
sudo apt install libreoffice
```

#### CentOS/RHEL/Fedora Linux
```bash
# CentOS/RHEL
sudo yum install libreoffice

# Fedora
sudo dnf install libreoffice
```

#### Windows
Download and install from [LibreOffice official website](https://www.libreoffice.org/download/download/)

**Note**: The application will automatically detect LibreOffice from common installation paths. No additional PATH configuration is required after installation.

### Installation & Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To run the full application stack, you'll need **three separate terminal windows**:

#### Terminal 1: Streamlit Frontend
```bash
streamlit run app.py --server.maxUploadSize=1024
```

#### Terminal 2: API Endpoint
```bash
python endpoint.py
```

## 💡 Features

- **Document Processing**: Upload and process various document formats (.pdf, .docx, .pptx, .txt, .md, and legacy formats like .ppt, .doc, .xls)
- **AI Integration**: Multiple AI service integrations (Azure OpenAI, etc.)
- **Streamlit Interface**: User-friendly web interface
- **Knowledge Base Management**: Create and manage custom knowledge bases with RAG retrieval
- **Legacy Document Support**: Automatic conversion of legacy Office documents using LibreOffice

## 🔧 Troubleshooting

### LibreOffice Issues
If you encounter errors like `FileNotFoundError: [Errno 2] No such file or directory: 'libreoffice'`:

1. **Verify LibreOffice installation**:
   ```bash
   # macOS - try both commands
   libreoffice --version
   /Applications/LibreOffice.app/Contents/MacOS/soffice --version
   
   # Linux
   libreoffice --version
   ```

2. **Reinstall LibreOffice** if the commands above don't work:
   ```bash
   # macOS
   brew install --cask libreoffice
   
   # Linux (Ubuntu/Debian)
   sudo apt install --reinstall libreoffice
   ```

3. **The application automatically detects LibreOffice** from these common paths:
   - `libreoffice` (if in PATH)
   - `/Applications/LibreOffice.app/Contents/MacOS/soffice` (macOS)
   - `/usr/local/bin/libreoffice` (Homebrew formula)
   - `/opt/homebrew/bin/libreoffice` (Homebrew on Apple Silicon)
   - `/usr/bin/libreoffice` (Linux standard)

### Document Upload Issues
- Ensure the API endpoint is running (`python endpoint.py`)
- Check file size limits (default max: 1024MB)
- Verify supported file formats: .pdf, .docx, .pptx, .txt, .md, .ppt, .doc, .xls
