from flask import Flask, request, jsonify
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter
import pandas as pd
import numpy as np
import re
import os
import time
import mimetypes
import tiktoken
import warnings
import docling.models.easyocr_model
import subprocess
import tempfile

# Disable symlinks for HuggingFace on Windows (avoids permission errors)
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# OCR is enabled for text extraction from image-based PDFs
# def skip_ocr(self, *args, **kwargs):
#     return []
# docling.models.easyocr_model.EasyOcrModel.__call__ = skip_ocr

# Suppress only the specific WMF loader warning from Pillow
warnings.filterwarnings("ignore", message=".*cannot find loader for this WMF file.*")

# Suppress PyTorch pin_memory warnings (GPU not available on this system)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

app = Flask(__name__)

DELIMITER = "#--------------------------------------#"
pattern = re.compile(r"^\s*\d{6}[A-Za-z]+")

# -------------------------------
# Legacy Office Conversion Helpers
# -------------------------------

def convert_office_to_modern(filepath):
    """Convert legacy Office files to modern formats using LibreOffice."""
    ext_map = {"ppt": "pptx", "doc": "docx", "xls": "xlsx"}
    ext = os.path.splitext(filepath)[1][1:].lower()
    if ext not in ext_map:
        return filepath  # not legacy, no conversion
    
    outdir = os.path.dirname(filepath)
    target_ext = ext_map[ext]
    print(f"[INFO] Converting {filepath} -> .{target_ext} via LibreOffice...", flush=True)

    # Try multiple common LibreOffice installation paths
    libreoffice_paths = [
        "libreoffice",  # If in PATH
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS standard location
        "/usr/local/bin/libreoffice",  # Homebrew formula install
        "/opt/homebrew/bin/libreoffice",  # Homebrew on Apple Silicon
        "/usr/bin/libreoffice"  # Linux standard location
    ]
    
    libreoffice_cmd = None
    for path in libreoffice_paths:
        try:
            # Test if this path works by checking version
            test_result = subprocess.run(
                [path, "--version"], 
                capture_output=True, 
                timeout=10
            )
            if test_result.returncode == 0:
                libreoffice_cmd = path
                break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    if libreoffice_cmd is None:
        raise RuntimeError(
            "LibreOffice not found. Please install LibreOffice:\n"
            "  macOS: brew install --cask libreoffice\n"
            "  Linux: sudo apt install libreoffice (Ubuntu/Debian) or sudo yum install libreoffice (CentOS/RHEL)"
        )

    result = subprocess.run(
        [libreoffice_cmd, "--headless", "--convert-to", target_ext, filepath, "--outdir", outdir],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed: {result.stderr.decode()}")

    newfile = os.path.join(
        outdir, os.path.splitext(os.path.basename(filepath))[0] + f".{target_ext}"
    )
    if not os.path.exists(newfile):
        raise RuntimeError(f"Conversion failed: {newfile} not found")
    
    print(f"[INFO] Conversion successful: {newfile}", flush=True)
    return newfile

def preprocess_office_file(filepath: str) -> str:
    """
    Run conversion step if the file is a legacy Office format.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".doc", ".ppt", ".xls"]:
        return convert_office_to_modern(filepath)
    return filepath

# -------------------------------
# Excel Chunker
# -------------------------------

def row_to_cleaned_line(values):
    cleaned = [str(v).replace(",", "") if not pd.isna(v) else "nan" for v in values]
    return ",".join(cleaned)

def excel_chunker(file_path):
    print(f"[EXCEL] Starting Excel chunking for: {file_path}", flush=True)
    final_chunks = []
    
    # Step 1: Load Excel file structure
    print(f"[EXCEL] Loading Excel file structure...", flush=True)
    sheet_names = pd.ExcelFile(file_path).sheet_names
    print(f"[EXCEL] Found {len(sheet_names)} sheets: {sheet_names}", flush=True)
    
    CHUNK_LINE_LIMIT = 3
    CONTEXT_LINES = 5
    pattern = re.compile(r"^\s*\d{6}[A-Za-z]+")
    
    # Step 2: Process each sheet
    for sheet_idx, sheet_name in enumerate(sheet_names, 1):
        print(f"[EXCEL] Processing sheet {sheet_idx}/{len(sheet_names)}: {sheet_name}", flush=True)
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        raw_lines = []

        # Add header row
        header_row = row_to_cleaned_line(df.columns.tolist())
        raw_lines.append(header_row)

        found_first_pattern = False

        # Process each row
        for _, row in df.iterrows():
            row_values = row.tolist()
            values = []

            for i, val in enumerate(row_values):
                remaining_vals = row_values[i+1:]
                if pd.isna(val) or str(val).lower() == "nan":
                    if all(pd.isna(v) or str(v).lower() == "nan" for v in remaining_vals):
                        continue
                    else:
                        values.append("nan")
                else:
                    if isinstance(val, (int, float, np.number)):
                        val = round(val, 2)
                    values.append(str(val).replace(",", "").strip())

            if not values:
                raw_lines.append("")
                continue

            first_col_val = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
            if pattern.match(first_col_val):
                if found_first_pattern:
                    raw_lines.append(DELIMITER)
                else:
                    found_first_pattern = True

            raw_lines.append(row_to_cleaned_line(values))

        # Chunk the lines by delimiters
        cleaned_lines = []
        prev_was_delim = False
        for line in raw_lines:
            if line.strip() == "" or line.strip() == DELIMITER:
                if not prev_was_delim:
                    cleaned_lines.append(DELIMITER)
                    prev_was_delim = True
            else:
                cleaned_lines.append(line)
                prev_was_delim = False

        current_chunk = []
        for line in cleaned_lines + [DELIMITER]:
            if line.strip() == DELIMITER:
                if current_chunk:
                    context = current_chunk[:CONTEXT_LINES]
                    remaining = current_chunk
                    is_first_chunk = True
                    while remaining:
                        chunk_body = remaining[:CHUNK_LINE_LIMIT]
                        remaining = remaining[CHUNK_LINE_LIMIT:]
                        if is_first_chunk:
                            chunk = chunk_body
                            is_first_chunk = False
                        else:
                            chunk = context + chunk_body
                        chunk_text = "\n".join(chunk)

                        match_val = None
                        for row in chunk:
                            for cell in row.split(","):
                                if pattern.match(cell.strip()):
                                    match_val = cell.strip()
                                    break
                            if match_val:
                                break

                        if match_val:
                            prepend_line = ("{} ".format(match_val) * 5).strip()
                            chunk_text = prepend_line + "\n" + chunk_text

                        # Sheet name and document name prepend
                        filename = os.path.basename(file_path)
                        chunk_text = f"The following data is from the {filename} file, {sheet_name} Sheet.\n" + chunk_text

                        final_chunks.append(chunk_text)

                current_chunk = []
            else:
                current_chunk.append(line)

    print(f"[EXCEL] Excel chunking completed: {len(final_chunks)} total chunks", flush=True)
    return final_chunks if final_chunks else ["Empty Excel file"]

# -------------------------------
# Docling Chunker
# -------------------------------

def docling_chunker(file_path, model_name="text-embedding-3-large", max_tokens=7000):
    print(f"[DOCLING] Starting chunking for: {file_path}", flush=True)
    
    # Check file size and warn about large files
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"[DOCLING] File size: {file_size_mb:.1f} MB", flush=True)
    
    if file_size_mb > 50:
        print(f"[DOCLING] WARNING: Large file ({file_size_mb:.1f} MB) - may take several minutes", flush=True)
    
    print(f"[DOCLING] Processing parameters: model={model_name}, max_tokens={max_tokens}", flush=True)
    
    try:
        # Step 1: Document conversion with basic timeout
        print(f"[DOCLING] Converting document...", flush=True)
        
        # Try with OCR first, then fallback to no-OCR if it fails
        doc = None
        try:
            doc = DocumentConverter().convert(source=file_path).document
            print(f"[DOCLING] Document converted successfully with OCR", flush=True)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['opencv', 'resize', 'ocr', 'easyocr', 'image']):
                print(f"[DOCLING] OCR failed, retrying without OCR: {str(e)[:100]}", flush=True)
                try:
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    from docling.document_converter import PdfFormatOption
                    
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = False
                    
                    converter = DocumentConverter(
                        format_options={
                            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                        }
                    )
                    doc = converter.convert(source=file_path).document
                    print(f"[DOCLING] Document converted successfully without OCR", flush=True)
                except Exception as e2:
                    raise Exception(f"Both OCR and non-OCR conversion failed: {str(e2)}")
            else:
                raise e
        
        if not doc:
            raise Exception("Document conversion failed - no document returned")
            
        print(f"[DOCLING] Document has {len(doc.texts)} text elements", flush=True)
        
        # Step 2: Chunker initialization
        print(f"[DOCLING] Initializing chunker...", flush=True)
        chunker_obj = HybridChunker()
        chunker_obj.tokenizer.max_tokens = max_tokens
        print(f"[DOCLING] Chunker initialized with max_tokens={max_tokens}", flush=True)

        # Step 3: Tokenizer setup
        print(f"[DOCLING] Setting up tokenizer for {model_name}...", flush=True)
        enc = tiktoken.encoding_for_model(model_name)
        chunk_iter = chunker_obj.chunk(dl_doc=doc)
        print(f"[DOCLING] Tokenizer configured successfully", flush=True)
        
        final_chunks = []
        chunk_count = 0
        
        # Convert iterator to list to get total count
        print(f"[DOCLING] Preparing chunks for processing...", flush=True)
        chunk_list = list(chunk_iter)
        print(f"[DOCLING] Found {len(chunk_list)} initial chunks to process", flush=True)
        
        # Step 4: Process chunks
        print(f"[DOCLING] Processing {len(chunk_list)} chunks...", flush=True)
        for chunk_idx, chunk in enumerate(chunk_list, 1):
            chunk_count += 1
            enriched = chunker_obj.contextualize(chunk=chunk)
            tokens = enc.encode(enriched)
            
            if len(tokens) <= max_tokens:
                final_chunks.append(enriched)
            else:
                # Handle large chunks
                num_splits = (len(tokens) + max_tokens - 1) // max_tokens
                print(f"[DOCLING] Splitting large chunk {chunk_idx} ({len(tokens)} tokens) into {num_splits} parts", flush=True)
                for i in range(0, len(tokens), max_tokens):
                    sub_tokens = tokens[i:i + max_tokens]
                    sub_chunk = enc.decode(sub_tokens)
                    final_chunks.append(sub_chunk)
        
        print(f"[DOCLING] Chunking completed: {len(final_chunks)} final chunks from {len(chunk_list)} initial chunks", flush=True)
        return final_chunks
        
    except Exception as e:
        print(f"[DOCLING] Error during chunking: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return []

def postprocess_chunk(chunk, filename, delimiter=DELIMITER):
    """Add context headers to each chunk for better RAG performance"""
    lines = chunk.strip().split("\n")
    
    # Remove any leading delimiter lines
    while lines and lines[0].strip() == delimiter:
        lines.pop(0)
    
    if not lines:
        return chunk
    
    # Use first line as context
    context_line = lines[0].strip()
    lines = lines[1:]
    
    # Add only ONE con  text header instead of 5 repetitive ones
    header_lines = [f"Document: {filename} - Section: {context_line}"]

    # Build new chunk with context headers
    new_chunk = f"{delimiter}\n" + "\n".join(header_lines) + "\n" + "\n".join(lines) + f"\n{delimiter}"
    
    return new_chunk

# -------------------------------
# Flask Route
# -------------------------------

@app.route("/process", methods=["POST", "PUT"])
def process_file():
    start_time = time.time()
    print(f"\n{'='*50}", flush=True)
    print(f"🚀 PROCESS FILE ROUTE HIT - {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*50}", flush=True)
    
    try:
        # Step 1: Handle file upload
        print(f"[STEP 1] Handling file upload...", flush=True)
        # Handle multipart form data from Streamlit
        if 'file' in request.files:
            # Streamlit sends files as multipart form data
            uploaded_file = request.files['file']
            filename = uploaded_file.filename or 'uploaded_file'
            file_data = uploaded_file.read()
            print(f"[STEP 1] Received file via multipart: {filename} ({len(file_data)} bytes)", flush=True)
        else:
            # Fallback to the original method for other clients
            filename = request.headers.get("X-File-Name") or request.headers.get("file-name") or "uploaded_file"
            file_data = request.data
            print(f"[STEP 1] Received file via headers: {filename} ({len(file_data)} bytes)", flush=True)
            
        # Step 2: Process filename and extension
        print(f"[STEP 2] Processing filename and extension...", flush=True)
        if not os.path.splitext(filename)[1]:
            guessed_ext = mimetypes.guess_extension(request.content_type)
            if guessed_ext:
                filename += guessed_ext
                print(f"[STEP 2] Added guessed extension: {filename}", flush=True)

        # Step 3: Save file
        # Use system temp directory (cross-platform compatible)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        print(f"[STEP 3] Saving file to: {filepath}", flush=True)
        with open(filepath, "wb") as f:
            f.write(file_data)

        # Step 4: Preprocess Office files
        print(f"[STEP 4] Preprocessing Office files...", flush=True)
        # Convert legacy Office formats first
        filepath = preprocess_office_file(filepath)
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        print(f"[STEP 4] File ready for chunking: {ext} file ({filename})", flush=True)

        # Step 5: Select and execute chunking method
        print(f"[STEP 5] Selecting and executing chunking method...", flush=True)
        if ext in [".xlsx", ".xls"]:
            print(f"[STEP 5] Using Excel chunker for {ext} file", flush=True)
            chunks = excel_chunker(filepath)
            print(f"[STEP 5] Excel chunker returned {len(chunks)} chunks", flush=True)
            
            # Step 6: Post-process Excel chunks with context headers
            print(f"[STEP 6] Post-processing {len(chunks)} Excel chunks...", flush=True)
            processed_chunks = []
            for chunk_idx, chunk in enumerate(chunks, 1):
                processed_chunks.append(postprocess_chunk(chunk, filename))
            
            print(f"[STEP 6] After postprocessing: {len(processed_chunks)} Excel chunks", flush=True)
            
            response = [
                {"page_content": chunk, "metadata": {"filename": filename, "chunk_index": i}}
                for i, chunk in enumerate(processed_chunks)
            ]
        else:
            print(f"[STEP 5] Using docling chunker for {ext} file", flush=True)
            chunks = docling_chunker(filepath)
            print(f"[STEP 5] Docling chunker returned {len(chunks)} raw chunks", flush=True)
            
            if chunks:
                print(f"[STEP 5] Sample raw chunk length: {len(chunks[0])} characters", flush=True)
            
            # Step 6: Post-process chunks (only for non-Excel files)
            print(f"[STEP 6] Post-processing {len(chunks)} chunks...", flush=True)
            processed_chunks = []
            for chunk_idx, chunk in enumerate(chunks, 1):
                processed_chunks.append(postprocess_chunk(chunk, filename))
            
            print(f"[STEP 6] After postprocessing: {len(processed_chunks)} chunks", flush=True)
            
            if processed_chunks:
                print(f"[STEP 6] Sample processed chunk length: {len(processed_chunks[0])} characters", flush=True)
            
            response = [
                {"page_content": chunk, "metadata": {"filename": filename, "chunk_index": i}}
                for i, chunk in enumerate(processed_chunks)
            ]

        # Step 7: Save chunks locally
        print(f"[STEP 7] Saving chunks locally...", flush=True)
        output_path = os.path.join(os.path.dirname(__file__), "chunks.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in response:
                f.write(chunk["page_content"] + "\n" + DELIMITER + "\n")
        print(f"[STEP 7] Chunks saved locally: {output_path}", flush=True)
        
        # Step 8: Complete
        elapsed_time = time.time() - start_time
        print(f"[COMPLETE] 🎉 CHUNKING COMPLETED IN {elapsed_time:.2f} sec", flush=True)
        print(f"[COMPLETE] Total chunks generated: {len(response)}", flush=True)
        print(f"{'='*50}", flush=True)

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9876)