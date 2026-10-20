"""
Extract text from messy PDFs with tables,
headers/footers, and multi-column layouts.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import pdfplumber
import fitz #pymupdf
from PIL import Image
import pytesseract
import io

@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    rows: List[List[str]]
    headers: List[str]
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_markdown(self) -> str:
        """COnvert table to markdown format for chunking."""
        if not self.rows:
            return ""
        lines = []
        #header row
        lines.append("| "+" | ".join(self.headers)+" |")
        lines.append("| "+ " | ".join(["---"]* len(self.headers)) +" |")
        #Data rows
        for rows in self.rows:
            #Pad row if it has fewer columns than headers
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| "+ " | ".join(padded[:len(self.headers)]) + " |")
            return "\n".join(lines)

    def to_json(self) -> List[Dict[str,str]]:
        """Convert table to list of dicts for structured access"""
        return [
            {h: row[i] if i < len(row) else "" for i,h in enumerate(self.headers)}
            for row in self.rows
        ]

@dataclass
class PageContent:
    """Extracted content from a single PDF page"""
    page_number: int
    text: str
    tables: List[ExtractedTable] = field(default_factory = list)
    is_scanned: bool = False

def detect_headers_footers(pages_text: List[str], threshold: int = 3) -> Dict[str, List[str]]:
    """
    Detect repeated text across pages that likely represents headers/footers.
    Strategy: Text that appears identically on `threshold` or more pages,
    positioned at the top or bottom of the page, is likely a header/footer.
    Args:
        pages_text: List of full page text, one string per page
        threshold: Minimum number of pages a line must appear on
    Returns:
        Dict with 'headers' and 'footers' lists
    """

    # Extract first and last 3 lines from each page
    top_lines = Counter()
    bottom_lines = Counter()
    for page_text in pages_text:
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        if not lines:
            continue

        #Top 3 lines
        for line in lines[:3]:
            top_lines[line] +=1

        #Bottom 3 lines
        for line in lines[:3]:
            bottom_lines[line] +=1

    #Lines appearing on threshold pages are headers/footers
    headers = [line for line, count in top_lines.items() if count >= threshold]
    footers = [line for line, count in bottom_lines.items() if count >= threshold]

    #Also catch page number patterns
    page_num_pattern = re.compile(
        r'^(Page\s+\d+(\s+of\s+\d+)?|\d+\s*$|—\s*\d+\s*—|\[\d+\])',
        re.IGNORECASE
    )
    for line_counter in [top_lines, bottom_lines]:
        for line in line_counter:
            if page_num_pattern.match(line) and line not in headers+footers:
                footers.append(line)

    return {"headers":headers, "footers":footers}

def remove_headers_footers(text: str, headers_footers:Dict[str, List[str]]) -> str:
    """remove detected headers and footers from page text"""
    lines = text.split('\n')
    all_noise = set(headers_footers["headers"] + headers_footers["footers"])
    cleaned = [line for line in lines if line.strip() not in all_noise]
    #Also remove page number patterns inline
    page_num_patterns = [
        r'Page\s+\d+\s+of\s+\d+',
        r'^\s*\d+\s*$',
        r'—\s*\d+\s*—',
    ]
    result = '\n'.join(cleaned)
    for pattern in page_num_patterns:
        result = re.sub(pattern, '', result, flags=re.MULTILINE)
    return result.strip()

def extract_tables_pdfplumber(pdf_path: str) ->List[ExtractedTable]:
    """
    Extract tables from a PDF using pdfplumber.
    pdfplumber detects tables by finding intersecting lines that form
    a grid pattern. It handles:
    - Bordered tables
    - Tables with partial borders
    - Tables spanning most of the page width
    It struggles with:
    - Borderless tables (no lines at all)
    - Tables made with tabs/spaces only
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 2,
                }
            )
            for table_data in page_tables:
                if not table_data or len(table_data) < 2:
                    continue
                #First row is typically headers
                headers = [str(cell or "").strip() for cell in table_data[0]]
                rows = [
                    [str(cell or "").strip for cell in row ]
                    for row in table_data[1:]
                ]
                tables.append(ExtractedTable(
                    page_number = page_num,
                    rows = rows,
                    headers = headers,
                ))

        return tables

def extract_with_column_detection(pdf_path: str) -> List[PageContent]:
    """
    Extract text from PDF with multi-column layout detection.
    Uses pymupdf (fitz) to get text blocks with bounding boxes,
    then sorts by column position before reading order.
    """
    pages=[]
    with fitz.open(pdf_path) as doc:
        for page_num , page in enumerate(doc, 1):
            #Get text blocks with position info
            blocks = page.get_text("dict")["blocks"]
            #Separate text blocks from image blocks
            text_blocks = []
            has_images = False
            for block in blocks:
                if block["type"] ==0: #text block
                    text_blocks.append(block)
                elif block["type"] ==1: #image block
                    has_images = True
            
            if not text_blocks:
                pages.append(PageContent(
                        page_number = page_num,
                        text="",
                        is_scanned=has_images,
                ))
                continue

            #Detect columns by analysing x-coordinate clusters
            x_positions = [block["bbox"][0] for block in text_blocks]
            columns = detect_columns(x_positions , page.rect.widht)
            if len(columns) > 1:
                #Multi-column: sort blocks by column, then y-position
                sorted_blocks = sort_by_columns(text_blocks, columns)
            else:
                #Sinlge column: sort by y-position (top to bottom)
                sorted_blocks = sorted(text_blocks, key=lambda b:b["bbox"][1])                     

            #Extract text from sorted blocks
            page_text = ""
            for block in sorted_blocks:
                for line in block.get("lines",[]):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text +=span.get("text", "")
                    if line_text.strip():
                        page_text += line_text.strip() + "\n"
                page_text += "\n"
                        
                pages.append(PageContent(
                    page_number=page_num,
                    text=page_text.strip(),
                    is_scanned=has_images and not page_text.strip(),
                ))
            return pages

def detect_columns(x_postions: List[float], page_width: float) -> List[Tuple[float, float]]:
    """
    Detect column boundaries from text block x-coordinates.
    Returns list of (left_edge, right_edge) tuples for each column.
    """
    if not x_postions:
        return [(0, page_width)]
    #Cluster x-postions using a simple gap-based method
    sorted_x = sorted(set(x_postions))
    if len(sorted_x) < 2:
        return [(0, page_width)]
    #Find gaps larger than 15% of page width
    min_gap = page_width * 0.15
    columns = []
    col_start = sorted_x[0]
    for i in range(1, len(sorted_x)):
        gap = sorted_x[i] - sorted_x[i-1]
        if gap > min_gap:
            columns.append((col_start, sorted_x[i-1] + min_gap /2))
            col_start = sorted_x[i]

    columns.append((col_start, page_width))
    return columns

def sort_by_columns(blocks: List[Dict], columns: List[Tuple[float, float]]) -> List[Dict]:
    """
    Sort text blocks: left column top-to-bottom, then right column top-to-bottom
    """
    column_blocks = {i: [] for i in range(len(columns))}
    for block in blocks:
        block_x = block["bbox"][0]
        #Assign block to nearest column
        best_col =0
        best_dist = float("inf")
        for i, (left, right) in enumerate(columns):
            dist = abs(block_x - left)
            if dist < best_dist:
                best_dist = dist
                best_col = i
                column_blocks[best_col].append(block)
    #Sort each column by y-postion, then concatenate columsn
    sorted_blocks = []
    for col_idx in sorted(column_blocks.keys()):
        col = sorted(column_blocks[col_idx], key=lambda b: b["bbox"][1])
        sorted_blocks.extend(col)
    return sorted_blocks

def ocr_page(pdf_path: str, page_num: int, dpi: int=300) -> str:
    """
    OCR a single page of a scanned PDF.
    Uses pymupdf to render the page as an image, then pytesseract for OCR.
    Args:
        pdf_path: Path to the PDF file
        page_num: 0-indexed page number
        dpi: Resolution for rendering (300 recommended minimum)
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        #Render page as image
        mat = fitz.Matrix(dpi/ 72, dpi/72)
        pix = page.get_pixmap(matrix = mat)
        #Convert to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        #Preprocess for better OCR
        img = img.convert("L") #Grayscale
        #OCR with pytesseract
        text = pytesseract.image_to_string(
            img,
            config="--psm 6 --oem 3", #Assume uniform block of text, LSTM engine
        )

    return text

def extract_session_hierarchy(text: str) -> List[Dict]:
    """
    Detect section headers and build a hierarchy.
    Looks for patterns like:
    - "1. Section Title"
    - "1.1 Subsection Title"
    - "## Markdown Headers"
    - ALL CAPS LINES (common in legal/financial docs)
    """
    lines = text.split('\n')
    sections = []
    current_path = []
    #patterns for section headers
    numbered_pattern = re.compile(r'^(\d+(?:\.\d+)*)[.)\s]+(.+)$')
    caps_pattern = re.compile(r'^[A-Z][A-Z\s]{5,}$')
    markdown_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        #Check numbered sections (1., 1.1, 2.3.1)
        match = numbered_pattern.match(line)
        if match:
            number = match.group(1)
            title = match.group(2).strip()
            level = number.count('.') +1
            sections.append({
                "line_index": i,
                "level": level,
                "number":number,
                "title":title,
                "full_header": line,
            })
            continue
    
        #check all caps headers
        if caps_pattern.match(line) and len(line.split()) <=8:
            sections.append({
                "line_index": i,
                "level": 1,
                "number":"",
                "title":line.title,
                "full_header": line,
            })
            continue

        # Check markdown headers
        match = markdown_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
            "line_index": i,
                "level": level,
                "number":"",
                "title":title,
                "full_header": line, 
            })

    return sections

def chunk_with_structure(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Chunk text while preserving document structure.
    Each chunk includes:
    - The text content
    - The section hierarchy (breadcrumb) it belongs to
    - Page number if available
    """
    sections = extract_sction_hierarchy(text)
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    current_section_path = []
    current_size = 0
    for i, line in enumerate(lines):
        #Check if this line is a section header
        for section in sections:
            if section["line_index"] == i:
                #update sectiuon path
                level = section["level"]
                #Trim path to current level and add new section
                current_section_path = [
                    s for s in current_section_path if s["level"] < level
                ]
                current_section_path.append(section)
                break
        current_chunk_lines.append(line)
        current_size += len(line) +1 #+1 for new line
        if current_size >= max_chunk_size:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                breadcrumb = " > ".join(s["title"] for s in current_section_path)
                chunks.append({
                    "text": chunk_text,
                    "section_path": breadcrumb or "Unknown Section",
                    "metadata": {
                        "section_path" : [s["title"] for s in current_section_path],
                        "section_level": current_section_path[-1]["level"]
                        if current_section_path else 0,
                        "char_count": len (chunk_text),
                    },
                })
            #Keep overlap
            overlap_lines=[]
            overlap_size = 0
            for line in reversed(current_chunk_lines):
                if overlap_size+ len(line) > overlap:
                    break
                overlap_lines.insert(0,line)
                overlap_size += len(line) +1
                
            current_chunk_lines = overlap_lines
            current_size = overlap_lines
    #Dont forget the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            breadcrumb = ' > '.join(s["title"] for s in current_section_path)
            chunks.append({
                "text":chunk_text,
                "section": breadcrumb or "Unknown Section",
                "metadata": {
                    "section_path": [s["title"] for s in current_section_path],
                    "section_level": current_section_path[-1]["level"]
                    if current_section_path else 0,
                    "char_count": len(chunk_text),
                },
            })
    return chunks

class PDFExtractionPipeline:
    """
    Production PDF extraction pipeline.
    Handles:
        1. Header/footer detection and removal
        2. Table extraction and markdown conversion
        3. Multi-column layout detection
        4. OCR for scanned pages
        5. Section hierarchy preservation
        6. Structure-aware chunking
    """
    def __init__(self):
        pass

