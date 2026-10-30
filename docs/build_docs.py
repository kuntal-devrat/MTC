#!/usr/bin/env python3
"""
MTC Documentation & Research Paper Builder
Generates publication-quality HTML and PDF for:
1. Teacher Submission Project Report (AKTU / GL Bajaj B.Tech format)
2. IEEE Research Paper (Standard 2-Column Manuscript format)
"""

import os
import re
import subprocess
from markdown_it import MarkdownIt

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))

def clean_katex_in_html(html: str) -> str:
    """Ensure math formulas display cleanly in both browser and headless LibreOffice."""
    def replace_block_math(m):
        eq = m.group(1).strip()
        return f'<div class="display-math" data-math="{eq}">$${eq}$$</div>'
    html = re.sub(r'\$\$(.*?)\$\$', replace_block_math, html, flags=re.DOTALL)
    
    def replace_inline_math(m):
        eq = m.group(1).strip()
        return f'<span class="inline-math" data-math="{eq}">${eq}$</span>'
    html = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', replace_inline_math, html)
    
    return html

def build_teacher_report():
    print("=== Building Teacher Submission Project Report ===")
    md_path = os.path.join(DOCS_DIR, "MTC_Project_Report_Teacher_Submission.md")
    html_path = os.path.join(DOCS_DIR, "MTC_Project_Report_Teacher_Submission.html")
    pdf_path = os.path.join(DOCS_DIR, "MTC_Project_Report_Teacher_Submission.pdf")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Find where the Certificate of Approval begins
    cert_marker = "## CERTIFICATE OF APPROVAL"
    cert_idx = md_text.find(cert_marker)
    if cert_idx != -1:
        body_md = md_text[cert_idx:]
    else:
        body_md = md_text

    md_parser = MarkdownIt("gfm-like")
    body_html = md_parser.render(body_md)
    body_html = clean_katex_in_html(body_html)

    # Formal signature blocks
    cert_sig_table = """
    <table style="width: 100%; border: none; margin-top: 45pt; margin-bottom: 15pt;">
        <tr>
            <td style="width: 50%; border: none; text-align: left; vertical-align: top; background: transparent; padding: 0;">
                <div style="width: 80%; border-bottom: 1.5px solid #000; margin-bottom: 6pt;"></div>
                <b>Ms. Gaurisha Sisodia</b><br>
                Project Supervisor<br>
                Assistant Professor, Dept. of CSE<br>
                G. L. Bajaj Group of Institutions, Mathura
            </td>
            <td style="width: 50%; border: none; text-align: right; vertical-align: top; background: transparent; padding: 0;">
                <div style="width: 80%; border-bottom: 1.5px solid #000; margin-bottom: 6pt; margin-left: auto;"></div>
                <b>Head of Department</b><br>
                Department of Computer Science &amp; Engineering<br>
                G. L. Bajaj Group of Institutions, Mathura
            </td>
        </tr>
    </table>
    """
    
    decl_sig_table = """
    <table style="width: 100%; border-collapse: collapse; margin-top: 25pt; margin-bottom: 15pt;">
        <tr style="background-color: #f1f5f9;">
            <th style="width: 40%; border: 1px solid #444; padding: 5pt 7pt;">Student Name</th>
            <th style="width: 30%; border: 1px solid #444; padding: 5pt 7pt;">University Roll Number</th>
            <th style="width: 30%; border: 1px solid #444; padding: 5pt 7pt; text-align: center;">Candidate Signature</th>
        </tr>
        <tr>
            <td style="border: 1px solid #444; padding: 5pt 7pt;"><b>Ajay Sisodia</b></td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; font-family: monospace;">2405111530012</td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; text-align: center; color: #888;">____________________</td>
        </tr>
        <tr>
            <td style="border: 1px solid #444; padding: 5pt 7pt;"><b>Harsh Chaudhary</b></td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; font-family: monospace;">2405111530067</td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; text-align: center; color: #888;">____________________</td>
        </tr>
        <tr>
            <td style="border: 1px solid #444; padding: 5pt 7pt;"><b>Devrat Kuntal</b></td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; font-family: monospace;">2405111530055</td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; text-align: center; color: #888;">____________________</td>
        </tr>
        <tr>
            <td style="border: 1px solid #444; padding: 5pt 7pt;"><b>Aryan Upadhyay</b></td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; font-family: monospace;">2405111530037</td>
            <td style="border: 1px solid #444; padding: 5pt 7pt; text-align: center; color: #888;">____________________</td>
        </tr>
    </table>
    """

    # Replace certificate raw sig (both supervisor and HOD blocks)
    body_html = re.sub(
        r'<p><br />\s*\\</p>\s*<hr />\s*<p><strong>Ms\. Gaurisha Sisodia.*?Head of Department.*?G\. L\. Bajaj Group of Institutions, Mathura<\/p>',
        cert_sig_table,
        body_html,
        flags=re.DOTALL
    )

    # Replace declaration raw sig
    body_html = re.sub(
        r'<ol>\s*<li><strong>Ajay Sisodia</strong>.*?Aryan Upadhyay.*?<\/ol>',
        decl_sig_table,
        body_html,
        flags=re.DOTALL
    )

    # Put page-break-before: always directly on headers
    page_break_headers = [
        ("<h2>CERTIFICATE OF APPROVAL</h2>", '<h2 style="page-break-before: always; margin-top: 0;">CERTIFICATE OF APPROVAL</h2>'),
        ("<h2>CANDIDATE&#39;S DECLARATION</h2>", '<h2 style="page-break-before: always; margin-top: 0;">CANDIDATE\'S DECLARATION</h2>'),
        ("<h2>CANDIDATE'S DECLARATION</h2>", '<h2 style="page-break-before: always; margin-top: 0;">CANDIDATE\'S DECLARATION</h2>'),
        ("<h2>ACKNOWLEDGEMENTS</h2>", '<h2 style="page-break-before: always; margin-top: 0;">ACKNOWLEDGEMENTS</h2>'),
        ("<h2>ABSTRACT</h2>", '<h2 style="page-break-before: always; margin-top: 0;">ABSTRACT</h2>'),
        ("<h2>TABLE OF CONTENTS</h2>", '<h2 style="page-break-before: always; margin-top: 0;">TABLE OF CONTENTS</h2>'),
        ("<h2>LIST OF FIGURES</h2>", '<h2 style="page-break-before: always; margin-top: 0;">LIST OF FIGURES</h2>'),
        ("<h2>LIST OF TABLES</h2>", '<h2 style="page-break-before: always; margin-top: 0;">LIST OF TABLES</h2>'),
        ("<h1>CHAPTER 1:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 1:'),
        ("<h1>CHAPTER 2:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 2:'),
        ("<h1>CHAPTER 3:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 3:'),
        ("<h1>CHAPTER 4:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 4:'),
        ("<h1>CHAPTER 5:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 5:'),
        ("<h1>CHAPTER 6:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 6:'),
        ("<h1>CHAPTER 7:", '<h1 style="page-break-before: always; margin-top: 0;">CHAPTER 7:'),
        ("<h2>REFERENCES</h2>", '<h2 style="page-break-before: always; margin-top: 0;">REFERENCES</h2>')
    ]
    
    for old_h, new_h in page_break_headers:
        if old_h in body_html:
            body_html = body_html.replace(old_h, new_h, 1)

    # Build the official GL Bajaj Cover Page that fits exactly on Page 1
    cover_page_html = f"""
    <div class="border-box">
        <div class="inst-title">G. L. BAJAJ GROUP OF INSTITUTIONS, MATHURA</div>
        <div class="dept-title">DEPARTMENT OF COMPUTER SCIENCE &amp; ENGINEERING<br>
            <span style="font-size: 8.5pt; font-weight: normal; font-style: italic;">(Affiliated to Dr. A.P.J. Abdul Kalam Technical University, Lucknow)</span>
        </div>
        
        <div><img class="logo" src="{DOCS_DIR}/images/gl_bajaj_logo_tight.png" alt="GL Bajaj Logo"></div>
        
        <div class="report-label">A MINOR PROJECT REPORT ON</div>
        <div class="proj-title">MICRO-TRAIN-COMPILER (MTC)</div>
        <div class="proj-sub">An Ahead-of-Time Static Compiler for Zero-Heap On-Device Neural Network Training on 32-Bit Microcontrollers</div>
        
        <div class="degree">Submitted in partial fulfillment of the requirements for the award of the degree of<br>
            <b>Bachelor of Technology in Computer Science &amp; Engineering</b>
        </div>
        
        <div class="session">Academic Session: 2026–2027</div>
        
        <table class="sub-table">
            <tr>
                <td style="width: 55%; font-weight: bold; text-decoration: underline; padding-bottom: 2px;">Submitted By:</td>
                <td style="width: 45%; font-weight: bold; text-decoration: underline; padding-bottom: 2px;">Under the Supervision of:</td>
            </tr>
            <tr>
                <td>
                    <b>Ajay Sisodia</b> (Roll No: 2405111530012)<br>
                    <b>Harsh Chaudhary</b> (Roll No: 2405111530067)<br>
                    <b>Devrat Kuntal</b> (Roll No: 2405111530055)<br>
                    <b>Aryan Upadhyay</b> (Roll No: 2405111530037)
                </td>
                <td>
                    <b>Ms. Gaurisha Sisodia</b><br>
                    Assistant Professor<br>
                    Department of Computer Science &amp; Engineering<br>
                    GL Bajaj Group of Institutions, Mathura
                </td>
            </tr>
        </table>
    </div>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MTC - Minor Project Report (Teacher Submission)</title>
    
    <!-- KaTeX for formula rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>

    <style>
        @page {{
            size: A4 portrait;
            margin: 10mm 14mm 10mm 14mm;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Times New Roman', 'Liberation Serif', serif;
            font-size: 10.5pt;
            line-height: 1.48;
            color: #111111;
            background: #ffffff;
            text-align: justify;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}

        /* ===== COVER PAGE STYLES (Single Page) ===== */
        .border-box {{
            border: 2px solid #000000;
            padding: 4mm 6mm;
            text-align: center;
            box-sizing: border-box;
            page-break-after: always;
        }}
        .inst-title {{
            font-size: 14.5pt;
            font-weight: bold;
            margin-bottom: 1mm;
            color: #000000;
        }}
        .dept-title {{
            font-size: 10.5pt;
            font-weight: bold;
            margin-bottom: 1.5mm;
            color: #222222;
        }}
        .logo {{
            width: 32mm;
            height: 14mm;
            display: block;
            margin: 1mm auto;
            border: none !important;
        }}
        .report-label {{
            font-size: 10pt;
            font-weight: bold;
            letter-spacing: 0.5px;
            margin-bottom: 1mm;
            color: #000000;
        }}
        .proj-title {{
            font-size: 15.5pt;
            font-weight: bold;
            margin-bottom: 1mm;
            color: #000000;
        }}
        .proj-sub {{
            font-size: 9.5pt;
            font-style: italic;
            color: #333333;
            line-height: 1.25;
            max-width: 92%;
            margin: 0 auto 1.5mm auto;
        }}
        .degree {{
            font-size: 9pt;
            color: #222222;
            line-height: 1.25;
            margin-bottom: 1.5mm;
        }}
        .session {{
            font-size: 10pt;
            font-weight: bold;
            margin-bottom: 2mm;
            color: #000000;
        }}
        .sub-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 2mm;
            border-top: 1.5px solid #000000;
            padding-top: 2mm;
            text-align: left;
        }}
        .sub-table td {{
            border: none;
            padding: 1px 2px;
            font-size: 8.8pt;
            line-height: 1.25;
            vertical-align: top;
            background: transparent !important;
        }}

        /* ===== REPORT BODY STYLES ===== */
        .report-body {{
            padding: 0;
        }}

        h1 {{
            font-size: 14pt;
            font-weight: bold;
            color: #000000;
            margin-top: 16pt;
            margin-bottom: 9pt;
            text-transform: uppercase;
            border-bottom: 1.5px solid #000000;
            padding-bottom: 3pt;
        }}

        h2 {{
            font-size: 12.5pt;
            font-weight: bold;
            color: #000000;
            margin-top: 14pt;
            margin-bottom: 7pt;
            text-transform: uppercase;
        }}

        h3 {{
            font-size: 11pt;
            font-weight: bold;
            color: #111111;
            margin-top: 11pt;
            margin-bottom: 5pt;
        }}

        h4 {{
            font-size: 10pt;
            font-weight: bold;
            font-style: italic;
            color: #222222;
            margin-top: 9pt;
            margin-bottom: 3pt;
        }}

        p {{
            margin-bottom: 7pt;
            text-indent: 1.4em;
        }}

        p:has(img), p:has(table), p:has(pre) {{
            text-indent: 0;
        }}

        ul, ol {{
            margin-left: 1.8em;
            margin-bottom: 7pt;
        }}

        li {{
            margin-bottom: 2.5pt;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 9pt;
        }}

        th, td {{
            border: 1px solid #444444;
            padding: 4pt 5.5pt;
            text-align: left;
        }}

        th {{
            background-color: #f1f5f9;
            font-weight: bold;
            color: #000000;
        }}

        tr:nth-child(even) td {{
            background-color: #fafbfc;
        }}

        /* Code Blocks */
        pre {{
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 3px solid #0b69a3;
            padding: 6pt 8pt;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 8pt;
            line-height: 1.28;
            margin: 8pt 0;
            overflow-x: auto;
            white-space: pre-wrap;
        }}

        code {{
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 8.5pt;
            background-color: #f1f5f9;
            padding: 1pt 2.5pt;
            border-radius: 3px;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        /* Figures */
        img {{
            max-width: 85%;
            height: auto;
            display: block;
            margin: 10pt auto 4pt auto;
            border: 1px solid #cccccc;
        }}

        p > em {{
            display: block;
            text-align: center;
            font-size: 8.5pt;
            color: #444444;
            margin-top: 3pt;
            margin-bottom: 10pt;
            font-style: italic;
        }}

        blockquote {{
            border-left: 3px solid #64748b;
            padding-left: 9pt;
            color: #334155;
            margin: 7pt 0;
            font-style: italic;
        }}

        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 12pt 0;
        }}

        .display-math {{
            text-align: center;
            margin: 7pt 0;
            font-size: 10pt;
        }}

        .inline-math {{
            font-size: 9.5pt;
        }}
    </style>
</head>
<body>

{cover_page_html}

<div class="report-body">
{body_html}
</div>

</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML: {html_path} ({len(full_html)} bytes)")

    print("Converting to PDF via LibreOffice headless...")
    cmd = ["libreoffice", "--headless", "--convert-to", "pdf", html_path, "--outdir", DOCS_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully generated PDF: {pdf_path}")
    else:
        print(f"LibreOffice conversion failed: {res.stderr}")

def build_research_paper():
    print("\n=== Building IEEE Research Paper ===")
    md_path = os.path.join(DOCS_DIR, "MTC_Research_Paper.md")
    html_path = os.path.join(DOCS_DIR, "MTC_Research_Paper.html")
    pdf_path = os.path.join(DOCS_DIR, "MTC_Research_Paper.pdf")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Find the Abstract start to avoid duplicate title banner
    abstract_marker = "### Abstract"
    abs_idx = md_text.find(abstract_marker)
    if abs_idx != -1:
        body_md = md_text[abs_idx:]
    else:
        body_md = md_text

    md_parser = MarkdownIt("gfm-like")
    body_html = md_parser.render(body_md)
    body_html = clean_katex_in_html(body_html)

    # Convert ### Abstract to IEEE style abstract
    body_html = body_html.replace("<h3>Abstract</h3>", "<p><b><i>Abstract</i>—", 1)
    
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MTC: Ahead-of-Time Memory-Planned Automatic Differentiation for Zero-Heap On-Device Deep Learning on 32-Bit Microcontrollers</title>
    
    <!-- KaTeX for formula rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>

    <style>
        @page {{
            size: A4 portrait;
            margin: 14mm 12mm 14mm 12mm;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Times New Roman', 'Liberation Serif', serif;
            font-size: 9.8pt;
            line-height: 1.33;
            color: #000000;
            background: #ffffff;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}

        /* IEEE Title & Author Banner (Full Width) */
        .paper-header {{
            text-align: center;
            margin-bottom: 10pt;
            padding-bottom: 7pt;
            border-bottom: 1px solid #cccccc;
        }}

        .paper-title {{
            font-size: 17pt;
            font-weight: bold;
            line-height: 1.2;
            margin-bottom: 5pt;
            color: #000000;
        }}

        .paper-authors {{
            font-size: 10pt;
            font-weight: bold;
            margin-bottom: 3pt;
        }}

        .paper-affil {{
            font-size: 8.5pt;
            font-style: italic;
            color: #333333;
            line-height: 1.25;
        }}

        .equal-contrib {{
            font-size: 7.8pt;
            font-style: italic;
            color: #555555;
            margin-top: 2.5pt;
        }}

        /* IEEE Two-Column Body Layout */
        .paper-columns {{
            column-count: 2;
            column-gap: 5.5mm;
            text-align: justify;
        }}

        h1 {{
            font-size: 9.5pt;
            font-weight: bold;
            text-align: center;
            text-transform: uppercase;
            margin-top: 11pt;
            margin-bottom: 4pt;
            letter-spacing: 0.5px;
            break-after: avoid;
        }}

        h2 {{
            font-size: 9pt;
            font-weight: bold;
            font-style: italic;
            margin-top: 8pt;
            margin-bottom: 3pt;
            break-after: avoid;
        }}

        h3 {{
            font-size: 8.8pt;
            font-weight: normal;
            font-style: italic;
            margin-top: 6pt;
            margin-bottom: 2.5pt;
            break-after: avoid;
        }}

        p {{
            text-indent: 1.3em;
            margin-bottom: 3.5pt;
        }}

        p:has(table), p:has(img), p:has(pre) {{
            text-indent: 0;
        }}

        ul, ol {{
            margin-left: 1.4em;
            margin-bottom: 4pt;
        }}

        li {{
            margin-bottom: 1.5pt;
        }}

        /* Tables formatted to IEEE style: Top/bottom heavy rules, subtle middle rules */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 6pt 0;
            font-size: 7.2pt;
            line-height: 1.2;
            break-inside: avoid;
        }}

        th, td {{
            border-top: 0.5px solid #000000;
            border-bottom: 0.5px solid #000000;
            border-left: none;
            border-right: none;
            padding: 2pt 3pt;
            text-align: left;
        }}

        th {{
            font-weight: bold;
            text-transform: uppercase;
            font-size: 7pt;
            border-top: 1.2px solid #000000;
            border-bottom: 0.8px solid #000000;
            background-color: #fafafa;
        }}

        tr:last-child td {{
            border-bottom: 1.2px solid #000000;
        }}

        /* Figures */
        img {{
            width: 100%;
            height: auto;
            display: block;
            margin: 5pt auto 2pt auto;
            border: 0.5px solid #999999;
        }}

        p > em {{
            display: block;
            text-align: center;
            font-size: 7.5pt;
            color: #222222;
            margin-top: 2pt;
            margin-bottom: 5pt;
        }}

        /* Code & Pre */
        pre {{
            background-color: #f7f9fa;
            border: 0.5px solid #d0d7de;
            padding: 3pt 4pt;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 6.8pt;
            line-height: 1.2;
            margin: 4pt 0;
            overflow-x: auto;
            break-inside: avoid;
        }}

        code {{
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 7.2pt;
            background-color: #f3f4f6;
            padding: 0.5pt 1.5pt;
            border-radius: 2px;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        .display-math {{
            text-align: center;
            margin: 4pt 0;
            font-size: 8pt;
            break-inside: avoid;
        }}

        .inline-math {{
            font-size: 8pt;
        }}

        hr {{
            border: none;
            border-top: 0.5px solid #dddddd;
            margin: 5pt 0;
        }}
    </style>
</head>
<body>

<div class="paper-header">
    <div class="paper-title">MTC: Ahead-of-Time Memory-Planned Automatic Differentiation for Zero-Heap On-Device Deep Learning on 32-Bit Microcontrollers</div>
    <div class="paper-authors">Ajay Sisodia*, Harsh Chaudhary*, Devrat Kuntal*, Aryan Upadhyay*, and Gaurisha Sisodia</div>
    <div class="paper-affil">
        Department of Computer Science & Engineering, G. L. Bajaj Group of Institutions, Mathura, India<br>
        Affiliated with Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
    </div>
    <div class="equal-contrib">*Equal student authorship. Corresponding faculty supervisor: Gaurisha Sisodia (gaurisha.sisodia@glbajaj.org).</div>
</div>

<div class="paper-columns">
{body_html}
</div>

</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML: {html_path} ({len(full_html)} bytes)")

    print("Converting to PDF via LibreOffice headless...")
    cmd = ["libreoffice", "--headless", "--convert-to", "pdf", html_path, "--outdir", DOCS_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully generated PDF: {pdf_path}")
    else:
        print(f"LibreOffice conversion failed: {res.stderr}")

if __name__ == "__main__":
    build_teacher_report()
    build_research_paper()
