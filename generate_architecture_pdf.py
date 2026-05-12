#!/usr/bin/env python3
"""
Generate PDF from architecture markdown document.
Uses available libraries to convert markdown to PDF.
"""

import sys
import os
from pathlib import Path

def try_weasyprint():
    """Try using weasyprint to convert markdown to PDF."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        print("Using weasyprint for PDF generation...")
        
        # Read markdown
        md_path = Path(__file__).parent / 'ARCHITECTURE.md'
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['codehilite', 'fenced_code'])
        
        # Add CSS styling
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 1in;
                }}
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #95a5a6;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #555;
                    margin-top: 20px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    overflow-x: auto;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #ecf0f1;
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        output_path = Path(__file__).parent / 'ARCHITECTURE.pdf'
        HTML(string=html_doc).write_pdf(str(output_path))
        
        print(f"✓ PDF generated successfully: {output_path}")
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with weasyprint: {e}")
        return False

def try_pypandoc():
    """Try using pypandoc to convert markdown to PDF."""
    try:
        import pypandoc
        
        print("Using pypandoc for PDF generation...")
        
        md_path = Path(__file__).parent / 'ARCHITECTURE.md'
        output_path = Path(__file__).parent / 'ARCHITECTURE.pdf'
        
        pypandoc.convert_file(
            str(md_path),
            'pdf',
            outputfile=str(output_path),
            extra_args=['--pdf-engine=pdflatex']
        )
        
        print(f"✓ PDF generated successfully: {output_path}")
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with pypandoc: {e}")
        return False

def try_matplotlib():
    """Fallback: Create a simple PDF using matplotlib."""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        print("Using matplotlib for PDF generation (simple format)...")
        
        md_path = Path(__file__).parent / 'ARCHITECTURE.md'
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        output_path = Path(__file__).parent / 'ARCHITECTURE.pdf'
        
        with PdfPages(str(output_path)) as pdf:
            # Split content into pages
            page_lines = []
            lines_per_page = 50
            
            for i, line in enumerate(lines):
                page_lines.append(line.rstrip())
                
                if len(page_lines) >= lines_per_page or i == len(lines) - 1:
                    # Create a page
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    
                    # Add text
                    text = '\n'.join(page_lines)
                    ax.text(0.1, 0.95, text, transform=ax.transAxes,
                           fontsize=8, verticalalignment='top',
                           family='monospace', wrap=True)
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    page_lines = []
        
        print(f"✓ PDF generated successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error with matplotlib: {e}")
        return False

def main():
    """Try different methods to generate PDF."""
    print("Generating architecture PDF...")
    print("=" * 60)
    
    # Try methods in order of preference
    methods = [
        ("weasyprint", try_weasyprint),
        ("pypandoc", try_pypandoc),
        ("matplotlib", try_matplotlib)
    ]
    
    for name, method in methods:
        print(f"\nTrying {name}...")
        if method():
            return
    
    print("\n" + "=" * 60)
    print("ERROR: Could not generate PDF.")
    print("\nPlease install one of the following:")
    print("  1. weasyprint: pip install weasyprint markdown")
    print("  2. pypandoc: pip install pypandoc (requires pandoc and pdflatex)")
    print("  3. matplotlib: pip install matplotlib")
    print("\nAlternatively, you can:")
    print("  - Use an online markdown to PDF converter")
    print("  - Use pandoc directly: pandoc ARCHITECTURE.md -o ARCHITECTURE.pdf")
    print("  - Open ARCHITECTURE.md in a markdown viewer and print to PDF")
    sys.exit(1)

if __name__ == '__main__':
    main()
