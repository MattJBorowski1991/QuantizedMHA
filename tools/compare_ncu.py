#!/usr/bin/env python3
"""
Compare two NCU (NVIDIA Nsight Compute) output files side-by-side.
Creates a markdown report with tables compared in parallel columns.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def parse_ncu_file(filepath: str) -> List[Tuple[str, List[str], str]]:
    """
    Parse NCU output file into list of (table_name, table_lines, comments).
    Tables are identified by lines containing only dashes/equals.
    Comments (OPT, INF, etc.) are captured after each table.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    tables = []
    current_table = []
    current_name = ""
    current_comments = ""
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a separator line (table header underline)
        if re.match(r'^[\s=\-]+$', line) and len(line.strip()) > 3:
            # Table ended, save it and capture comments
            if current_table and current_name:
                # Collect comment lines after table
                comments = []
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    # Stop if we hit a new table header
                    if j + 1 < len(lines) and re.match(r'^[\s=\-]+$', lines[j + 1]):
                        break
                    # Check if this looks like a comment (not a table row)
                    if not re.match(r'^[\s\d\.\-\+\%\*]+$', lines[j]):
                        comments.append(lines[j].strip())
                    j += 1
                
                current_comments = '\n'.join(comments)
                tables.append((current_name, current_table, current_comments))
                current_table = []
                current_name = ""
                current_comments = ""
        
        # Check if next line is a separator (current line is table header)
        elif i + 1 < len(lines) and re.match(r'^[\s=\-]+$', lines[i + 1]):
            current_name = line.strip()
            i += 1  # Skip separator
            # Start collecting table rows
            i += 1
            while i < len(lines):
                if re.match(r'^[\s=\-]+$', lines[i]):
                    break
                if lines[i].strip():
                    current_table.append(lines[i])
                i += 1
            i -= 1  # Back up one since loop will increment
        
        i += 1
    
    # Save last table if exists
    if current_table and current_name:
        tables.append((current_name, current_table, current_comments))
    
    return tables


def compare_tables(name1: str, lines1: List[str], comments1: str, name2: str, lines2: List[str], comments2: str) -> str:
    """
    Create a markdown comparison of two tables with their comments.
    """
    # Determine max lines needed
    max_lines = max(len(lines1), len(lines2))
    
    md = f"### {name1}\n\n"
    md += "| | | |\n"
    md += "|---|---|---|\n"
    
    for i in range(max_lines):
        col1 = lines1[i] if i < len(lines1) else ""
        col2 = lines2[i] if i < len(lines2) else ""
        
        # Escape pipe characters in content
        col1 = col1.replace('|', '\\|')
        col2 = col2.replace('|', '\\|')
        
        # Align columns
        md += f"| {i+1} | {col1} | {col2} |\n"
    
    md += "\n"
    
    # Add comments if they exist
    if comments1 or comments2:
        md += "#### Notes\n\n"
        if comments1:
            md += f"**{name1}:**\n"
            md += f"> {comments1}\n\n"
        if comments2:
            md += f"**{name2}:**\n"
            md += f"> {comments2}\n\n"
    
    md += "\n"
    return md


def create_comparison(file1: str, file2: str, output_file: str) -> None:
    """
    Create markdown comparison document.
    """
    print(f"Parsing {file1}...")
    tables1 = parse_ncu_file(file1)
    
    print(f"Parsing {file2}...")
    tables2 = parse_ncu_file(file2)
    
    # Build table lookup for file2
    tables2_dict = {name: (lines, comments) for name, lines, comments in tables2}
    
    # Create markdown
    md_content = "# NCU Performance Comparison\n\n"
    md_content += f"**File 1:** `{Path(file1).name}`\n\n"
    md_content += f"**File 2:** `{Path(file2).name}`\n\n"
    md_content += "---\n\n"
    
    for name1, lines1, comments1 in tables1:
        if name1 in tables2_dict:
            lines2, comments2 = tables2_dict[name1]
            md_content += compare_tables(
                f"{Path(file1).stem} - {name1}",
                lines1,
                comments1,
                f"{Path(file2).stem} - {name1}",
                lines2,
                comments2
            )
        else:
            print(f"Warning: Table '{name1}' not found in {file2}")
    
    # Add tables only in file2
    for name2, lines2, comments2 in tables2:
        if not any(name == name2 for name, _, _ in tables1):
            print(f"Warning: Table '{name2}' only in {file2}")
    
    # Write output
    print(f"Writing comparison to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"✓ Comparison saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two NCU output files side-by-side"
    )
    parser.add_argument("file1", help="First NCU output file")
    parser.add_argument("file2", help="Second NCU output file")
    parser.add_argument(
        "-o", "--output",
        help="Output markdown file (default: ncu_comparison.md in same dir as file1)"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.file1).exists():
        print(f"Error: {args.file1} not found")
        return 1
    if not Path(args.file2).exists():
        print(f"Error: {args.file2} not found")
        return 1
    
    # Determine output file
    output = args.output
    if not output:
        output_dir = Path(args.file1).parent
        output = str(output_dir / "ncu_comparison.md")
    
    create_comparison(args.file1, args.file2, output)
    return 0


if __name__ == "__main__":
    exit(main())
