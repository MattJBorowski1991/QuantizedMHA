#!/usr/bin/env python3
"""
Compare two NCU (NVIDIA Nsight Compute) output files side-by-side.
Merges matching tables into single comparison tables with proper columns.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def parse_ncu_file(filepath: str) -> List[Tuple[str, List[str], str]]:
    """
    Parse NCU output file into list of (table_name, table_lines, comments).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    tables = []
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # Look for separator line (all dashes/equals, at least 10 chars)
        if re.match(r'^[\s=\-]{10,}$', line):
            # Previous non-empty line is the table name
            name_idx = i - 1
            while name_idx >= 0 and not lines[name_idx].strip():
                name_idx -= 1
            
            if name_idx >= 0:
                table_name = lines[name_idx].strip()
                i += 1
                
                # Collect table rows (non-empty lines until next separator)
                table_lines = []
                comments = []
                
                while i < len(lines):
                    curr_line = lines[i].rstrip('\n')
                    
                    # Check if this is a separator (next table)
                    if re.match(r'^[\s=\-]{10,}$', curr_line):
                        break
                    
                    # Collect non-empty lines
                    stripped = curr_line.strip()
                    if stripped:
                        # Data lines have 2+ spaces separating columns (NOT stripped)
                        # Use original line to check
                        if '  ' in curr_line:
                            table_lines.append(curr_line)
                        else:
                            # Single column text = comment
                            comments.append(stripped)
                    
                    i += 1
                
                comment_text = ' '.join(comments) if comments else ""
                
                # Only add if table has data
                if table_lines:
                    tables.append((table_name, table_lines, comment_text))
                
                continue
        
        i += 1
    
    return tables


def parse_table_rows(lines: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parse table rows into (metric, unit, value) tuples.
    NCU format: metric name, unit, value separated by 2+ spaces.
    """
    rows = []
    
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        
        # Split by 2+ spaces (NCU column separator)
        parts = re.split(r'  +', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 3:
            metric = parts[0]
            unit = parts[1]
            value = parts[2]
            rows.append((metric, unit, value))
        elif len(parts) == 2:
            metric = parts[0]
            # Check if second part looks like a unit or value
            # Units typically have no spaces, values might have parentheses
            unit = parts[1]
            value = ""
            rows.append((metric, unit, value))
    
    return rows


def compare_tables(table_name: str, lines1: List[str], comments1: str, 
                   lines2: List[str], comments2: str, file1_name: str, file2_name: str) -> str:
    """
    Create a markdown comparison of two tables merged into one.
    """
    # Parse both tables
    rows1 = parse_table_rows(lines1)
    rows2 = parse_table_rows(lines2)
    
    # Build lookup for file2
    rows2_dict = {(metric, unit): value for metric, unit, value in rows2}
    
    md = f"### {table_name}\n\n"
    md += "| Metric | Unit | " + file1_name + " | " + file2_name + " |\n"
    md += "|---|---|---|---|\n"
    
    # Merge rows from file1 with file2
    for metric, unit, value1 in rows1:
        value2 = rows2_dict.get((metric, unit), "—")
        
        # Escape pipe characters
        metric_esc = metric.replace('|', '\\|')
        unit_esc = unit.replace('|', '\\|')
        value1_esc = value1.replace('|', '\\|')
        value2_esc = value2.replace('|', '\\|')
        
        md += f"| {metric_esc} | {unit_esc} | {value1_esc} | {value2_esc} |\n"
    
    # Add any rows only in file2
    for metric, unit, value2 in rows2:
        if (metric, unit) not in {(m, u) for m, u, _ in rows1}:
            metric_esc = metric.replace('|', '\\|')
            unit_esc = unit.replace('|', '\\|')
            value2_esc = value2.replace('|', '\\|')
            md += f"| {metric_esc} | {unit_esc} | — | {value2_esc} |\n"
    
    md += "\n"
    
    # Add comments if they exist
    if comments1 or comments2:
        md += "**Notes:**\n\n"
        if comments1:
            md += f"- {file1_name}: {comments1}\n"
        if comments2:
            md += f"- {file2_name}: {comments2}\n"
        md += "\n"
    
    md += "\n"
    return md


def create_standalone_table(table_name: str, lines: List[str], comments: str, file_name: str) -> str:
    """
    Create markdown for a table that appears in only one file.
    """
    rows = parse_table_rows(lines)
    
    md = f"### {table_name} ({file_name} only)\n\n"
    md += "| Metric | Unit | Value |\n"
    md += "|---|---|---|\n"
    
    for metric, unit, value in rows:
        metric_esc = metric.replace('|', '\\|')
        unit_esc = unit.replace('|', '\\|')
        value_esc = value.replace('|', '\\|')
        md += f"| {metric_esc} | {unit_esc} | {value_esc} |\n"
    
    md += "\n"
    
    if comments:
        md += f"**Notes:** {comments}\n\n"
    
    md += "\n"
    return md


def create_comparison(file1: str, file2: str, output_file: str) -> None:
    """
    Create markdown comparison document with paired and unpaired tables.
    """
    print(f"Parsing {file1}...")
    tables1 = parse_ncu_file(file1)
    
    print(f"Parsing {file2}...")
    tables2 = parse_ncu_file(file2)
    
    # Build table lookups
    tables1_dict = {name: (lines, comments) for name, lines, comments in tables1}
    tables2_dict = {name: (lines, comments) for name, lines, comments in tables2}
    
    # Get filenames for headers
    file1_name = Path(file1).stem
    file2_name = Path(file2).stem
    
    # Create markdown
    md_content = "# NCU Performance Comparison\n\n"
    md_content += f"**File 1:** `{Path(file1).name}`\n\n"
    md_content += f"**File 2:** `{Path(file2).name}`\n\n"
    md_content += "---\n\n"
    
    # Collect all unique table names
    all_table_names = set(tables1_dict.keys()) | set(tables2_dict.keys())
    
    for table_name in sorted(all_table_names):
        if table_name in tables1_dict and table_name in tables2_dict:
            # Both files have this table - create comparison
            lines1, comments1 = tables1_dict[table_name]
            lines2, comments2 = tables2_dict[table_name]
            md_content += compare_tables(
                table_name,
                lines1,
                comments1,
                lines2,
                comments2,
                file1_name,
                file2_name
            )
        elif table_name in tables1_dict:
            # Only in file1
            lines1, comments1 = tables1_dict[table_name]
            md_content += create_standalone_table(table_name, lines1, comments1, file1_name)
        else:
            # Only in file2
            lines2, comments2 = tables2_dict[table_name]
            md_content += create_standalone_table(table_name, lines2, comments2, file2_name)
    
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
