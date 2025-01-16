
import json
import re

def scrape_ukhaan_to_json(file_path):
    """
    Scrapes Nepali ukhaan (proverbs) from README.md and outputs JSON format
    
    Args:
        file_path (str): Path to README.md file
        
    Returns:
        list: List of dictionaries containing ukhaan fields
    """

    
    # Initialize list to store results
    ukhaan_list = []
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find table content between headers and next section
    table_pattern = r'\| Nepali \| Roman \| Meaning \| Example.*?\n(.*?)(?=<span|\Z)'
    table_match = re.search(table_pattern, content, re.DOTALL)
    print(f"table_match: {table_match}")
    if table_match:
        # Get all content after the table header
        table_content = content.split('| Nepali | Roman | Meaning | Example')[1]
        print(f"table_content: {table_content}")
        
        # Split into rows and process each row
        rows = table_content.strip().split('\n')
        print(f"rows: {rows}")
        for idx, row in enumerate(rows):
            # Skip header separator row and empty rows
            if '---' in row or not row.strip() or '<span id=' in row:
                continue
                
            # Split row into columns and clean up
            cols = [col.strip() for col in row.split('|')]
            
            print(f"idx: {idx}, row: {row}, cols: {cols}" )
                
            # Create dictionary for this ukhaan
            ukhaan_dict = {
                'nepali': cols[0].strip(),
                'roman': cols[1].strip(),
                'meaning': cols[2].strip() if len(cols)>2 else None,
                'example': cols[3].strip() if len(cols) > 3 and cols[3].strip() else None
            }
            
            ukhaan_list.append(ukhaan_dict)
    
    return ukhaan_list

def main():
    json_file = scrape_ukhaan_to_json('nepali-ukhaan/README.md')
    print(json_file)




if __name__ == "__main__":
    main()