import pandas as pd
import json
import os

def read_excel_to_json(excel_file_path, output_json_path):
    # Read the Excel file with header on row 3 (index 3)
    df = pd.read_excel(excel_file_path, engine='openpyxl', header=3)
    
    # Clean the data - replace NaN values with empty strings
    df = df.fillna('')
    
    # Initialize the JSON structure
    country_data = {
        "country": {
            "code": "",
            "name": "",
            "center": "",
            "regions": []
        }
    }
    
    # Dictionary to hold all entries by their code
    entries = {}
    
    # Dictionaries to hold categorized entries
    regions = {}
    districts = {}
    settlements = {}
    
    # First pass: collect all entries
    for _, row in df.iterrows():
        code = str(row['МҲОБТ коди']).strip()
        name_latin = str(row['Объектнинг номи (лотинча)']).strip()
        center_latin = str(row['Маркази (лотинча)']).strip()
        
        # Clean data
        if name_latin == 'nan' or name_latin == '': 
            continue  # Skip empty entries
        if center_latin == 'nan': 
            center_latin = ''
        
        # Skip administrative category headers
        if ("viloyatining tumanlari" in name_latin.lower() or
            "tumanining shaharchalari" in name_latin.lower() or
            "tumanining qishloq fuqarolar yig'inlari" in name_latin.lower() or
            "tumanining tuman ahamiyatiga ega shaharlari" in name_latin.lower() or
            "viloyatining viloyat ahamiyatiga ega shaharlari" in name_latin.lower() or
            "shahar hokimiyatiga qarashli shaharchalari" in name_latin.lower() or
            "sh. xok-tiga qarashli qishloq fuqarolar yig'inlari" in name_latin.lower()):
            continue
        
        # Create entry object
        entry = {
            "code": code,
            "name": name_latin,
            "center": center_latin
        }
        
        entries[code] = entry
    
    # Second pass: categorize entries based on code length and pattern
    for code, entry in entries.items():
        # Country level (2 digits: "17")
        if code == "17":
            country_data["country"].update({
                "code": entry["code"],
                "name": entry["name"],
                "center": entry["center"]
            })
        
        # Region level (4 digits, like "1703")
        elif len(code) == 4 and code.startswith("17"):
            region_entry = entry.copy()
            region_entry["districts"] = []
            regions[code] = region_entry
        
        # District level (7 digits)
        elif len(code) == 7 and code.startswith("17"):
            district_entry = entry.copy()
            district_entry["settlements"] = []
            districts[code] = district_entry
        
        # Settlement level (10+ digits)
        elif len(code) >= 10 and code.startswith("17"):
            settlements[code] = entry.copy()
        
        # Any other entries with code starting with "17"
        elif code.startswith("17"):
            # Determine category based on code length
            if len(code) > 7:
                settlements[code] = entry.copy()
            elif len(code) > 4:
                district_entry = entry.copy()
                district_entry["settlements"] = []
                districts[code] = district_entry
    
    # Third pass: build the hierarchical structure
    # Add settlements to their respective districts
    for settlement_code, settlement in settlements.items():
        # Find parent district (first 7 characters)
        parent_district_code = settlement_code[:7]
        if parent_district_code in districts:
            districts[parent_district_code]["settlements"].append(settlement)
    
    # Add districts to regions
    for district_code, district in districts.items():
        # Find parent region (first 4 characters)
        region_code = district_code[:4]
        if region_code in regions:
            regions[region_code]["districts"].append(district)
    
    # Add regions to country
    country_data["country"]["regions"] = list(regions.values())
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(country_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON file created at: {output_json_path}")
    print(f"Total entries processed: {len(entries)}")
    print(f"Regions: {len(regions)}")
    print(f"Districts: {len(districts)}")
    print(f"Settlements: {len(settlements)}")

# Example usage
# Use the absolute path to the Excel file
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(current_dir, "СОАТО-20.04.2022.xlsx")
output_json = os.path.join(current_dir, "uzbekistan_admin_divisions.json")
read_excel_to_json(excel_file, output_json)