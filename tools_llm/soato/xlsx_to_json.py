import pandas as pd
import json
import os
import re
from collections import defaultdict

def read_excel_to_json(excel_file_path, output_json_path):
    # Read the Excel file with header on row 3 (index 3)
    df = pd.read_excel(excel_file_path, engine='openpyxl', header=3)
    
    # Clean the data - replace NaN values with empty strings
    df = df.fillna('')
    
    # Initialize the JSON structure
    country_data = {
        "country": {
            "code": "",
            "name_latin": "",
            "name_cyrillic": "",
            "name_russian": "",
            "center_latin": "",
            "center_cyrillic": "",
            "center_russian": "",
            "regions": []
        }
    }
    
    # Dictionary to hold all entries by their code
    entries = {}
    
    # Dictionaries to hold categorized entries
    regions = {}
    districts = {}
    cities = {}
    urban_settlements = {}
    rural_assemblies = {}
    
    # First pass: collect all entries
    for _, row in df.iterrows():
        code = str(row['МҲОБТ коди'])
        name_latin = str(row['Объектнинг номи (лотинча)'])
        name_cyrillic = str(row['Объектнинг номи (кирилча)'])
        name_russian = str(row['Объектнинг \nрус тилидаги номи'])  # Note the newline in column name
        center_latin = str(row['Маркази (лотинча)'])
        center_cyrillic = str(row['Маркази (кирилча)'])
        center_russian = str(row['Маркази'])
        
        # Clean data
        if name_latin == 'nan': name_latin = ''
        if name_cyrillic == 'nan': name_cyrillic = ''
        if name_russian == 'nan': name_russian = ''
        if center_latin == 'nan': center_latin = ''
        if center_cyrillic == 'nan': center_cyrillic = ''
        if center_russian == 'nan': center_russian = ''
        
        # Create entry object
        entry = {
            "code": code,
            "name_latin": name_latin,
            "name_cyrillic": name_cyrillic,
            "name_russian": name_russian,
            "center_latin": center_latin,
            "center_cyrillic": center_cyrillic,
            "center_russian": center_russian
        }
        
        entries[code] = entry
    
    # Second pass: categorize entries
    for code, entry in entries.items():
        # Country level (2 digits)
        if code == "17":
            country_data["country"].update({
                "code": entry["code"],
                "name_latin": entry["name_latin"],
                "name_cyrillic": entry["name_cyrillic"],
                "name_russian": entry["name_russian"],
                "center_latin": entry["center_latin"],
                "center_cyrillic": entry["center_cyrillic"],
                "center_russian": entry["center_russian"]
            })
        
        # Region level (4 digits, like 1703)
        elif len(code) == 4 and code.startswith("17"):
            region_entry = entry.copy()
            region_entry["districts"] = []
            regions[code] = region_entry
        
        # District level (7 digits ending in specific patterns)
        elif len(code) == 7:
            # Districts usually end with 200, 203, etc. or contain "tumani"
            if (code.endswith("200") or code.endswith("203") or 
                "tumani" in entry["name_latin"].lower() or 
                "тумани" in entry["name_cyrillic"].lower() or
                "район" in entry["name_russian"].lower()):
                
                district_entry = entry.copy()
                district_entry["urban_settlements"] = []
                district_entry["rural_assemblies"] = []
                districts[code] = district_entry
        
        # Settlement level (10 digits)
        elif len(code) == 10:
            # Urban settlements (shaharchalar) - codes like 1703202550, 1703203550
            if (code.endswith("550") or 
                "shaharchalari" in entry["name_latin"].lower() or
                "шаҳарчалари" in entry["name_cyrillic"].lower() or
                "поселки" in entry["name_russian"].lower()):
                # This is a category header, skip it
                continue
            
            # Rural assemblies (qishloq fuqarolar yig'inlari) - codes like 1703202800
            elif (code.endswith("800") or
                  "qishloq fuqarolar yig'inlari" in entry["name_latin"].lower() or
                  "қишлоқ фуқаролар йиғинлари" in entry["name_cyrillic"].lower() or
                  "сходы граждан" in entry["name_russian"].lower()):
                # This is a category header, skip it
                continue
            
            # Individual urban settlements (codes like 1703202552, 1703202554, etc.)
            elif code[7:10] >= "550" and code[7:10] < "600":
                urban_settlements[code] = entry.copy()
            
            # Individual rural assemblies (codes like 1703202804, 1703202807, etc.)
            elif code[7:10] >= "800" and code[7:10] < "900":
                rural_assemblies[code] = entry.copy()
            
            # Other settlements that might be cities or villages
            else:
                # Determine if it's an urban settlement or rural assembly based on parent district
                parent_district_code = code[:7]
                if parent_district_code in districts:
                    # For now, treat as urban settlement
                    urban_settlements[code] = entry.copy()
    
    # Third pass: build the hierarchical structure
    # Add urban settlements and rural assemblies to their respective districts
    for district_code, district in districts.items():
        # Add urban settlements to this district
        for settlement_code, settlement in urban_settlements.items():
            if settlement_code.startswith(district_code):
                district["urban_settlements"].append(settlement)
        
        # Add rural assemblies to this district
        for assembly_code, assembly in rural_assemblies.items():
            if assembly_code.startswith(district_code):
                district["rural_assemblies"].append(assembly)
    
    # Add districts to regions
    for district_code, district in districts.items():
        region_code = district_code[:4]
        if region_code in regions:
            regions[region_code]["districts"].append(district)
    
    # Add regions to country
    country_data["country"]["regions"] = list(regions.values())
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(country_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON file has been created at: {output_json_path}")

# Example usage
# Use the absolute path to the Excel file
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(current_dir, "СОАТО-20.04.2022.xlsx")
output_json = os.path.join(current_dir, "uzbekistan_admin_divisions.json")
read_excel_to_json(excel_file, output_json)