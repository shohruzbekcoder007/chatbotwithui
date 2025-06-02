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
        # Country level
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
        
        # Region level
        elif code.startswith("17") and len(code) == 4:
            region_entry = entry.copy()
            region_entry["districts"] = []
            region_entry["cities"] = []
            regions[code] = region_entry
        
        # District level
        elif len(code) == 7 and (code.endswith("00") or 
                              entry["name_latin"].lower().endswith('tumani') or 
                              entry["name_cyrillic"].lower().endswith('тумани')):
            district_entry = entry.copy()
            district_entry["urban_settlements"] = []
            district_entry["rural_assemblies"] = []
            district_entry["cities"] = []
            districts[code] = district_entry
        
        # City level
        elif len(code) == 7 and (code.endswith("01") or code.endswith("05") or code.endswith("08") or 
                              entry["name_latin"].endswith('sh.') or entry["name_latin"].endswith('shahar') or 
                              entry["name_cyrillic"].endswith('ш.') or entry["name_cyrillic"].endswith('шаҳар') or
                              'shahar' in entry["name_latin"].lower() or 'shaxar' in entry["name_latin"].lower() or
                              'город' in entry["name_russian"].lower() or 'г.' in entry["name_russian"]):
            cities[code] = entry.copy()
        
        # Urban settlement level
        elif len(code) == 7 and (code.endswith("550") or code.endswith("50") or
                              entry["name_latin"].lower().endswith('shaharchasi') or
                              entry["name_cyrillic"].lower().endswith('шаҳарчаси')):
            urban_settlements[code] = entry.copy()
        
        # Rural assembly level
        elif len(code) == 7 and (code.endswith("800") or
                              entry["name_latin"].lower().endswith('qishloq fuqarolar yig\'ini') or
                              entry["name_cyrillic"].lower().endswith('қишлоқ фуқаролар йиғини')):
            rural_assemblies[code] = entry.copy()
    
    # Third pass: build the hierarchical structure
    # Third pass: build the hierarchical structure
    
    # Create a mapping of city names to their objects
    city_name_map = {}
    for city_code, city in cities.items():
        city_name_map[city["name_latin"].lower()] = city
        if city["name_cyrillic"]:
            city_name_map[city["name_cyrillic"].lower()] = city
    
    # First, add cities to districts based on district centers
    for district_code, district in districts.items():
        # Extract city name from district center
        center_name = district["center_latin"]
        if center_name:
            city_name = center_name.split()[0].lower()  # Get first word of center name
            
            # Find matching city
            for name, city in city_name_map.items():
                if city_name in name or name in city_name:
                    district["cities"].append(city.copy())
                    break
    
    # Add remaining cities to regions
    for city_code, city in cities.items():
        region_code = city_code[:4]
        if region_code in regions:
            regions[region_code]["cities"].append(city)
    
    # Add districts to regions and add urban settlements and rural assemblies to districts
    for district_code, district in districts.items():
        region_code = district_code[:4]
        
        # Add district to region
        if region_code in regions:
            # Add urban settlements to this district
            for settlement_code, settlement in urban_settlements.items():
                if settlement_code.startswith(district_code[:6]):
                    district["urban_settlements"].append(settlement)
            
            # Add rural assemblies to this district
            for assembly_code, assembly in rural_assemblies.items():
                if assembly_code.startswith(district_code[:6]):
                    district["rural_assemblies"].append(assembly)
            
            # Add district to region
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