import pandas as pd
import json
import os
import numpy as np

def xlsx_to_json(excel_file_path, output_json_path=None, sheet_name=None):
    """
    Convert Excel file to JSON, specifically formatted for 12-invest report
    in a structure similar to transport4.json
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_json_path (str, optional): Path to save the JSON file. If None, will save in the same directory with same name
        sheet_name (str or int, optional): Name or index of the sheet to convert. If None, all sheets will be processed
    
    Returns:
        str: Path to the saved JSON file
    """
    try:
        # Check if file exists
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
        
        # Read Excel file
        print(f"Reading Excel file: {excel_file_path}")
        
        # Read all sheets to check structure
        xls = pd.ExcelFile(excel_file_path)
        sheet_names = xls.sheet_names
        print(f"Found sheets: {sheet_names}")
        
        # Process each sheet or specific sheet if provided
        sheets_to_process = [sheet_name] if sheet_name is not None else sheet_names
        
        # Create transport4.json-like structure
        result = [
            {
                "report_form": {
                    "name": "12-invest shakli",
                    "submitters": "Investitsiyalar va qurilish faoliyati to'g'risidagi hisobot",
                    "structure": f"{len(sheets_to_process)}ta bob, titul va yakuniy qismdan iborat",
                    "indicators": "Hisobot asosida investitsiyalar va qurilish faoliyati to'g'risidagi ma'lumotlar aniqlanadi"
                },
                "sections": []
            }
        ]
        
        # Process each sheet as a separate section (bob)
        for sheet_index, sheet in enumerate(sheets_to_process):
            if sheet not in sheet_names:
                print(f"Warning: Sheet '{sheet}' not found, skipping.")
                continue
                
            print(f"Processing sheet: {sheet}")
            
            try:
                # Try to read the Excel file with different header options to find the best structure
                # First, try without any header
                df_no_header = pd.read_excel(excel_file_path, sheet_name=sheet, header=None)
                
                # Look for the row with most non-NaN values in the first 10 rows
                max_non_na_count = 0
                header_row = 0
                for i in range(min(10, len(df_no_header))):
                    non_na_count = df_no_header.iloc[i].notna().sum()
                    if non_na_count > max_non_na_count:
                        max_non_na_count = non_na_count
                        header_row = i
                
                print(f"Detected header row at index {header_row}")
                
                # Read the data with the detected header row
                df = pd.read_excel(excel_file_path, sheet_name=sheet, header=header_row)
                
                # Clean column names
                df.columns = [str(col).strip() for col in df.columns]
                
                # Replace NaN values with None for proper JSON serialization
                df = df.replace({np.nan: None, 'nan': None, 'NaN': None})
                
                # Create section structure
                section = {
                    "bob": f"{sheet_index+1}-BOB",
                    "title": sheet,
                    "description": f"12-invest shakli {sheet} bo'limi",
                    "section_type": "statistic",
                    "rows": []
                }
                
                # Process each row in the dataframe
                row_counter = 1
                for row_index, row_data in df.iterrows():
                    # Skip rows where all values are None or NaN
                    if all(pd.isna(x) or x is None or x == '' for x in row_data.values):
                        continue
                    
                    # Create row structure similar to transport4.json
                    row_dict = {
                        "row_code": str(row_counter),
                        "row_name": f"Satr {row_counter}",  # Default row name
                        "columns": []
                    }
                    row_counter += 1
                    
                    # Try to find a row name/title in the first column
                    first_col = df.columns[0] if len(df.columns) > 0 else None
                    if first_col is not None:
                        first_col_value = row_data[first_col]
                        if first_col_value is not None and not pd.isna(first_col_value) and str(first_col_value).strip():
                            row_dict["row_name"] = str(first_col_value).strip()
                    
                    # Process each column in the row
                    for col_index, col_name in enumerate(df.columns):
                        value = row_data[col_name]
                        
                        # Skip empty cells or nan values
                        if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                            continue
                        
                        # Format value based on type
                        if isinstance(value, (int, float)):
                            formatted_value = str(value)
                        else:
                            formatted_value = str(value).strip()
                        
                        # Clean column name
                        clean_col_name = str(col_name).strip()
                        if not clean_col_name or clean_col_name == 'nan' or pd.isna(clean_col_name):
                            clean_col_name = f"Ustun {col_index + 1}"
                        
                        # Create column structure similar to transport4.json
                        column_dict = {
                            "column": str(col_index + 1),
                            "description": clean_col_name,
                            "strict_logical_controls": [],
                            "non_strict_logical_controls": []
                        }
                        
                        # Add value to description like in transport4.json
                        if ":" not in clean_col_name:
                            column_dict["description"] = f"{clean_col_name}: {formatted_value}"
                        else:
                            column_dict["description"] = f"{clean_col_name} {formatted_value}"
                        
                        row_dict["columns"].append(column_dict)
                    
                    # Only add rows that have columns
                    if row_dict["columns"]:
                        section["rows"].append(row_dict)
                
                # Only add section if it has rows
                if section["rows"]:
                    result[0]["sections"].append(section)
                    
            except Exception as sheet_error:
                print(f"Error processing sheet {sheet}: {str(sheet_error)}")
                continue
        
        # Generate output path if not provided
        if output_json_path is None:
            file_name = os.path.splitext(os.path.basename(excel_file_path))[0]
            output_dir = os.path.dirname(excel_file_path)
            output_json_path = os.path.join(output_dir, f"{file_name}.json")
        
        # Save JSON file
        print(f"Converting to JSON in transport4.json format...")
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        
        print(f"JSON file saved at: {output_json_path}")
        return output_json_path
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Path to the Excel file
    excel_file = "12 invest Hisobot (AI chat uchun).xlsx"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, excel_file)
    
    # Convert to JSON
    json_path = xlsx_to_json(excel_path)
    
    if json_path:
        print(f"Conversion completed successfully!")
    else:
        print(f"Conversion failed.")