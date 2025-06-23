import pandas as pd
import json
import os

def xlsx_to_json(excel_file_path, output_json_path=None):
    """
    Convert Excel file to JSON
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_json_path (str, optional): Path to save the JSON file. If None, will save in the same directory with same name
    
    Returns:
        str: Path to the saved JSON file
    """
    try:
        # Check if file exists
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
        
        # Read Excel file
        print(f"Reading Excel file: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        
        # Generate output path if not provided
        if output_json_path is None:
            file_name = os.path.splitext(os.path.basename(excel_file_path))[0]
            output_dir = os.path.dirname(excel_file_path)
            output_json_path = os.path.join(output_dir, f"{file_name}.json")
        
        # Convert to JSON
        print(f"Converting to JSON...")
        data = df.to_dict(orient='records')
        
        # Save JSON file
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        print(f"JSON file saved at: {output_json_path}")
        return output_json_path
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Path to the Excel file
    excel_file = "4-Transport (AI chat uchun1).xlsx"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, excel_file)
    
    # Convert to JSON
    json_path = xlsx_to_json(excel_path)
    
    if json_path:
        print(f"Conversion completed successfully!")
    else:
        print(f"Conversion failed.")