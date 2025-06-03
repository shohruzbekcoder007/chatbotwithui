import re

test_str = '01.11.11.1 qaysi mahsulot kodi?'
print(f"Test string: {test_str}")

# Turli xil regex variantlarini sinab ko'ramiz
regex_patterns = [
    r'\b(\d+(?:\.\d+){0,3})\b',
    r'\b(\d+(\.\d+)*(\.\d+)?)\b',
    r'(\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+|\d+\.\d+|\d+)',
    r'(\d+\.\d+\.\d+\.\d+)',
    r'(\d+\.\d+\.\d+)',
    r'(\d+\.\d+)',
    r'(\d+)',
    r'(\d+[\.\d]*)'
]

for i, pattern in enumerate(regex_patterns):
    match = re.search(pattern, test_str)
    result = match.group(1) if match else 'Topilmadi'
    print(f"Pattern {i+1}: {pattern} -> {result}")
