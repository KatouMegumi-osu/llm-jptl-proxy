# parse_jmdict.py
import xml.etree.ElementTree as ET
import json
from collections import defaultdict

def parse_jmdict_to_json(input_file='JMdict_e', output_file='jmdict.json'):
    """
    Parses the JMdict_e XML file and creates a simplified JSON dictionary.
    This process can take a minute or two.
    """
    print(f"Starting parsing of '{input_file}'...")
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"ERROR: The dictionary file '{input_file}' was not found.")
        print("Please download and place it in the same directory as this script.")
        return
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse the XML file. It might be corrupted. Details: {e}")
        return

    dictionary = {}
    entry_count = 0
    
    # JMdict has <entry> tags for each word.
    for entry in root.findall('entry'):
        entry_count += 1
        if entry_count % 10000 == 0:
            print(f"  Processed {entry_count} entries...")

        # Extract all possible Japanese forms (kanji and kana)
        japanese_forms = set()
        for k_ele in entry.findall('k_ele'): # Kanji elements
            keb = k_ele.find('keb')
            if keb is not None:
                japanese_forms.add(keb.text)
        
        for r_ele in entry.findall('r_ele'): # Reading elements (kana)
            reb = r_ele.find('reb')
            if reb is not None:
                japanese_forms.add(reb.text)
        
        # Extract English definitions (glosses)
        definitions = []
        for sense in entry.findall('sense'):
            # We will take the first 3 glosses for conciseness
            glosses = [g.text for g in sense.findall('gloss')[:3]]
            if glosses:
                definitions.extend(glosses)
        
        if not definitions:
            continue
            
        # Combine definitions into a single string
        final_definition = "; ".join(definitions)
        
        # Add all forms of the word to our dictionary
        for form in japanese_forms:
            if form not in dictionary: # Avoid overwriting with less common definitions
                dictionary[form] = final_definition

    print(f"Parsing complete. Found {len(dictionary)} unique word forms.")
    
    print(f"Saving to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
        
    print("Done. You can now run the main proxy application.")

if __name__ == "__main__":
    parse_jmdict_to_json()
