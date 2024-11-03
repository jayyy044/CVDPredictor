import os
import re

def list_top_level_contents(folder_path):
    # Check if the path exists
    if not os.path.exists(folder_path):
        print("The specified path does not exist.")
        return
    
    print(f"Listing contents of: {folder_path}")
    
    # List directories and files
    directories = []
    ages = []
    heights = []
    weights = []
    
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            directories.append(entry)


    # Print directories
    if directories:
        print("Directories:")
        for dir_name in directories:
            print(f"  - {dir_name}")
            # Call the function recursively for each directory
            dictOfValues = list_readme_files(os.path.join(folder_path, dir_name))
            ages.append(dictOfValues['Age'])
            heights.append(dictOfValues['Height'])
            weights.append(dictOfValues['Weight'])
    ageTotal = 0 
    heightTotal = 0 
    weightTotal = 0

    for age in ages:
        ageTotal += age
    for height in heights:
        heightTotal+= height
    for weight in weights:
        weightTotal += weight
    print()
    print("The Average Age for the analyzed individuals is: " , round(ageTotal/len(ages),2))
    print("The Average Height(cm) for the analyzed individuals is: " , round(heightTotal/len(heights),2))
    print("The Average Weight(Kg) for the analyzed individuals is: " , round(weightTotal/len(weights),2))



    
    print("\n" + "-"*40)

def list_readme_files(directory):
    # Look for README files in the directory

    for entry in os.listdir(directory):
        if 'readme' in entry.lower() and os.path.isfile(os.path.join(directory, entry)):
            readme_path = os.path.join(directory, entry)
            print("    Found README file")
            # Read and print the contents of the README file
            with open(readme_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                # Extract personal information
                personal_info = extract_personal_info(contents)
                print(f"    Extracted Information: {personal_info}")
                return personal_info

def extract_personal_info(content):
    # Regular expression to find personal information
    info = {}
    age = []
    height = []
    weight = []
    
    # Extracting values using regex patterns
    age_match = re.search(r'Age:\s*(\d+)', content)
    height_match = re.search(r'Height \(cm\):\s*(\d+)', content)
    weight_match = re.search(r'Weight \(kg\):\s*(\d+)', content)
   
    
    if age_match:
        info['Age'] = int(age_match.group(1))
        age.append(int(age_match.group(1)))
    if height_match:
        info['Height'] = int(height_match.group(1))
        height.append(int(height_match.group(1)))
    if weight_match:
        info['Weight'] = int(weight_match.group(1))
        weight.append(int(weight_match.group(1)))



    return info

# Example usage
folder_path = r"C:\Users\Maanas\Downloads\WESAD\WESAD"
list_top_level_contents(folder_path)