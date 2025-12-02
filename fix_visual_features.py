"""
Fix visual_features in data1.json to use programmatic feature names
instead of descriptive text for proper feature matching.
"""

import json

# Mapping of diseases to their correct programmatic visual features
VISUAL_FEATURE_MAPPING = {
    "splinter_hemorrhage": [
        "vertical_lines",
        "red_areas"
    ],
    "muehrcke_s_lines": [
        "horizontal_lines"
    ],
    "beau_s_lines": [
        "horizontal_lines"
    ],
    "koilonychia": [
        "spoon_shape",
        "nail_thickening"
    ],
    "clubbing": [
        "nail_clubbing"
    ],
    "darier_s_disease": [
        "vertical_lines",
        "v_shaped_notch"
    ],
    "leukonychia": [
        "white_discoloration"
    ],
    "bluish_nail": [
        "blue_discoloration"
    ],
    "pale_nail": [
        "pale_appearance"
    ],
    "yellow_nails": [
        "yellow_discoloration"
    ],
    "white_nail": [
        "white_discoloration",
        "pale_appearance"
    ],
    "aloperia_areata": [
        "nail_pitting",
        "surface_roughness"
    ],
    "eczema": [
        "nail_pitting",
        "horizontal_lines",
        "surface_roughness"
    ],
    "onycholysis": [
        "nail_separation"
    ],
    "red_lunula": [
        "red_areas"
    ],
    "terry_s_nail": [
        "white_discoloration",
        "pale_appearance"
    ],
    "half_and_half_nails": [
        "white_discoloration"
    ]
}

# Load data1.json
with open("data1.json", "r", encoding="utf-8") as f:
    diseases = json.load(f)

# Update visual_features for each disease
for disease in diseases:
    disease_name = disease.get("name", "")
    
    if disease_name in VISUAL_FEATURE_MAPPING:
        # Save original as descriptions
        original_features = disease.get("visual_features", [])
        disease["visual_feature_descriptions"] = original_features
        
        # Replace with programmatic names
        disease["visual_features"] = VISUAL_FEATURE_MAPPING[disease_name]
        
        print(f"✓ Updated {disease_name}")
        print(f"  Old: {original_features[:2]}...")
        print(f"  New: {disease['visual_features']}")
    else:
        print(f"⚠ Skipped {disease_name} (no mapping defined)")

# Save updated data
with open("data1.json", "w", encoding="utf-8") as f:
    json.dump(diseases, f, indent=4, ensure_ascii=False)

print("\n✓ data1.json updated successfully!")
print("\nNow restart the server to see the changes take effect.")
