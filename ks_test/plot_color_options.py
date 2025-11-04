"""
Generate multiple color/styling options for the combined p-value plots.
Run this to see different styling options and choose your favorite.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color scheme options
COLOR_SCHEMES = {
    "option1_vibrant": {
        "name": "Vibrant Contrast",
        "description": "High contrast, easy to distinguish",
        "colors": {
            "segnn": "#0173B2",       # Strong blue
            "ponita": "#DE8F05",      # Strong orange
            "equiformer_v2": "#029E73", # Teal/green
            "cgenn": "#CC78BC",       # Pink/magenta
            "graph_transformer": "#CA9161"          # Tan/brown
        },
        "linewidth": 2.5,
        "markersize": 5,
        "star_size": 200
    },
    
    "option2_colorblind": {
        "name": "Colorblind-Friendly (Wong palette)",
        "description": "Optimized for colorblind viewers",
        "colors": {
            "segnn": "#0173B2",       # Blue
            "ponita": "#ECE133",      # Yellow
            "equiformer_v2": "#029E73", # Bluish green
            "cgenn": "#DE8F05",       # Orange
            "graph_transformer": "#CC78BC"          # Reddish purple
        },
        "linewidth": 2.5,
        "markersize": 5,
        "star_size": 200
    },
    
    "option3_bold": {
        "name": "Bold & Thick",
        "description": "Thicker lines, larger markers for presentations",
        "colors": {
            "segnn": "#1f77b4",
            "ponita": "#ff7f0e",
            "equiformer_v2": "#2ca02c",
            "cgenn": "#d62728",
            "graph_transformer": "#9467bd"
        },
        "linewidth": 3.5,
        "markersize": 7,
        "star_size": 250
    },
    
    "option4_distinct": {
        "name": "Maximum Distinction",
        "description": "Different line styles + distinct colors",
        "colors": {
            "segnn": "#1E88E5",       # Blue
            "ponita": "#FFC107",      # Amber
            "equiformer_v2": "#00C853", # Green
            "cgenn": "#E91E63",       # Pink
            "graph_transformer": "#9C27B0"          # Purple
        },
        "linewidth": 2.5,
        "markersize": 6,
        "star_size": 200,
        "line_styles": {
            "segnn": "-",
            "ponita": "-",
            "equiformer_v2": "--",
            "cgenn": "-.",
            "graph_transformer": ":"
        }
    },
    
    "option5_seaborn": {
        "name": "Seaborn Deep",
        "description": "Professional, academic style",
        "colors": {
            "segnn": "#4C72B0",
            "ponita": "#DD8452",
            "equiformer_v2": "#55A868",
            "cgenn": "#C44E52",
            "graph_transformer": "#8172B3"
        },
        "linewidth": 2.5,
        "markersize": 5,
        "star_size": 180
    }
}

def print_color_swatches():
    """Print color options in a readable format"""
    print("=" * 80)
    print("COLOR & STYLING OPTIONS FOR P-VALUE PLOTS")
    print("=" * 80)
    print()
    
    for opt_key, opt_data in COLOR_SCHEMES.items():
        print(f"\n{opt_key.upper()}: {opt_data['name']}")
        print(f"  Description: {opt_data['description']}")
        print(f"  Line width: {opt_data['linewidth']}, Marker size: {opt_data['markersize']}")
        print(f"  Colors:")
        for model, color in opt_data['colors'].items():
            model_display = {
                "segnn": "SEGNN",
                "ponita": "Ponita", 
                "equiformer_v2": "EquiformerV2",
                "cgenn": "CGENN",
                "graph_transformer": "Graph Transformer"
            }[model]
            line_style = opt_data.get('line_styles', {}).get(model, '—')
            print(f"    {model_display:20} {color:10}  {line_style}")
        print()
    
    print("=" * 80)
    print("\nRECOMMENDATIONS:")
    print("  • For papers/publications: option1_vibrant or option5_seaborn")
    print("  • For presentations: option3_bold")
    print("  • For accessibility: option2_colorblind")
    print("  • For print (B&W): option4_distinct (uses line styles)")
    print()
    print("To apply an option, edit plot_combined_pvalues_multi_model.py")
    print("and replace the COLOR_SCHEME variable at the top of the file.")
    print("=" * 80)

if __name__ == "__main__":
    print_color_swatches()


