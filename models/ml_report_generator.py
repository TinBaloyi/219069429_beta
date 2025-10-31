"""
ML Report Generator - Creates HTML reports from model cards
"""
from pathlib import Path
from datetime import datetime
import json

def generate_ml_report_html(domain, model_cards, output_path, predictions_sample=None):
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{domain.capitalize()} ML Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        .features-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }}
        .feature-tag {{
            background: #e9ecef;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{domain.capitalize()} ML Model Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
"""
    
    # Add model cards
    for model_name, card in model_cards.items():
        html_content += f"""
    <div class="card">
        <h2>{model_name.upper()} Model</h2>
        """
        
        # Metrics
        if "metrics" in card:
            html_content += "<h3>Performance Metrics</h3>"
            for split_name, metrics in card["metrics"].items():
                html_content += f"<h4>{split_name.upper()} Set</h4>"
                html_content += '<div class="metric-grid">'
                for metric_name, value in metrics.items():
                    html_content += f"""
                    <div class="metric">
                        <div class="metric-label">{metric_name}</div>
                        <div class="metric-value">{value:.3f}</div>
                    </div>
                    """
                html_content += '</div>'
        
        # Features
        if "features" in card:
            html_content += "<h3>Features Used</h3>"
            for feat_type, feat_list in card["features"].items():
                if feat_list:
                    html_content += f"<h4>{feat_type.capitalize()}</h4>"
                    html_content += '<div class="features-list">'
                    for feat in feat_list:
                        html_content += f'<span class="feature-tag">{feat}</span>'
                    html_content += '</div>'
        
        # Split sizes
        if "split_sizes" in card:
            html_content += "<h3>Dataset Splits</h3>"
            html_content += '<div class="metric-grid">'
            for split, size in card["split_sizes"].items():
                html_content += f"""
                <div class="metric">
                    <div class="metric-label">{split}</div>
                    <div class="metric-value">{size:,}</div>
                </div>
                """
            html_content += '</div>'
        
        html_content += "</div>"
    
    # Add predictions sample
    if predictions_sample:
        html_content += """
    <div class="card">
        <h2>Sample Predictions</h2>
        <table>
            <thead>
                <tr>
"""
        # Get column names from first prediction
        if predictions_sample:
            cols = list(predictions_sample[0].keys())[:8]  # Limit columns
            for col in cols:
                html_content += f"<th>{col}</th>"
        
        html_content += """
                </tr>
            </thead>
            <tbody>
"""
        for pred in predictions_sample[:10]:  # Show first 10 predictions
            html_content += "<tr>"
            for col in cols:
                value = pred.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.3f}"
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
            </tbody>
        </table>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[INFO] Generated ML report: {output_path}")