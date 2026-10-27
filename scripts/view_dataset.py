import os
import csv
from pathlib import Path
import http.server
import socketserver
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
metadata_path = PROJECT_ROOT / "data" / "metadata" / "metadata_chunks.csv"
html_path = PROJECT_ROOT / "dataset_viewer.html"

def generate_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dataset Viewer - 52 Files</title>
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px; }
            h1 { text-align: center; color: #38bdf8; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }
            th, td { padding: 15px; text-align: left; border-bottom: 1px solid #334155; }
            th { background-color: #38bdf8; color: #0f172a; font-weight: bold; }
            tr:hover { background-color: #334155; }
            audio { width: 300px; height: 40px; }
            .text-content { font-size: 1.1em; line-height: 1.5; max-width: 600px; }
            .badge { background: #0284c7; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>STT Dataset Viewer (52 Validated Files)</h1>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Audio</th>
                    <th>Transcript / Label</th>
                    <th>Language</th>
                    <th>Duration (s)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            audio_url = "/" + row["audio_path"]
            html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td><audio controls src="{audio_url}"></audio><br><small style="color:#94a3b8">{row['audio_path'].split('/')[-1]}</small></td>
                    <td class="text-content">{row['transcript']}</td>
                    <td><span class="badge">{row['language'].upper()}</span></td>
                    <td>{row['duration']}</td>
                </tr>
            """
            
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def start_server():
    PORT = 8080
    os.chdir(PROJECT_ROOT)
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Enable address reuse to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✅ UI Ready! Serving at port {PORT}")
        print(f"Open http://localhost:{PORT}/dataset_viewer.html in your browser.")
        httpd.serve_forever()

if __name__ == "__main__":
    print("Generating UI...")
    generate_html()
    start_server()
