from flask import Flask, render_template, request, send_file, url_for
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from functional import lifecycle_duration_plot, is_public_github_repo, basic_stats, generate_plots, get_plot_url, generate_ocdfg, convert_to_ocel_and_get_summary, generate_text_summary, generate_ocpn, contributor_analysis, file_analysis
from repo_parser import process_github_repo
from label_classific import predict
import tempfile
from datetime import datetime
import shutil
import json
import pickle

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['TEMP_FOLDER'] = tempfile.mkdtemp()

def save_results_locally(df, ocel_data, objects_summary, temporal_summary, plots, contributor_data, file_data, repo_url):
    # Extract repository name from URL
    repo_name = repo_url.split('/')[-1]
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create folder name
    folder_name = f"{repo_name}_{timestamp}"
    folder_path = os.path.join(os.getcwd(), 'analysis_results', folder_name)
    
    # Create folders
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'data'), exist_ok=True)
    
    try:
        # Save DataFrame
        df.to_csv(os.path.join(folder_path, 'data', 'commits_data.csv'), index=False)
        
        # Save OCEL data and summaries
        with open(os.path.join(folder_path, 'data', 'ocel_data.pkl'), 'wb') as f:
            pickle.dump(ocel_data, f)
        objects_summary.to_csv(os.path.join(folder_path, 'data', 'objects_summary.csv'))
        temporal_summary.to_csv(os.path.join(folder_path, 'data', 'temporal_summary.csv'))
        
        # Copy OCDFG and OCPN
        shutil.copy2(
            os.path.join(app.config['TEMP_FOLDER'], 'ocdfg.png'),
            os.path.join(folder_path, 'plots', 'ocdfg.png')
        )
        shutil.copy2(
            os.path.join(app.config['TEMP_FOLDER'], 'ocpn.png'),
            os.path.join(folder_path, 'plots', 'ocpn.png')
        )
        
        # Save plots
        for i, plot_data in enumerate(plots):
            plot_filename = f'plot_{i}.png'
            plot_path = os.path.join(folder_path, 'plots', plot_filename)
            with open(plot_path, 'wb') as f:
                f.write(base64.b64decode(plot_data))
        
        # Save contributor analysis
        with open(os.path.join(folder_path, 'data', 'contributor_analysis.json'), 'w') as f:
            # Convert plot URLs to actual files
            contributor_data_copy = contributor_data.copy()
            if 'cluster_plot' in contributor_data_copy:
                cluster_plot_path = os.path.join(folder_path, 'plots', 'cluster_plot.png')
                with open(cluster_plot_path, 'wb') as plot_file:
                    plot_file.write(base64.b64decode(contributor_data_copy['cluster_plot']))
                contributor_data_copy['cluster_plot'] = 'plots/cluster_plot.png'
            
            if 'network_plot' in contributor_data_copy:
                network_plot_path = os.path.join(folder_path, 'plots', 'network_plot.png')
                with open(network_plot_path, 'wb') as plot_file:
                    plot_file.write(base64.b64decode(contributor_data_copy['network_plot']))
                contributor_data_copy['network_plot'] = 'plots/network_plot.png'
            
            json.dump(contributor_data_copy, f, indent=4)
        
        # Save file analysis
        with open(os.path.join(folder_path, 'data', 'file_analysis.json'), 'w') as f:
            file_data_copy = file_data.copy()
            for key in file_data_copy:
                if key.endswith('_plot'):
                    plot_path = os.path.join(folder_path, 'plots', f'{key}.png')
                    with open(plot_path, 'wb') as plot_file:
                        plot_file.write(base64.b64decode(file_data_copy[key]))
                    file_data_copy[key] = f'plots/{key}.png'
            json.dump(file_data_copy, f, indent=4)
        
        return folder_path
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    result = ''
    download_links = []
    preview = None
    plots = None
    contributor_data = None
    file_data = None
    ocdfg_data = None
    ocpn_data = None
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        
        if is_public_github_repo(user_input):
            is_conventional_standard = False
            if request.form.get('checkbox'):
                is_conventional_standard = True
                df = process_github_repo(user_input, conventional_commits=is_conventional_standard)
            else:
                df = process_github_repo(user_input, conventional_commits=is_conventional_standard)
                df['ocel:activity'] = [predict(message) for message in df['commit_message']]
            
            # Save data.csv
            csv_filename = 'data.csv'
            csv_path = os.path.join(app.config['TEMP_FOLDER'], csv_filename)
            df.to_csv(csv_path, index=False)
            download_links.append(('data.csv', url_for('download_file', filename=csv_filename)))
            
            # Generate ocdfg.png
            ocel_data, temporal_summary, objects_summary = convert_to_ocel_and_get_summary(df)
            authors_list = df['ocel:type:author'].unique()


            ocdfg_filename = 'ocdfg.png'
            ocdfg_path = os.path.join(app.config['TEMP_FOLDER'], ocdfg_filename)
            generate_ocdfg(ocel_data, ocdfg_path)
            download_links.append(('ocdfg.png', url_for('download_file', filename=ocdfg_filename)))
            
            # Generate ocpn.png
            ocpn_filename = 'ocpn.png'
            ocpn_path = os.path.join(app.config['TEMP_FOLDER'], ocpn_filename)
            generate_ocpn(ocel_data, ocpn_path)
            download_links.append(('ocpn.png', url_for('download_file', filename=ocpn_filename)))
                        
            result = generate_text_summary(df, ocel_data, objects_summary)

            # Generate preview
            preview = df.head().to_html(classes='table table-striped table-hover', index=False)

            # Generate plots
            plots = generate_plots(df)

            # Generate contributor analysis
            contributor_data = contributor_analysis(df, ocel_data)

            # Generate files analysis
            objects_summary_files = objects_summary[~objects_summary['ocel:oid'].isin(authors_list)]
            file_data = file_analysis(objects_summary_files, ocel_data)
            
            # Generate OCDFG and OCPN
            ocdfg_path = os.path.join(app.config['TEMP_FOLDER'], 'ocdfg.png')
            ocpn_path = os.path.join(app.config['TEMP_FOLDER'], 'ocpn.png')
            
            generate_ocdfg(ocel_data, ocdfg_path)
            generate_ocpn(ocel_data, ocpn_path)
            
            # Read the generated images and convert to base64
            with open(ocdfg_path, 'rb') as f:
                ocdfg_data = base64.b64encode(f.read()).decode('utf-8')
            
            with open(ocpn_path, 'rb') as f:
                ocpn_data = base64.b64encode(f.read()).decode('utf-8')
            
            if request.form.get('save_locally'):
                saved_folder = save_results_locally(
                    df, ocel_data, objects_summary, temporal_summary,
                    plots, contributor_data, file_data, user_input
                )
                if saved_folder:
                    result += f"\n\nResults saved locally in: {saved_folder}"
        
        else:
            result = "The input is not a valid GitHub link to a public repository."
        
    return render_template('index.html', 
                         user_input=user_input, 
                         result=result, 
                         download_links=download_links,
                         preview=preview, 
                         plots=plots, 
                         contributor_data=contributor_data, 
                         file_analysis=file_data,
                         ocdfg_data=ocdfg_data,
                         ocpn_data=ocpn_data)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['TEMP_FOLDER'], filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
