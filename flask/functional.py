import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pm4py
import seaborn as sns
import io
import base64
import requests
import os
from sklearn.cluster import DBSCAN
import networkx as nx

def generate_text_summary(original_data, ocel_data, objects_summary):
    """
    Generate a text summary of the repository data.

    Parameters:
    original_data: pandas dataframe.
    ocel_data: OCEL converted data.
    objects_summary: objects summary of the OCEL data.
    Returns:
    str: A formatted string containing the summary of the repository data.
    """
    get_object_types = pm4py.ocel.ocel_get_object_types(ocel_data)
    get_attribute_names = pm4py.ocel.ocel_get_attribute_names(ocel_data)
    unique_contributors, average_commits_per_contributor = basic_stats(original_data)

    result = f"""Repository processed successfully.
• Data shape: {original_data.shape}
• Number of unique contributors: {unique_contributors}
• Average number of commits per contributor: {average_commits_per_contributor:.2f}
• {len(get_object_types)} object types: {', '.join(get_object_types)}
• {len(get_attribute_names)} attributes: {', '.join(get_attribute_names)}
• Total objects: {len(objects_summary)}
• Objects with lifecycle duration > 0: {len(objects_summary[objects_summary.lifecycle_duration > 0])}"""
    
    return result
           
            
def convert_to_ocel_and_get_summary(data):
    ocel_data = pm4py.convert.convert_log_to_ocel(data,
                                              activity_column = 'ocel:activity',
                                              timestamp_column = 'ocel:timestamp',
                                              object_types = ['ocel:type:files', 'ocel:type:author'],
                                              obj_separator = ',',
                                              additional_event_attributes = ['commit_message', 'author_email', 'merge'])
    # ocel_data.event_timestamp = pd.to_datetime(ocel_data.event_timestamp)
    temporal_summary = pm4py.ocel_temporal_summary(ocel_data)
    objects_summary = pm4py.ocel.ocel_objects_summary(ocel_data)
    return ocel_data, temporal_summary, objects_summary


def lifecycle_duration_plot(objects_summary):
    fig = sns.displot(objects_summary.lifecycle_duration)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig.fig)  # Close the figure to free memory
    buf.seek(0)

    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_url


def is_public_github_repo(input_text):
    # Regular expression to match GitHub repository URLs
    github_url_pattern = r'https?://github\.com/[\w-]+/[\w.-]+'

    # Check if the input matches the GitHub URL pattern
    match = re.match(github_url_pattern, input_text)

    if not match:
        return False

    # Extract the API URL for the repository
    api_url = f"https://api.github.com/repos/{input_text.split('github.com/')[1]}"

    try:
        # Send a GET request to the GitHub API
        response = requests.get(api_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            repo_data = response.json()
            # Check if the repository is public
            return not repo_data.get('private', True)
        else:
            return False
    except requests.RequestException:
        return False

def basic_stats(df):
    unique_contributors = df.author_email.nunique()
    average_commits_per_contributor = len(df) / df.author_email.nunique()
    return unique_contributors, average_commits_per_contributor

def plot_commits_per_contributor(df):
    plt.figure(figsize=(12, 8), dpi=300)
    commits_per_contributor = df['ocel:type:author'].value_counts()
    
    if len(commits_per_contributor) >= 20:
        top_contributors = commits_per_contributor.nlargest(20)
        ax = top_contributors.plot(kind='bar')
        plt.title('Top 20 Contributors by Commits', fontsize=14, pad=20)
    else:
        ax = commits_per_contributor.plot(kind='bar')
        plt.title('Commits per Contributor', fontsize=14, pad=20)
    
    plt.xlabel('Contributor', fontsize=12)
    plt.ylabel('Number of Commits', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return get_plot_url(dpi=300)

def plot_commits_over_time(df):
    plt.figure(figsize=(12, 8), dpi=300)
    df['ocel:timestamp'] = pd.to_datetime(df['ocel:timestamp'], utc=True)
    df = df.set_index('ocel:timestamp').sort_index()
    ax = df.resample('W')['ocel:eid'].count().plot(linewidth=2)
    
    plt.title('Commits Over Time', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Commits', fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return get_plot_url(dpi=300)

def plot_commit_activity_heatmap(df):
    plt.figure(figsize=(15, 10), dpi=300)
    df['ocel:timestamp'] = pd.to_datetime(df['ocel:timestamp'], utc=True)
    
    df['day_of_week'] = df['ocel:timestamp'].dt.day_name()
    df['hour'] = df['ocel:timestamp'].dt.hour
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    sns.heatmap(heatmap_data, 
                cmap='YlGnBu', 
                annot=True, 
                fmt='d',
                cbar_kws={'label': 'Number of Commits'},
                annot_kws={'size': 8},
                linewidths=0.5)
    
    plt.title('Commit Activity Heatmap', fontsize=14, pad=20)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    return get_plot_url(dpi=300)

def plot_commit_type_proportions(df):
    plt.figure(figsize=(8, 8))
    
    # Count the occurrences of each commit type
    commit_types = df['ocel:activity'].value_counts()
    
    # Combine small categories into 'Other'
    threshold = 5  # Define a threshold for the minimum count to display
    commit_types = commit_types[commit_types >= threshold]
    
    # If there are any categories below the threshold, combine them into 'Other'
    if len(commit_types) < len(df['ocel:activity'].value_counts()):
        other_count = df['ocel:activity'].value_counts()[df['ocel:activity'].value_counts() < threshold].sum()
        # Create a new Series for 'Other'
        other_series = pd.Series({'Other': other_count})
        # Use pd.concat to combine the two Series
        commit_types = pd.concat([commit_types, other_series])

    # Create a pie chart
    plt.pie(commit_types, labels=commit_types.index, autopct='%1.1f%%', startangle=140)
    plt.title('Proportion of Each Commit Type')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)

    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_url

def plot_commit_type_trends(df):
    plt.figure(figsize=(12, 6))
    
    # Ensure timestamp is datetime
    df['ocel:timestamp'] = pd.to_datetime(df['ocel:timestamp'], utc=True)
    
    # Create a DataFrame with commit counts per type over time
    df_trends = df.set_index('ocel:timestamp')
    
    # Resample by week and count occurrences of each commit type
    commit_trends = pd.get_dummies(df_trends['ocel:activity']).resample('W').sum()
    min_commits = 10
    frequent_types = commit_trends.sum()[commit_trends.sum() > min_commits].index
    commit_trends = commit_trends[frequent_types]
    # Apply rolling average to smooth the lines (adjust window size as needed)
    window_size = 4  # 4-week rolling average
    smoothed_trends = commit_trends.rolling(window=window_size, min_periods=1).mean()
    
    # Plot trends for each commit type
    for column in smoothed_trends.columns:
        plt.plot(smoothed_trends.index, smoothed_trends[column], label=column, marker='', linewidth=2)
    
    plt.title('Commit Type Trends Over Time (4-Week Rolling Average)')
    plt.xlabel('Date')
    plt.ylabel('Number of Commits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_url

def generate_plots(df):
    plots = []
    
    # Generate each plot and append the URLs to the plots list
    plots.append(plot_commits_per_contributor(df))
    plots.append(plot_commits_over_time(df))
    plots.append(plot_commit_activity_heatmap(df))
    plots.append(plot_commit_type_proportions(df))
    plots.append(plot_commit_type_trends(df))
    return plots

def get_plot_url(dpi=300):
    """Enhanced version of get_plot_url with quality settings"""
    img = io.BytesIO()
    plt.savefig(img, 
                format='png',
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_ocdfg(ocel_data, filepath):
    ocdfg = pm4py.ocel.discover_ocdfg(ocel_data)
    
    # Increase the DPI (dots per inch) for higher resolution
    # Adjust the figure size for a larger image
    # Set a white background to ensure clarity
    pm4py.save_vis_ocdfg(ocdfg, filepath, 
                         annotation='frequency',
                         bgcolor="white",
                         dpi=300,
                         figsize=(20, 16))

def generate_ocpn(ocel_data, filepath):
    # Discover the Object-Centric Petri Net from the OCEL data
    ocpn = pm4py.discover_oc_petri_net(ocel_data)
    
    # Increase the DPI (dots per inch) for higher resolution
    # Adjust the figure size for a larger image
    # Set a white background to ensure clarity
    pm4py.save_vis_ocpn(ocpn, filepath, 
                        parameters={
                            "bgcolor": "white",
                            "format": "png",
                            "dpi": 300,
                            "figsize": (20, 16)
                        })

def create_author_mapping(data):
    """
    Create a bidirectional mapping between author emails and names.
    
    Parameters:
    data: DataFrame containing author information
    
    Returns:
    tuple: (email_to_name dict, name_to_email dict)
    """
    email_to_name = dict(zip(data['author_email'], data['ocel:type:author']))
    name_to_email = dict(zip(data['ocel:type:author'], data['author_email']))
    return email_to_name, name_to_email

def contributor_analysis(data, ocel_data):
    # Create author mapping
    email_to_name, name_to_email = create_author_mapping(data)
    
    # Count the number of commits per contributor (using emails first, then map to names)
    commits_per_contributor = data['author_email'].value_counts()
    top_contributors = commits_per_contributor.head(5)

    originator_by_task_matrix = pd.pivot_table(data, 
                                           values='ocel:eid', 
                                           index=['author_email'],
                                           columns=['ocel:activity'], 
                                           aggfunc="count",
                                           fill_value=0)
    
    matrix_df = pd.DataFrame(originator_by_task_matrix)
    clustering = DBSCAN().fit(matrix_df)
    labels = clustering.labels_

    # Generate the histogram plot
    plt.figure(figsize=(10, 6))
    fig = sns.histplot(labels, kde=False)
    plt.title('Contributor Clustering')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Contributors')
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    fig.figure.savefig(buf, format='png')
    plt.close(fig.figure)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Create the cluster summary table
    cluster_summary = []
    all_authors = []
    
    # Sort cluster labels to ensure desired order
    sorted_cluster_labels = sorted(set(labels), key=lambda x: (x != -1, x))
    
    for cluster_id in sorted_cluster_labels:
        if cluster_id == -1:
            cluster_name = "Most Active Contributors"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        cluster_authors_emails = matrix_df.index[labels == cluster_id].tolist()
        cluster_authors_names = [email_to_name[email] for email in cluster_authors_emails]
        cluster_data = data[data['author_email'].isin(cluster_authors_emails)]
        
        total_commits = cluster_data.shape[0]
        total_files = cluster_data['ocel:type:files'].str.split(',').explode().nunique()
        avg_files_per_author = cluster_data.groupby('author_email')['ocel:type:files'].apply(
            lambda x: x.str.split(',').explode().nunique()
        ).mean()
        
        avg_commits = total_commits / len(cluster_authors_emails) if cluster_authors_emails else 0
        
        cluster_summary.append({
            'Cluster ID': cluster_name,
            'Number of Authors': len(cluster_authors_emails),
            'Total Commits': total_commits,
            'Average Commits': avg_commits,
            'Total Files Edited': total_files,
            'Avg Files per Author': avg_files_per_author
        })
        
        all_authors.extend([
            (name, cluster_name, avg_commits) 
            for name in sorted(cluster_authors_names)
        ])

    cluster_summary_df = pd.DataFrame(cluster_summary)
    
    # Format numeric columns
    cluster_summary_df['Average Commits'] = cluster_summary_df['Average Commits'].apply(lambda x: f"{x:.2f}")
    cluster_summary_df['Avg Files per Author'] = cluster_summary_df['Avg Files per Author'].apply(lambda x: f"{x:.2f}")

    # Sort all_authors list by cluster order, then by author name
    all_authors.sort(key=lambda x: (
        sorted_cluster_labels.index(int(x[1].split()[-1]) if x[1] != "Most Active Contributors" else -1), 
        x[0]
    ))
    all_authors = [f"{author} ({cluster})" for author, cluster, _ in all_authors]
    
    # Get most active authors' emails and generate network
    most_active_authors_emails = matrix_df.index[labels == -1].tolist()
    network_analysis = generate_author_network(ocel_data, most_active_authors_emails, email_to_name)
    
    return {
        'top_contributors': {email_to_name[email]: count for email, count in top_contributors.items()},
        'cluster_plot': plot_url,
        'cluster_summary': cluster_summary_df.to_dict('records'),
        'all_authors': all_authors,
        'network_plot': network_analysis['plot'],
        'network_metrics': network_analysis['metrics']
    }

def generate_author_network(ocel_data, authors, email_to_name):
    """
    Generate a social network graph showing interactions between selected authors based on shared file edits.
    
    Parameters:
    ocel_data: OCEL converted data
    authors: list of author emails to include in the network
    email_to_name: dictionary mapping emails to names
    
    Returns:
    dict: Contains plot URL and network metrics
    """
    # Flatten the OCEL data for files
    flat = pm4py.ocel_flattening(ocel_data, 'ocel:type:files')
    
    # Filter for selected authors
    flat = flat[flat['author_email'].isin(authors)]
    
    # Create a dictionary to store file-author relationships
    file_authors = {}
    for _, row in flat.iterrows():
        file_name = row['case:concept:name']
        author = row['author_email']
        if file_name not in file_authors:
            file_authors[file_name] = set()
        file_authors[file_name].add(author)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges between authors who edited the same files
    for file_name, authors in file_authors.items():
        authors = list(authors)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                if G.has_edge(authors[i], authors[j]):
                    G[authors[i]][authors[j]]['weight'] += 1
                else:
                    G.add_edge(authors[i], authors[j], weight=1)
    
    # Relabel nodes with author names
    G = nx.relabel_nodes(G, email_to_name)
    
    # Calculate node sizes based on number of connections
    node_sizes = [G.degree(node) * 100 for node in G.nodes()]
    
    # Calculate edge widths based on weight
    edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    # Create the plot
    plt.figure(figsize=(15, 15))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw(G, pos,
            node_color='lightblue',
            node_size=node_sizes,
            width=edge_widths,
            with_labels=True,
            font_size=8,
            edge_color='gray',
            alpha=0.7)
    
    plt.title('Author Collaboration Network\n(Node size: number of collaborations, Edge width: number of shared files)')
    
    # Save plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    
    # Calculate network metrics
    metrics = {
        'num_authors': len(G.nodes()),
        'num_connections': len(G.edges()),
        'avg_connections': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
        'density': nx.density(G),
        'connected_components': nx.number_connected_components(G),
        'avg_clustering': nx.average_clustering(G)
    }
    
    return {
        'plot': base64.b64encode(buf.getvalue()).decode('utf-8'),
        'metrics': metrics
    }

def file_contributors_dist(ocel_data):
    """
    Generate a distribution plot showing how many contributors work on each file.
    
    Parameters:
    ocel_data: OCEL converted data
    
    Returns:
    str: Base64 encoded plot URL
    """
    flat = pm4py.ocel_flattening(ocel_data, 'ocel:type:files')
    file_contributors_num = flat.groupby(by='case:concept:name').author_email.nunique()
    
    # Create the distribution plot
    plt.figure(figsize=(10, 6))
    fig = sns.displot(data=file_contributors_num, bins=30)
    plt.title('Distribution of Contributors per File')
    plt.xlabel('Number of Contributors')
    plt.ylabel('Count of Files')
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig.fig)  # Close the figure to free memory
    buf.seek(0)
    
    # Encode the plot
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_url

def analyze_file_activities(objects_summary):
    """
    Analyze file activities and create a ranking of the most edited files.
    
    Parameters:
    objects_summary: DataFrame from pm4py.ocel.ocel_objects_summary
    
    Returns:
    dict: Contains 'top_files' DataFrame and 'plot_url' for visualization
    """
    # Create a new DataFrame with file edits count
    file_analysis = pd.DataFrame({
        'file_name': objects_summary['ocel:oid'],
        'total_edits': objects_summary['activities_lifecycle'].apply(len),
        'unique_activities': objects_summary['activities_lifecycle'].apply(lambda x: len(set(x))),
        'activity_breakdown': objects_summary['activities_lifecycle'].apply(
            lambda x: dict(pd.Series(x).value_counts())
        )
    })
    
    # Sort by total edits
    file_analysis = file_analysis.sort_values('total_edits', ascending=False)

    # Define a threshold for filtering infrequent activities
    threshold = 5  # Minimum occurrences to keep an activity

    # Filter out infrequent activities from the activity breakdown
    file_analysis['activity_breakdown'] = file_analysis['activity_breakdown'].apply(
        lambda x: {k: v for k, v in x.items() if v >= threshold}
    )
    
    # Create a visualization for top 20 files
    plt.figure(figsize=(12, 6))
    top_20 = file_analysis.head(20)
    
    # Create stacked bar chart for top files
    activity_types = sorted(list(set(
        activity 
        for activities in top_20['activity_breakdown'] 
        for activity in activities.keys()
    )))
    
    data = []
    for activity in activity_types:
        data.append([
            activities.get(activity, 0) 
            for activities in top_20['activity_breakdown']
        ])
    
    # Create stacked bar chart
    plt.figure(figsize=(15, 8))
    bottom = np.zeros(len(top_20))
    
    for i, activity_data in enumerate(data):
        plt.bar(range(len(top_20)), activity_data, bottom=bottom, 
                label=activity_types[i])
        bottom += activity_data
    
    plt.title('Activity Breakdown for Top 20 Most Edited Files')
    plt.xlabel('Files')
    plt.ylabel('Number of Activities')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(len(top_20)), top_20['file_name'], rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Prepare the top files summary
    top_files = file_analysis.head(20).copy()
    # Convert activity_breakdown to string representation for easier display
    top_files['activity_breakdown'] = top_files['activity_breakdown'].apply(
        lambda x: ', '.join([f"{k}: {v}" for k, v in x.items()])
    )
    
    return {
        'top_files': top_files.to_dict('records'),
        'plot_url': plot_url
    }

def file_analysis(objects_summary, ocel_data):
    """
    Comprehensive analysis of file-related metrics in the repository.
    
    Parameters:
    df: Original DataFrame with commit data
    objects_summary: DataFrame from pm4py.ocel.ocel_objects_summary
    ocel_data: OCEL converted data
    
    Returns:
    dict: Contains various file analysis metrics and visualizations
    """
    lifecycle_duration_plt = lifecycle_duration_plot(objects_summary)
    file_contributors_dist_ = file_contributors_dist(ocel_data)
    file_activities_analysis = analyze_file_activities(objects_summary)
    
    return {
        'lifecycle_duration_plot': lifecycle_duration_plt,
        'file_contributors_distribution': file_contributors_dist_,
        'top_files': file_activities_analysis['top_files'],
        'activity_breakdown_plot': file_activities_analysis['plot_url']
    }

    
