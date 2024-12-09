from pydriller import Repository
import pandas as pd

def process_github_repo(repo_url, conventional_commits=False):
    timestamps = []
    hashes = []
    activities = []
    messages = []
    authors_names = []
    authors_emails = []
    merge_yns = []
    files = []
    total_files = set()

    for commit in Repository(repo_url).traverse_commits():
        try:
            # Convert author_date to datetime and append to timestamps
            timestamp = pd.to_datetime(commit.author_date, errors='coerce', utc=True)
            if pd.isnull(timestamp):
                print(f"Warning: Invalid date encountered in commit {commit.hash}")
            timestamps.append(timestamp)
        except Exception as e:
            print(f"Error processing commit {commit.hash}: {e}")
            timestamps.append(pd.NaT)  # Append Not a Time for problematic entries

        hashes.append(commit.hash)

        if conventional_commits:
            activities.append(commit.msg.split(' ')[0].split('(')[0].replace(':', '').lower())

        messages.append(commit.msg.replace(',', ''))
        authors_names.append(commit.author.name)
        authors_emails.append(commit.author.email)
        merge_yns.append(commit.merge)

        try:
            files.append(",".join(str(file.new_path) for file in commit.modified_files))
            total_files.update([str(file.new_path) for file in commit.modified_files])
        except:
            files.append("")

    if conventional_commits:
        df = pd.DataFrame({
            'ocel:timestamp': timestamps, 
            'ocel:eid': hashes, 
            'ocel:activity': activities, 
            'commit_message': messages,
            'ocel:type:author': authors_names, 
            'author_email': authors_emails, 
            'merge': merge_yns,
            'ocel:type:files': files, 
            # 'ocel:type:branches': branches
        })
    else:
        df = pd.DataFrame({
            'ocel:timestamp': timestamps, 
            'ocel:eid': hashes, 
            'ocel:activity': messages, 
            'commit_message': messages, 
            'ocel:type:author': authors_names, 
            'author_email': authors_emails, 
            'merge': merge_yns, 
            'ocel:type:files': files,
            # 'ocel:type:branches': branches
        })

    # Debugging checks
    if df.empty:
        print("DataFrame is empty. No commits found.")
        return df  # or handle the empty case appropriately

    print("DataFrame columns:", df.columns)
    print(f"Processed {len(timestamps)} commits.")

    return df

