import markdown
import os

https://github.com/docker/docs/tree/main/content/get-started

def download_files_from_github(repo_name):
    repo_name = "https://github.com/docker/"  # Replace "main" with the branch name if different
    
    os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist
    
    for file_path in file_paths:
        url = f"{base_url}/{file_path}"
        response = requests.get(url)
        if response.status_code == 200:
            local_file_path = os.path.join(save_dir, os.path.basename(file_path))
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {file_path} -> {local_file_path}")
        else:
            print(f"Failed to download {file_path}: HTTP {response.status_code}")