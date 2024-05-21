import requests
from bs4 import BeautifulSoup

def download_file_from_google_drive(file_id, output_file):
    initial_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    session = requests.Session()

    # Step 1: Get the warning page
    response = session.get(initial_url)
    print("Initial URL Response Status Code:", response.status_code)

    # Parse the warning page
    soup = BeautifulSoup(response.content, 'html.parser')
    download_form = soup.find('form', {'id': 'download-form'})
    
    if download_form:
        download_url = download_form['action']
        hidden_inputs = download_form.find_all('input', {'type': 'hidden'})
        payload = {input_tag['name']: input_tag['value'] for input_tag in hidden_inputs}
        
        # Construct the final download URL with GET parameters
        download_url_with_params = f"{download_url}?id={payload['id']}&export={payload['export']}&confirm={payload['confirm']}"
        
        # Step 2: Make a GET request to download the file
        download_response = session.get(download_url_with_params, stream=True)
        print("Download URL Response Status Code:", download_response.status_code)
        
        if download_response.status_code == 200:
            # Step 3: Verify the content length
            content_length = download_response.headers.get('Content-Length')
            print("Content-Length:", content_length)

            if content_length and int(content_length) > 0:
                # Step 4: Write the file content to disk
                with open(output_file, 'wb') as file:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                print("File downloaded successfully to:", output_file)
            else:
                print("Failed: Content-Length is zero or undefined.")
        else:
            print("Failed to download file. Status Code:", download_response.status_code)
    else:
        print("Failed to find the download form. Please check the URL or the file permissions.")
