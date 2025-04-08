import requests
import json
import time
from requests.auth import HTTPDigestAuth

# API endpoints
api_url = "http://192.168.137.137/api/v1/status"
job_url = "http://192.168.137.137/api/v1/job"

headers = {
    # "X-Api-Key": api_key,
    "Authorization": 'Digest username="maker", realm="Printer API", nonce="dca577670000207b",  uri="/api/v1/status", response="9f8054f87daaa4e50f83988b6159e995'
}

# Auth
auth = HTTPDigestAuth('maker', 'swNuc3Wv6YHL5ho')

def get_printer_data():
    try:
        response = requests.get(api_url, auth=auth)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Status API Error]: {e}")
        return None

def get_job_details():
    try:
        response = requests.get(job_url, auth=auth) 
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Job API Error]: {e}")
        return {}

def log_data(data, filename="printer_data_log.json"):
    try:
        with open(filename, 'a') as file:
            json.dump(data, file)
            file.write("\n")
        print("Data logged successfully.")
    except IOError as e:
        print(f"Error writing to file: {e}")

# Collect data while printer is printing then automatically stop
def collect_data(interval=5, idle_timeout=60):
    print("Starting continuous data collection with  integrated timeout...")

    idle_start_time = None

    while True:
        data = get_printer_data()

        if data:
            state = data["printer"]["state"]
             # Print timestamp
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            # Get file info from separate /job endpoint
            job_details = get_job_details()
            file_info = job_details.get("file", {})
            data['filename'] = file_info.get("display_name", file_info.get("name", "unknown_file"))
            # Extract shape name
            raw_name = file_info.get("display_name", "")
            data['shape'] = raw_name.split("_")[0] if raw_name else "unknown_shape"
        
        if state == "PRINTING":
            idle_start_time = None
           
            log_data(data)

        elif state in ["PAUSED", "FINISHED", "IDLE"]:
            log_data(data)
            print(f"Printer state: {state}. Waiting for next print")
            if idle_start_time is None:
                idle_start_time = time.time()
            elif time.time() - idle_start_time > idle_timeout:
                print("Printer has been idle too long. Exiting...")
                break

        time.sleep(interval)

collect_data(interval=5)
