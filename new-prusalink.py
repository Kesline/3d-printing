import requests
import json
import time
from requests.auth import HTTPDigestAuth

api_url = "http://192.168.137.160/api/v1/status"
api_key = "2D7239TS679fcbfa" 

headers = {
    # "X-Api-Key": api_key,
    "Authorization": 'Digest username="maker", realm="Printer API", nonce="dca577670000207b",  uri="/api/v1/status", response="9f8054f87daaa4e50f83988b6159e995'
}

def get_printer_data():
    try:
        response = requests.get(api_url, auth=HTTPDigestAuth('maker', 'swNuc3Wv6YHL5ho'))
        response.raise_for_status()  
        data = response.json()  
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def log_data(data, filename="printer_data_log.json"):
    try:
        with open(filename, 'a') as file:
            json.dump(data, file)
            file.write("\n")
        print("Data logged successfully.")
    except IOError as e:
        print(f"Error writing to file: {e}")

# Collect data while printer is printing then automatically stop
def collect_data(interval=5):
    print("Starting data collection...")

    while True:
        data = get_printer_data()
        
        if data:
            state = data["printer"]["state"] # Get printer state

        
        if state == "PRINTING":
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            log_data(data)
        
        elif state in ["PAUSED", "FINISHED"]:
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            log_data(data)
            print(f"Print completed. Printer state: {state}. Stopping data collection.")
            break    # Exit once finished
            
        time.sleep(interval)

collect_data(interval=5)