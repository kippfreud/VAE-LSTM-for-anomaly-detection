from grafana_api.grafana_face import GrafanaFace
import requests
from datetime import datetime, timezone
from utils.boto3_tools import get_secret
import time
import json
import pytz

def get_timestamp(timestamp:str):
    """
    Will get an epoch timestamp in milliseconds from datetime 
    """
    eastern = pytz.timezone("Canada/Eastern")
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    tuple = dt.timetuple()
    return int(dt.replace(tzinfo=eastern).timestamp())*1000

def annotate_dashboard(env:str, client:str, equipment_name:str, measure_id:str, timestamp_from:str, timestamp_to:str, details:str):
    """
    Will create an annotation on the vibration panel for the anomaly period. The function also returns also a link to the dashboard
    """
    secret = get_secret(secret_name="grafana-secret")
    host =f"host_{env}"    
    token =f"token_{env}"
    grafana_api = GrafanaFace(
            auth=(secret["user"],secret["password"]),
            host=secret[host]
    )
    dashboard_list = grafana_api.search.search_dashboards(tag=client.lower())
    dashboard_url = None
    for dash in dashboard_list:
        if equipment_name in dash['title']:
            dashboard_url = 'https://' + secret[host] + f"/d/{dash['uid']}/{client.lower()}-{dash['uri'][3:]}?orgId=1"
            list_measure_points = []
            list_variables = grafana_api.dashboard.get_dashboard(dashboard_uid=dash["uid"])["dashboard"]["templating"]["list"][1]["options"][1:]
            for var in list_variables:
                list_measure_points.append(var["text"])
            index = list_measure_points.index(measure_id[-3:])
            if index == 0:
                panel_id = 65
            elif index ==1:
                panel_id = 190        
            timestamp_from = int(timestamp_from)
            timestamp_to = int(timestamp_to)
            headers = {
                "Accept": "application/json",
                "Content-Type" : "application/json",
                "Authorization": f"Bearer {secret[token]}",
            }
            grafana_url = 'http://' + secret[host]
            payload = {
                "dashboardId": dash["id"],
                "panelId":panel_id,
                "tags":["anomaly", f"{equipment_name}"],
                "text":f"{details}",
                "isRegion":True,
                "time" : timestamp_from,
                "timeEnd" :timestamp_to
            }
            url = grafana_url + "/api/annotations"
            payload_str = json.dumps(payload)
            try:
                response = requests.post(url, data=payload_str, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to post {payload_str} to {url} : {response.text}")
                else:
                    print("Posted annotation..!")
            except requests.exceptions.ConnectionError as exc:
                print(f"failed to post: {exc}")
    return dashboard_url


