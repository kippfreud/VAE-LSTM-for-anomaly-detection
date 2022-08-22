"""
This will contain all functions for using boto3 to access iotanalytics,
dynamodb, and RDS

..todo:: maybe this should be a big class "Boto3Manager" to avoid instantiating clients on every function call
"""

# -----------------------------------------------------------------------------

import boto3
import pandas as pd
import json
from sqlalchemy import create_engine
import time

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------
# IoTAnalytics Utils
# -----------------------------------------------------------------------

def lookup_iot_dataset_by_name(name):
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    datasets = iot_client.list_datasets()
    for i in datasets['datasetSummaries']:
        if name.lower() in i['datasetName']:
            return i['datasetName']

def get_secret(secret_name:str):
    """
    Will get a secret stored in the secret manager and return all its values
    """
    secret_client = boto3.client(service_name="secretsmanager")
    secret = json.loads(secret_client.get_secret_value(SecretId=secret_name)["SecretString"])
    return secret

def get_or_string(keys, filter_key):
    """
    Will create a string OR query with given keys and filter keys. e.g. if filter_key = "sensor" and \
    keys=["s00000023", "s00000026"] then (sensor="s00000023" or sensor="s00000026") will be returned.
    """
    if not isinstance(keys, list):
        print("keys must be a list instance!")
        exit(0)
    if len(keys) < 1:
        print("keys must have at least one sensor")
        exit(0)
    ret = "({} = '{}'".format(filter_key, keys[0])
    if len(keys) > 1:
        for i in range(1, len(keys)):
            ret += "OR {}='{}'".format(filter_key, keys[i])
    ret += ")"
    print(ret)
    return ret

def get_full_sql_query(short_sql_query, keys, interval_str=None, filter_key="sensor", order_by_key="ts"):
    """
    Will return a complete version of the given sql query, filled for sensors and interval.

    :param keys: Will create query which filters based on given keys. e.g. if keys = "s00000014" and \
    filter_key="sensor" then we will filter to allow only WHERE sensor="s00000014".
    :param interval_str: If given, will filter by given interval string.
    :param filter_key: If "sensors" will filter by given sensors, if "measure_points", will \
    filter by given measure points.
    :param order_by_key: Will order queried data by the key, descending.
    """
    if interval_str is None:
        ret = "{} WHERE {} ORDER BY {} DESC".format(short_sql_query, get_or_string(keys, filter_key), order_by_key)
    else:
        ret = "{} WHERE {} AND {} ORDER BY {} DESC".format(short_sql_query, get_or_string(keys, filter_key), interval_str, order_by_key)
    return ret

def create_dataset(dataset_name, sql_query, schedule_expression):
    """
    Will create a dataset named dataset_name from given sql_query.

    :param dataset_name: The name of the iotanalytics dataset to create.
    :param sql_query: The full sql query which will define the dataset to create.
    :param schedule_expression: The schedule expression to use to update the dataset - this can \
    be None if no scheduling is required.
    """
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    if not lookup_iot_dataset_by_name(dataset_name):
        if schedule_expression is not None:
            iot_client.create_dataset(datasetName=dataset_name,
                                      actions=[
                                          {
                                              'actionName': 'query',
                                              'queryAction': {
                                                  'sqlQuery': sql_query
                                              }
                                          }],
                                      triggers=[
                                          {
                                              'schedule': {
                                                  'expression': schedule_expression
                                              }
                                          }])
        else:
            iot_client.create_dataset(datasetName=dataset_name,
                                      actions=[
                                          {
                                              'actionName': 'query',
                                              'queryAction': {
                                                  'sqlQuery': sql_query
                                              }
                                          }])
        time.sleep(0.1)
        iot_client.create_dataset_content(datasetName=dataset_name)
        time.sleep(0.1)
    else:
        print('IOT dataset {} already exists'.format(dataset_name))

def update_dataset(dataset_name, sql_query, schedule_expression):
    """
    Will update a dataset from given sql_query.

    :param dataset_name: The name of the iotanalytics dataset to update.
    :param sql_query: The full sql query which will define the dataset to update.
    :param schedule_expression: The schedule expression to use to update the dataset - this can \
    be None if no scheduling is required.
    """
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    if not lookup_iot_dataset_by_name(dataset_name):
        print(f"ERROR: No dataset named '{dataset_name}' exists!")
        exit(0)
    if schedule_expression is not None:
        iot_client.update_dataset(datasetName=dataset_name,
                                      actions=[
                                          {
                                              'actionName': 'query',
                                              'queryAction': {
                                                  'sqlQuery': sql_query
                                              }
                                          }],
                                      triggers=[
                                          {
                                              'schedule': {
                                                  'expression': schedule_expression
                                              }
                                          }])
    else:
        iot_client.update_dataset(datasetName=dataset_name,
                                  actions=[
                                      {
                                          'actionName': 'query',
                                          'queryAction': {
                                              'sqlQuery': sql_query
                                          }
                                      }])
    time.sleep(0.1)
    iot_client.create_dataset_content(datasetName=dataset_name)
    time.sleep(0.1)

def get_iotanalytics_dataset_sql_query(dataset_name):
    """
    Returns the sqlquery associated with a given dataset.
    """
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    response = iot_client.describe_dataset(datasetName=dataset_name)
    return response['dataset']['actions'][0]['queryAction']['sqlQuery']

def get_iotanalytics_dataset_triggers(dataset_name):
    """
    Returns the sqlquery associated with a given dataset.
    """
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    response = iot_client.describe_dataset(datasetName=dataset_name)
    triggers = response['dataset']['triggers']
    if len(triggers) == 0:
        return None
    else:
        return [r['schedule']['expression'] for r in response['dataset']['triggers']]

def get_dataset(dataset_name):
    iot_client = boto3.client('iotanalytics', region_name='us-east-1')
    content = iot_client.get_dataset_content(datasetName=dataset_name, versionId='$LATEST')
    while content["status"]["state"] == "CREATING":
        print("Dataset content is being generated, trying again in 5s...")
        time.sleep(5)
        content = iot_client.get_dataset_content(datasetName=dataset_name, versionId='$LATEST')
    if content["status"]["state"] == "FAILED":
        print(f"Dataset content retrieval failed: '{content['status']['reason']}'")
        exit(0)
    dataset_url = content['entries'][0]['dataURI']
    full_df = pd.read_csv(dataset_url)
    return full_df

def check_dataset_completeness(sql_items, datastore, schedule, dataset_name):
    """
    Will check if all of the items in sql_items are contained in dataset,
    will check that same datastore being used, and will check that scheduling expression
    is the same (or if schedule==None, it will not check).
    """
    dataset_current_sql = get_iotanalytics_dataset_sql_query(dataset_name)
    for sql_item in sql_items:
        if sql_item not in dataset_current_sql:
            return False
    if datastore not in datastore:
        return False
    dataset_current_triggers = get_iotanalytics_dataset_triggers(dataset_name)
    if schedule not in dataset_current_triggers:
        return False
    return True

# -----------------------------------------------------------------------
# DynamoDB Utils
# -----------------------------------------------------------------------

def print_item(table_name, key):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    resp = table.get_item(
        Key=key
    )
    if 'Item' in resp:
        print(resp['Item'])

def get_dynamo_item(table_name, key, var_names):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    ret = table.get_item(
        Key=key,
        AttributesToGet=var_names
    )
    if "Item" not in ret.keys():
        return None
    if ret["Item"] == {}:
        return None
    return ret["Item"]

def update(table_name, key, var_name, var_val):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    table.update_item(
        Key=key,
        UpdateExpression=f"set {var_name} = :g",
        ExpressionAttributeValues={
            ':g': f"{var_val}"
        },
        ReturnValues="UPDATED_NEW"
    )

def delete_item(table_name, kvdict, var_name):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    table.update_item(
        Key=kvdict,
        UpdateExpression=f"REMOVE {var_name}",
        ReturnValues="UPDATED_NEW"
    )

# -----------------------------------------------------------------------
# RDS Utils
# -----------------------------------------------------------------------


def rds_make_connection(secret_id: str):
    """
     Returns database engine after creating the connection to the RDS using AWS secrets

    :params: secret_id: the ID of the desired secret values
    """
    secret_client = boto3.client(service_name="secretsmanager")
    secret = json.loads(secret_client.get_secret_value(SecretId=secret_id)["SecretString"])
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    return db_engine

def get_all_in_use_measuring_points(return_mapping_dict=False):
    """
    Returns a list of all the measuring point IDs that are currently in use

    :param return_mapping_dict: If True, will return a tuple, the list of in use MPs, as well as a dict of MP \
    information.
    """
    secret = get_secret(secret_name="secret-rds")
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    with db_engine.connect() as con:
        sql_query = """
            SELECT
            s.serial_number as serial_number,
                s.measure_point_id as measure_point_id,
                e.serial_number as equipment_serial_number,   
                e.short_description as equipment_name,   
                m.location as location,
                m.threshold_id as threshold_id,
                c.name as name,
                t.vibration_on as vibration_on,
                t.vibration_off as vibration_off,
                t.vib_avg_history as vib_avg_history
            FROM measure_point m
            JOIN sensor s ON m.measure_id = s.measure_point_id
            JOIN threshold t ON m.threshold_id = t.id
            JOIN client c ON s.client_id = c.id
            JOIN equipment e ON e.serial_number = m.equipment_id
            WHERE m.end_date ISNULL
        """
        query = con.execute(sql_query)
        mapping = {}
        for row in query:
            mapping.update({row.measure_point_id: row._asdict()})
    all_in_use_mps = sorted(list(mapping.keys()))
    if return_mapping_dict:
        return all_in_use_mps, mapping
    else:
        return all_in_use_mps

def get_all_measuring_points(return_mapping_dict=False):
    """
    Returns a list of all the measuring point IDs

    :param return_mapping_dict: If True, will return a tuple, the list of in use MPs, as well as a dict of MP \
    information.
    """
    secret = get_secret(secret_name="secret-rds")
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    with db_engine.connect() as con:
        sql_query = """
            SELECT
            s.serial_number as serial_number,
                s.measure_point_id as measure_point_id,
                e.serial_number as equipment_serial_number,   
                e.short_description as equipment_name,   
                m.location as location,
                m.threshold_id as threshold_id,
                c.name as name,
                t.vibration_on as vibration_on,
                t.vibration_off as vibration_off,
                t.vib_avg_history as vib_avg_history
            FROM measure_point m
            JOIN sensor s ON m.measure_id = s.measure_point_id
            JOIN threshold t ON m.threshold_id = t.id
            JOIN client c ON s.client_id = c.id
            JOIN equipment e ON e.serial_number = m.equipment_id
        """
        query = con.execute(sql_query)
        mapping = {}
        for row in query:
            mapping.update({row.measure_point_id: row._asdict()})
    all_in_use_mps = sorted(list(mapping.keys()))
    if return_mapping_dict:
        return all_in_use_mps, mapping
    else:
        return all_in_use_mps

def get_all_in_use_sensors(return_mapping_dict=False):
    """
    Returns a list of all the sensor IDs that are currently in use

    :param return_mapping_dict: If True, will return a tuple, the list of in use sensors, as well as a dict of sensor \
    information.
    """
    secret = get_secret(secret_name="secret-rds")
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    with db_engine.connect() as con:
        sql_query = """
            SELECT
            s.serial_number as serial_number,
                s.measure_point_id as measure_point_id,
                e.serial_number as equipment_serial_number,   
                e.short_description as equipment_name,   
                m.location as location,
                c.name as name,
                t.vibration_on as vibration_on,
                t.vibration_off as vibration_off
            FROM measure_point m
            JOIN sensor s ON m.measure_id = s.measure_point_id
            JOIN threshold t ON m.threshold_id = t.id
            JOIN client c ON s.client_id = c.id
            JOIN equipment e ON e.serial_number = m.equipment_id
            WHERE m.end_date ISNULL
        """
        query = con.execute(sql_query)
        mapping = {}
        for row in query:
            mapping.update({row.serial_number: row._asdict()})
    all_in_use_sensors = sorted(list(mapping.keys()))
    if return_mapping_dict:
        return all_in_use_sensors, mapping
    else:
        return all_in_use_sensors

def get_all_sensors(return_mapping_dict=False):
    """
    Returns a list of all the sensor IDs that have ever collected data.

    :param return_mapping_dict: If True, will return a tuple, the list of in use sensors, as well as a dict of sensor \
    information.
    """
    secret = get_secret(secret_name="secret-rds")
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    with db_engine.connect() as con:
        sql_query = """
            SELECT
            s.serial_number as serial_number,
                s.measure_point_id as measure_point_id,
                e.serial_number as equipment_serial_number,   
                e.short_description as equipment_name,   
                m.location as location,
                c.name as name,
                t.vibration_on as vibration_on,
                t.vibration_off as vibration_off
            FROM measure_point m
            JOIN sensor s ON m.measure_id = s.measure_point_id
            JOIN threshold t ON m.threshold_id = t.id
            JOIN client c ON s.client_id = c.id
            JOIN equipment e ON e.serial_number = m.equipment_id
        """
        query = con.execute(sql_query)
        mapping = {}
        for row in query:
            mapping.update({row.serial_number: row._asdict()})
    all_sensors = sorted(list(mapping.keys()))
    if return_mapping_dict:
        return all_sensors, mapping
    else:
        return all_sensors


def update_rds_db(sql_update_string):
    """
    returns: None, will perform the update
    :param: sql_update_string : str, update syntax is as follows
    f"UPDATE table_name SET column_name={columnName} WHERE someCondition"

    """
    # create the connection to the rds
    db_engine = rds_make_connection(secret_id="secret-rds")

    with db_engine.connect() as con:
        con.execute(sql_update_string)


def get_all_unregistered_sensors(return_mapping_dict=False):
    """
    Returns a list of all the sensor IDs that have no associated measuring points. This should mean they're
    at Motsai.

    :param return_mapping_dict: If True, will return a tuple, the list of in use sensors, as well as a dict of sensor \
    information.
    """
    secret = get_secret(secret_name="secret-rds")
    db_url = "postgresql://{username}:{password}@{host}:5432/{dbname}".format(**secret)
    db_engine = create_engine(db_url, pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=900)
    with db_engine.connect() as con:
        sql_query = """
            SELECT
            s.serial_number as serial_number
            FROM sensor s
            WHERE s.measure_point_id ISNULL
        """
        query = con.execute(sql_query)
        mapping = {}
        for row in query:
            mapping.update({row.serial_number: row._asdict()})
    all_sensors = sorted(list(mapping.keys()))
    if return_mapping_dict:
        return all_sensors, mapping
    else:
        return all_sensors


# -----------------------------------------------------------------------
# Events and Logging Utils
# -----------------------------------------------------------------------

def send_alert_to_eventbridge(alert_type: str, alert_detail: dict, eventbridge_rule:str):
    """
    Will send an alert to eventbridge.

    :param alert_type: Specifies what this alert is associated with, e.g for battery alerts \
    this param should be "battery".
    :param alert_detail: A dictionary containing alert information where keys are items to be \
    alerted about and values are alerts e.g = {"machine_462": "machine 462 has exploded"}
    :param eventbridge_rule: This is used as DetailType in the eventbridge boto3 call
    """
    if "type" in alert_detail.keys():
        print("'type' cannot be a key of alert_detail")
        exit(0)
    alert_detail["type"] = alert_type
    client_events = boto3.client("events", region_name="us-east-1")
    client_events.put_events(
        Entries=[
            {
                "EventBusName": "default",
                "Source": "microservice-user",
                "DetailType": eventbridge_rule,
                "Detail": json.dumps(alert_detail)
            }
        ],
    )

def send_logs_to_cloudwatch(log_group:str, log_stream: str, log: str):
    """
    Will send the given log to the given log_group and log_stream.
    """
    logs = boto3.client('logs')
    # Create log group if it does not already exist
    if log_stream not in [l["logStreamName"] for l in logs.describe_log_streams(logGroupName=log_group, logStreamNamePrefix=log_stream)["logStreams"]]:
        print(f"Creating log stream {log_stream} in log group {log_group}...")
        logs.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
    #Getting last sequence token
    response = logs.describe_log_streams(logGroupName=log_group,
                                         logStreamNamePrefix=log_stream)
    log_event = {
        'logGroupName': log_group,
        'logStreamName': log_stream,
        'logEvents': [
            {
                'timestamp': int(round(time.time() * 1000)),
                'message': log
            },
        ],
    }
    #Adding last sequence token to log event before sending logs if it exists
    if 'uploadSequenceToken' in response['logStreams'][0]:
        log_event.update(
            {'sequenceToken': response['logStreams'][0]['uploadSequenceToken']})
    _ = logs.put_log_events(**log_event)
    time.sleep(1)
