import csv
import json
import os
from datetime import datetime


def _parse_dt(s):
    from datetime import datetime
    if s is None: return None
    if isinstance(s, datetime): return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(s, fmt)
        except: pass
    try: return datetime.fromisoformat(s)
    except: return None

class DataSource:
    def __init__(self, source_id, source_name, source_type, connection_parameters,
                 authentication_credentials, data_format, file_path=None):
        self.source_id = source_id
        self.source_name = source_name
        self.source_type = source_type            # "MANUFACTURING", "LOGISTICS", etc.
        self.connection_parameters = connection_parameters  # dict
        self.authentication_credentials = authentication_credentials  # dict
        self.data_format = data_format            # "csv", "json", "database"
        self.file_path = file_path

    def configure(self, **kwargs) -> None:
        # Set or update parameters at runtime
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validate_connection(self):
        if self.data_format == "csv" and self.file_path and os.path.isfile(self.file_path):
            print(f"[DataSource] File '{self.file_path}' is accessible.")
            return True
        if self.data_format == "database":
            try:
                import psycopg2
                conn = psycopg2.connect(
                    dbname=self.connection_parameters.get("dbname"),
                    user=self.authentication_credentials.get("user"),
                    password=self.authentication_credentials.get("password"),
                    host=self.connection_parameters.get("host"),
                    port=self.connection_parameters.get("port"),
                )
                conn.close()
                print("[DataSource] Database connection successful.")
                return True
            except Exception as e:
                print(f"[DataSource] Database connection failed: {e}")
                return False
        print("[DataSource] Connection could not be validated.")
        return False


    def import_historical_data(self, time_column=None, time_range=None):
        data = []
        if self.data_format == "csv":
            with open(self.file_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
        elif self.data_format == "json":
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
        elif self.data_format == "database":
        # (left as stub)
            return []
        else:
            raise ValueError("Unsupported data format")

        # determine timestamp column
        ts_col = time_column or ("Timestamp" if self.source_type == "MANUFACTURING" else "timestamp")

        # filter by time (robust datetime parsing)
        t0 = t1 = None
        if time_range:
            t0, t1 = _parse_dt(time_range[0]), _parse_dt(time_range[1])

        out = []
        for row in data:
            ts = _parse_dt(row.get(ts_col))
            if not ts:
                continue
            if t0 and ts < t0: 
                continue
            if t1 and ts > t1:
                continue

         # normalize
            row[ts_col] = ts.strftime("%Y-%m-%dT%H:%M:%S")
            row["Day"] = ts.strftime("%Y-%m-%d")

            # logistics enrichment: build Route_Key from GPS bins
            if self.source_type == "LOGISTICS":
                lat = row.get("vehicle_gps_latitude"); lon = row.get("vehicle_gps_longitude")
                try:
                    latb = round(float(lat), 1) if lat not in (None, "") else None
                    lonb = round(float(lon), 1) if lon not in (None, "") else None
                except Exception:
                    latb = lonb = None
                if latb is not None and lonb is not None:
                    row["Route_Key"] = f"{latb:.1f},{lonb:.1f}"

            out.append(row)

        return out
 
class PipelineMonitor:
    def __init__(self):
        self.metrics = {}
        self.alertThreshold = {}
        self.performanceData = []

    def monitor_performance(self, stage_name, start_time, end_time, records_processed):
        elapsed = (end_time - start_time).total_seconds()
        self.metrics[stage_name] = {"time": elapsed, "records": records_processed}
        self.performanceData.append((stage_name, elapsed, records_processed))

    def generate_diagnosis_report(self):
        print("Pipeline Performance Report:")
        for stage, metric in self.metrics.items():
            print(f"{stage} - Time: {metric['time']}s, Records: {metric['records']}")

class HistoricalData:
    def __init__(self, data_id, source_id, time_range, raw_data):
        self.data_id = data_id
        self.source_id = source_id
        self.time_range = time_range
        self.raw_data = raw_data

#DataProcessor
class DataProcessor:
    def __init__(self, cleaningRules=None, transformationRules=None, validationRules=None):
        self.cleaningRules = cleaningRules or []
        self.transformationRules = transformationRules or []
        self.validationRules = validationRules or []

    def clean_and_process_data(self, raw_data, cleaning_columns=None):
        # Remove rows with missing data in any of the specified columns
        cleaned = []
        for row in raw_data:
            if cleaning_columns:
                if all(row.get(col, None) not in [None, "", "NaN"] for col in cleaning_columns):
                    cleaned.append(row)
            else:
                if all(v not in [None, "", "NaN"] for v in row.values()):
                    cleaned.append(row)
        return cleaned

    def transform_data(self, data, transformations=None):
        # Apply transformation functions
        if not transformations:
            return data
        transformed = []
        for row in data:
            new_row = row.copy()
            for col, func in transformations.items():
                if col in row:
                    try:
                        new_row[col] = func(row[col])
                    except Exception:
                        pass
            transformed.append(new_row)
        return transformed

    def validateQuality(self, data):
        
        num_rows = len(data)
        num_complete = sum(1 for row in data if all(row.values()))
        return {"completeness": num_complete / num_rows if num_rows else 0}

    def generateReport(self, data):
        return f"Processed {len(data)} records, {self.validateQuality(data)['completeness']*100:.2f}% complete"

# ProcessedData 
class ProcessedData:
    def __init__(self, processed_data_id, historical_data_id, cleaned_data, transformed_data):
        self.processed_data_id = processed_data_id
        self.historical_data_id = historical_data_id
        self.cleaned_data = cleaned_data
        self.transformed_data = transformed_data
