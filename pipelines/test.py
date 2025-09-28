import csv
import datetime

# ---------- Data Ingestion Pipeline Classes ----------

class DataSource:
    def __init__(self, source_id, source_name, source_type, connection_parameters, authentication_credentials, data_format):
        self.source_id = source_id
        self.source_name = source_name
        self.source_type = source_type
        self.connection_parameters = connection_parameters
        self.authentication_credentials = authentication_credentials
        self.data_format = data_format

    def configure(self):
        print(f"[DataSource] Configured for {self.source_name}")

    def validate_connection(self):
        path = self.connection_parameters.get("file_path")
        try:
            with open(path, "r") as f:
                print(f"[DataSource] File {path} accessible.")
                return True
        except:
            print(f"[DataSource] File {path} NOT accessible.")
            return False

    def import_historical_data(self, time_range=None):
        path = self.connection_parameters.get("file_path")
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return HistoricalData(f"hist_{self.source_id}", self.source_id, time_range, rows)


class HistoricalData:
    def __init__(self, data_id, source_id, time_range, raw_data):
        self.data_id = data_id
        self.source_id = source_id
        self.time_range = time_range
        self.raw_data = raw_data


class DataProcessor:
    def __init__(self, cleaningRules=None, transformationRules=None, validationRules=None):
        self.cleaningRules = cleaningRules or []
        self.transformationRules = transformationRules or []
        self.validationRules = validationRules or []

    def clean_and_process_data(self, raw_data):
        cleaned = [row for row in raw_data if all(row.values())]  # drop rows with missing
        return ProcessedData("proc_1", "hist_1", cleaned, cleaned)


class ProcessedData:
    def __init__(self, processed_data_id, historical_data_id, cleaned_data, transformed_data):
        self.processed_data_id = processed_data_id
        self.historical_data_id = historical_data_id
        self.cleaned_data = cleaned_data
        self.transformed_data = transformed_data


class PipelineMonitor:
    def __init__(self):
        self.performanceData = []

    def monitor_performance(self, stage_name, start_time, end_time, records_processed):
        elapsed = (end_time - start_time).total_seconds()
        self.performanceData.append((stage_name, elapsed, records_processed))

    def generate_diagnosis_report(self):
        print("Pipeline Performance Report:")
        for stage, (name, elapsed, records) in enumerate(self.performanceData, start=1):
            print(f"Stage {stage} - {name}: {records} records in {elapsed:.2f}s")


# ---------- Analysis & Detection Pipeline Classes ----------

class AnalysisParameters:
    def __init__(self, analysis_id, dataset_id, analysis_type, group_by_column, metric_column):
        self.analysis_id = analysis_id
        self.dataset_id = dataset_id
        self.analysis_type = analysis_type
        self.group_by_column = group_by_column
        self.metric_column = metric_column


class BottleneckDetector:
    def detect_bottlenecks(self, processed_data, analysis_param):
        groups = {}
        for row in processed_data.cleaned_data:
            try:
                key = row[analysis_param.group_by_column]
                value = float(row[analysis_param.metric_column])
                groups.setdefault(key, []).append(value)
            except Exception:
                continue

        if not groups:
            return []

        # find group with highest average (slowest / bottleneck)
        summary = {k: sum(v)/len(v) for k, v in groups.items()}
        bottleneck_key = max(summary, key=summary.get)
        now = datetime.datetime.now()
        return [BottleneckEvent(bottleneck_key, now, None, None, f"High {analysis_param.metric_column} = {summary[bottleneck_key]}")]


class BottleneckEvent:
    def __init__(self, machine_id, start_time, end_time, duration_minutes, reason):
        self.machine_id = machine_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration_minutes = duration_minutes
        self.reason = reason


# ---------- Pipeline Functions ----------

def run_ingestion_pipeline(file_path, source_type="GENERIC"):
    """Runs Data Ingestion Pipeline dynamically for any CSV file."""
    monitor = PipelineMonitor()
    start = datetime.datetime.now()

    # Step 1: Data source
    ds = DataSource("1", "CSVSource", source_type, {"file_path": file_path}, {}, "csv")
    if not ds.validate_connection():
        return None
    hist = ds.import_historical_data()

    # Step 2: Processing
    dp = DataProcessor()
    proc = dp.clean_and_process_data(hist.raw_data)

    end = datetime.datetime.now()
    monitor.monitor_performance("Ingestion", start, end, len(proc.cleaned_data))
    monitor.generate_diagnosis_report()

    return proc


def run_analysis_pipeline(processed_data, group_by, metric, analysis_type="GENERIC"):
    """Runs Analysis & Detection pipeline."""
    params = AnalysisParameters("a1", processed_data.processed_data_id, analysis_type, group_by, metric)
    detector = BottleneckDetector()
    events = detector.detect_bottlenecks(processed_data, params)

    if not events:
        print("[Analysis] No bottlenecks detected.")
    else:
        for e in events:
            print(f"[Bottleneck Detected] {e.machine_id} - {e.reason} at {e.start_time}")

    return events


# ---------- Example Usage ----------

if __name__ == "__main__":
    # Example: Manufacturing dataset
    proc = run_ingestion_pipeline("manufacturing.csv", "MANUFACTURING")
    if proc:
        run_analysis_pipeline(proc, group_by="Machine_ID", metric="Production_Speed_units_per_hr", analysis_type="MANUFACTURING")

    # Example: Logistics dataset
    proc2 = run_ingestion_pipeline("logistics.csv", "LOGISTICS")
    if proc2:
        run_analysis_pipeline(proc2, group_by="Area", metric="Delivery_Time", analysis_type="LOGISTICS")
