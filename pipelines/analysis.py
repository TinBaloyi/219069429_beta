


class AnalysisParameters:
    def __init__(self, analysis_id, dataset_id, analysis_type, group_by_column, metric_column, sensitivity_threshold=0.8, time_window="24h", modelSelection=None):
        self.analysis_id = analysis_id
        self.dataset_id = dataset_id
        self.analysis_type = analysis_type  # "MANUFACTURING" or "LOGISTICS"
        self.group_by_column = group_by_column
        self.metric_column = metric_column
        self.sensitivity_threshold = sensitivity_threshold
        self.time_window = time_window
        self.modelSelection = modelSelection or []
        

    def validateConfig(self):
        # Validate analysis parameters
        return self.analysis_type in ["MANUFACTURING", "LOGISTICS"]

    def selectModels(self):
        print(f"Selecting models: {self.modelSelection}")
        return self.modelSelection


class BottleneckDetector:
    def __init__(self, ml_model=None, detection_algorithms=None):
        self.ml_model = ml_model
        self.detection_algorithms = detection_algorithms or []

    def detect_bottlenecks(self, processed_data, analysis_param):
        key_column = analysis_param.group_by_column
        metric_column = analysis_param.metric_column
        mode = getattr(analysis_param, 'mode', 'min')  # 'max'|'min'|'avg'

        # 1) collect values by group
        group_metrics = {}
        for row in processed_data:
            key = row.get(key_column)
            if not key:
                continue
            try:
                val = float(str(row.get(metric_column, "")).replace("%",""))
            except Exception:
                val = None
            if val is None:
                continue
            group_metrics.setdefault(key, []).append(val)

        if not group_metrics:
            return []

        # 2) aggregate per group
        if mode == "min":
            summary = {k: min(v) for k, v in group_metrics.items()}
            target_value = min(summary.values())
        elif mode == "max":
            summary = {k: max(v) for k, v in group_metrics.items()}
            target_value = max(summary.values())
        else:  # avg
            summary = {k: (sum(v)/len(v)) for k, v in group_metrics.items()}
            target_value = max(summary.values())

        # 3) severity + confidence
        from statistics import pstdev, mean
        from datetime import datetime
        all_vals = list(summary.values())
        vmin, vmax = min(all_vals), max(all_vals)
        vrng = (vmax - vmin) or 1e-9
        mu, sd = mean(all_vals), (pstdev(all_vals) or 1e-9)

        now = datetime.utcnow()
        events = []
        for k, v in summary.items():
            sev = 100.0 * ((v - vmin) / vrng) if mode != "min" else 100.0 * ((vmax - v) / vrng)
            z = abs((v - mu) / sd)
            conf = max(0.0, min(1.0, z / 3.0))
            if v == target_value:
                ev = BottleneckEvent(
                    entity_id=k, start_time=now, end_time=None, duration_minutes=None,
                    reason=f"{mode} {metric_column} = {v:.3f}"
                )
                ev.metric_value = v
                ev.severity_score = sev
                ev.confidence_score = conf
                events.append(ev)
        return events


    
    def analyzePatterns(self, data):
        # Placeholder for future
        return {}

    def generateInsights(self, events):
        insights = [f"{e.entity_id} is a bottleneck: {e.reason}" for e in events]
        return insights


class BottleneckEvent:
    def __init__(self, entity_id, start_time, end_time, duration_minutes, reason):
        self.entity_id = entity_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration_minutes = duration_minutes
        self.reason = reason


class BottleneckPredictor:
    def __init__(self, modelType, trainedModel, features, accuracy):
        self.modelType = modelType
        self.trainedModel = trainedModel
        self.features = features
        self.accuracy = accuracy

    def trainModel(self, train_data, labels):
        print(f"Training {self.modelType} model... (placeholder)")

    def predictBottlenecks(self, test_data):
        # Placeholder for predictions
        return [PredictionEvent(entity_id="entity_1", predicted_occurrence_time=datetime.now(), likelihood=0.9, potential_impact="High")]

    def evaluatePerformance(self, test_data, true_labels):
        print("Evaluating performance... (placeholder)")
        return {"accuracy": self.accuracy}

    def updateModel(self, new_data):
        print("Updating model... (placeholder)")
        return True


class PredictionEvent:
    def __init__(self, entity_id, predicted_occurrence_time, likelihood, potential_impact):
        self.entity_id = entity_id
        self.predicted_occurrence_time = predicted_occurrence_time
        self.likelihood = likelihood
        self.potential_impact = potential_impact
