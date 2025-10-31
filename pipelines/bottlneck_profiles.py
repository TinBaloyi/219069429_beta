# bottleneck_profiles.py - FIXED with better column matching

class BottleneckProfile:
    def __init__(self, name, group_by_column, metric_column, mode, description):
        self.name = name
        self.group_by_column = group_by_column
        self.metric_column = metric_column
        self.mode = mode
        self.description = description


manufacturing_profiles = [
    BottleneckProfile(
        name="Resource Constraint (Low Throughput)",
        group_by_column="Machine_ID",
        metric_column="Production_Speed_units_per_hr",
        mode="min",
        description="Detect machines with lowest throughput"
    ),
    BottleneckProfile(
        name="Scheduling Bottleneck (High Error Rate)",
        group_by_column="Machine_ID",
        metric_column="Error_Rate_%",
        mode="max",
        description="Detect machines with highest error rate"
    ),
    BottleneckProfile(
        name="Quality Control Bottleneck",
        group_by_column="Machine_ID",
        metric_column="Quality_Control_Defect_Rate_%",
        mode="max",
        description="Detect machines with highest defect rates"
    ),
    BottleneckProfile(
        name="Power Consumption Anomaly",
        group_by_column="Machine_ID",
        metric_column="Power_Consumption_kW",
        mode="max",
        description="Detect machines with abnormally high power usage"
    ),
]


# FIXED: Logistics profiles now use ACTUAL column names from your CSV
logistics_profiles = [
    # Primary profile - delivery delays (most important!)
    BottleneckProfile(
        name="Route Delay Bottleneck",
        group_by_column="risk_classification",  # This exists in logistics.csv
        metric_column="delivery_time_deviation",  # This exists
        mode="max",
        description="Detect risk categories with highest delivery time deviation"
    ),
    
    # Fuel efficiency issues
    BottleneckProfile(
        name="Route Inefficiency (Fuel)",
        group_by_column="risk_classification",
        metric_column="fuel_consumption_rate",
        mode="max",
        description="Detect categories with unusually high fuel consumption"
    ),
    
    # High delay probability
    BottleneckProfile(
        name="High Delay Probability Routes",
        group_by_column="risk_classification",
        metric_column="delay_probability",
        mode="max",
        description="Detect categories with highest predicted delay probability"
    ),
    
    # Traffic congestion
    BottleneckProfile(
        name="Traffic Congestion Bottleneck",
        group_by_column="risk_classification",
        metric_column="traffic_congestion_level",
        mode="max",
        description="Detect categories with highest traffic congestion"
    ),
]