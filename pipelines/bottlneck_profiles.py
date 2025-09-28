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
]

logistics_profiles = [
    # Route delay (uses Route_Key added during ingestion from GPS bins)
    BottleneckProfile(
        name="Route Delay Bottleneck",
        group_by_column="Route_Key",                   # <-- derived in ingestion
        metric_column="delivery_time_deviation",       # <-- your header
        mode="max",
        description="Detect routes with the highest delivery time deviation"
    ),
    # Vehicle delay (if you have a vehicle identifier column)
    BottleneckProfile(
        name="Vehicle Delay Bottleneck",
        group_by_column="vehicle_id",                  # fallback if you have an id; else keep Route_Key profile
        metric_column="delivery_time_deviation",
        mode="max",
        description="Detect vehicles with consistently slow deliveries"
    ),
    # Vehicle inefficiency (fuel overuse independent of route delay)
    BottleneckProfile(
        name="Vehicle Inefficiency (Fuel)",
        group_by_column="vehicle_id",                  # or Route_Key if no vehicle id
        metric_column="fuel_consumption_rate",
        mode="max",
        description="Detect vehicles with unusually high fuel use"
    ),
]
