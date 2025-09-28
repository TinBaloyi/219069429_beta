#!/usr/bin/env python3
# Minimal ML for BOTH Manufacturing and Logistics (stdlib only)
# Algorithms: handcrafted StandardScaler, LogisticRegression (OVR), RandomForest

from __future__ import annotations
import argparse, csv, json, math, os, random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Union

# -----------------------------
# Math helpers (stdlib only)
# -----------------------------
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

# -----------------------------
# StandardScaler
# -----------------------------
@dataclass
class StandardScaler:
    mean: List[float]
    std: List[float]

    @staticmethod
    def fit(X: List[List[float]]) -> "StandardScaler":
        if not X: return StandardScaler([], [])
        n, m = len(X), len(X[0])
        mean = [0.0]*m
        for row in X:
            for j in range(m): mean[j] += row[j]
        mean = [v/n for v in mean]
        var = [0.0]*m
        for row in X:
            for j in range(m):
                d = row[j]-mean[j]
                var[j] += d*d
        var = [v/max(1,n-1) for v in var]
        std = [math.sqrt(v) if v>1e-12 else 1.0 for v in var]
        return StandardScaler(mean, std)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        Z = []
        for row in X:
            Z.append([(x-m)/s for x,m,s in zip(row, self.mean, self.std)])
        return Z

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StandardScaler":
        return StandardScaler(list(d["mean"]), list(d["std"]))

# -----------------------------
# Logistic Regression (OVR)
# -----------------------------
@dataclass
class LogisticRegressionOVR:
    classes_: List[int]
    weights_: List[List[float]]  # per class: bias + weights
    reg_lambda: float
    lr: float
    epochs: int

    @staticmethod
    def fit(X: List[List[float]], y: List[int], reg_lambda=1e-3, lr=0.1, epochs=200) -> "LogisticRegressionOVR":
        classes = sorted(set(y))
        m = len(X[0]) if X else 0
        N = max(1,len(X))
        weights: List[List[float]] = []
        for c in classes:
            w = [0.0]*(m+1)
            for _ in range(epochs):
                grad = [0.0]*(m+1)
                for xi, yi in zip(X, y):
                    yi_bin = 1.0 if yi == c else 0.0
                    z = w[0] + dot(w[1:], xi)
                    p = sigmoid(z)
                    err = p - yi_bin
                    grad[0] += err
                    for j in range(m): grad[j+1] += err*xi[j]
                for j in range(1,m+1): grad[j] += reg_lambda*w[j]
                for j in range(m+1): w[j] -= lr*grad[j]/N
            weights.append(w)
        return LogisticRegressionOVR(classes, weights, reg_lambda, lr, epochs)

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        k = max(1,len(self.weights_))
        for xi in X:
            scores = []
            for w in self.weights_:
                z = w[0] + dot(w[1:], xi)
                scores.append(sigmoid(z))
            s = sum(scores)
            probs = [v/s if s>0 else 1.0/k for v in scores]
            out.append(probs)
        return out

    def predict(self, X: List[List[float]]) -> List[int]:
        probs = self.predict_proba(X)
        idxs = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
        return [self.classes_[i] for i in idxs]

    def to_dict(self) -> Dict[str, Any]:
        return {"type":"logreg","classes":self.classes_,"weights":self.weights_,
                "reg_lambda":self.reg_lambda,"lr":self.lr,"epochs":self.epochs}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LogisticRegressionOVR":
        return LogisticRegressionOVR(list(d["classes"]),
                                     [list(w) for w in d["weights"]],
                                     float(d.get("reg_lambda",1e-3)),
                                     float(d.get("lr",0.1)),
                                     int(d.get("epochs",200)))

# -----------------------------
# Random Forest (stdlib)
# -----------------------------
@dataclass
class _TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["__class__"] = None
    right: Optional["__class__"] = None
    probs: Optional[List[float]] = None
    def is_leaf(self): return self.probs is not None
    def to_dict(self): 
        return {"leaf": True, "probs": self.probs} if self.is_leaf() else {
            "leaf": False, "f": self.feature, "t": self.threshold,
            "l": self.left.to_dict() if self.left else None,
            "r": self.right.to_dict() if self.right else None
        }
    @staticmethod
    def from_dict(d):
        if d.get("leaf"): return _TreeNode(probs=[float(x) for x in d["probs"]])
        node = _TreeNode(feature=int(d["f"]), threshold=float(d["t"]))
        node.left = _TreeNode.from_dict(d["l"]) if d.get("l") else None
        node.right = _TreeNode.from_dict(d["r"]) if d.get("r") else None
        return node

@dataclass
class RandomForestClassifier:
    n_trees: int = 100
    max_depth: int = 12
    min_samples_split: int = 8
    max_features: Union[int,str] = "sqrt"
    bootstrap: bool = True
    random_state: int = 42
    classes_: List[int] = None  # type: ignore
    trees_: List[_TreeNode] = None  # type: ignore
    feature_importances_: List[float] = None  # type: ignore

    @staticmethod
    def _gini(counts: Dict[int,int]) -> float:
        n = sum(counts.values()) or 1
        return 1.0 - sum((c/n)**2 for c in counts.values())

    def _counts(self, y: List[int]) -> Dict[int,int]:
        cc = {c:0 for c in self.classes_}
        for v in y: cc[v] = cc.get(v,0)+1
        return cc

    def _leaf_probs(self, y: List[int]) -> List[float]:
        c = self._counts(y); n = sum(c.values()) or 1
        return [c[k]/n for k in self.classes_]

    def _best_split(self, X: List[List[float]], y: List[int], feat_idxes: List[int]) -> Tuple[Optional[int],Optional[float],float,float]:
        best_f, best_t, best_imp = None, None, 1e9
        parent_imp = self._gini(self._counts(y))
        n = len(X)
        for j in feat_idxes:
            vals = sorted(set(row[j] for row in X))
            if len(vals) <= 1: continue
            step = max(1, len(vals)//20)
            for idx in range(step, len(vals), step):
                thr = (vals[idx-1]+vals[idx])/2.0
                left_y, right_y = [], []
                for xi, yi in zip(X, y):
                    (left_y if xi[j] <= thr else right_y).append(yi)
                if not left_y or not right_y: continue
                n_l, n_r = len(left_y), len(right_y)
                g = (n_l/n)*self._gini(self._counts(left_y)) + (n_r/n)*self._gini(self._counts(right_y))
                if g < best_imp:
                    best_imp, best_f, best_t = g, j, thr
        return best_f, best_t, best_imp, parent_imp

    def _fit_tree(self, X: List[List[float]], y: List[int], depth: int, rng: random.Random, feat_imp_acc: List[float]) -> _TreeNode:
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            return _TreeNode(probs=self._leaf_probs(y))
        m = len(X[0])
        if isinstance(self.max_features,int):
            k = max(1, min(self.max_features, m))
        else:
            k = max(1, int(math.sqrt(m)) if self.max_features=="sqrt" else (int(math.log2(m)) if self.max_features=="log2" else m))
        feat_idxes = rng.sample(list(range(m)), k)
        f,t,child_imp,parent_imp = self._best_split(X,y,feat_idxes)
        if f is None: return _TreeNode(probs=self._leaf_probs(y))
        X_l, y_l, X_r, y_r = [], [], [], []
        for xi, yi in zip(X,y):
            (X_l if xi[f] <= t else X_r).append(xi)
            (y_l if xi[f] <= t else y_r).append(yi)
        if not X_l or not X_r: return _TreeNode(probs=self._leaf_probs(y))
        # feature importance
        n = len(y)
        imp_after = (len(y_l)/n)*self._gini(self._counts(y_l)) + (len(y_r)/n)*self._gini(self._counts(y_r))
        feat_imp_acc[f] += max(0.0, parent_imp - imp_after)
        node = _TreeNode(feature=f, threshold=t)
        node.left = self._fit_tree(X_l,y_l,depth+1,rng,feat_imp_acc)
        node.right = self._fit_tree(X_r,y_r,depth+1,rng,feat_imp_acc)
        return node

    @staticmethod
    def _bootstrap(X: List[List[float]], y: List[int], rng: random.Random):
        n = len(X)
        idxs = [rng.randrange(n) for _ in range(n)] if n else []
        return [X[i] for i in idxs], [y[i] for i in idxs]

    def fit(self, X: List[List[float]], y: List[int]) -> "RandomForestClassifier":
        self.classes_ = sorted(set(y)) if y else [0,1]
        self.trees_ = []
        m = len(X[0]) if X else 0
        feat_imp_total = [0.0]*m
        rng = random.Random(self.random_state)
        for _ in range(self.n_trees):
            Xb, yb = self._bootstrap(X,y,rng) if self.bootstrap else (X[:], y[:])
            tree_imp = [0.0]*m
            tree = self._fit_tree(Xb,yb,0,rng,tree_imp)
            self.trees_.append(tree)
            for j in range(m): feat_imp_total[j] += tree_imp[j]
        s = sum(feat_imp_total) or 1.0
        self.feature_importances_ = [v/s for v in feat_imp_total]
        return self

    def _predict_tree_proba(self, node: _TreeNode, x: List[float]) -> List[float]:
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right  # type: ignore
        return node.probs or [1.0/len(self.classes_)]*len(self.classes_)

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        k = len(self.classes_) or 2
        out: List[List[float]] = []
        T = max(1, len(self.trees_ or []))
        for xi in X:
            agg = [0.0]*k
            for t in self.trees_ or []:
                p = self._predict_tree_proba(t, xi)
                for i in range(k): agg[i] += p[i]
            out.append([v/T for v in agg] if T>0 else [1.0/k]*k)
        return out

    def predict(self, X: List[List[float]]) -> List[int]:
        probs = self.predict_proba(X)
        idxs = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
        return [self.classes_[i] for i in idxs]

    def to_dict(self) -> Dict[str, Any]:
        return {"type":"rf","n_trees":self.n_trees,"max_depth":self.max_depth,
                "min_samples_split":self.min_samples_split,"max_features":self.max_features,
                "bootstrap":self.bootstrap,"random_state":self.random_state,
                "classes":self.classes_,"trees":[t.to_dict() for t in (self.trees_ or [])],
                "feature_importances_":self.feature_importances_}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RandomForestClassifier":
        rf = RandomForestClassifier(n_trees=int(d.get("n_trees",100)),
                                    max_depth=int(d.get("max_depth",12)),
                                    min_samples_split=int(d.get("min_samples_split",8)),
                                    max_features=d.get("max_features","sqrt"),
                                    bootstrap=bool(d.get("bootstrap",True)),
                                    random_state=int(d.get("random_state",42)))
        rf.classes_ = [int(c) for c in d["classes"]]
        rf.trees_ = [_TreeNode.from_dict(td) for td in d["trees"]]
        rf.feature_importances_ = list(d.get("feature_importances_", []))
        return rf

# -----------------------------
# Feature engineering & labels
# -----------------------------
STATUS_MAP = {None:0, "LOW":0, "MEDIUM":1, "HIGH":2}
MODE_MAP   = {None:0, "OFF":0, "IDLE":1, "MAINTENANCE":2, "ACTIVE":3}

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x,str) and x.strip()==""): return default
        s = str(x).strip()
        if s.endswith("%"): s = s[:-1]
        return float(s)
    except Exception:
        return default

# Manufacturing extractor (your exact headers)
class FeatureExtractorManufacturing:
    feature_names = [
        "speed","defect","error","temp","vibration","power",
        "latency","packet_loss","pm_score","eff_status","op_mode"
    ]
    @staticmethod
    def row_to_features(r: Dict[str, Any]) -> List[float]:
        return [
            _to_float(r.get("Production_Speed_units_per_hr")),
            _to_float(r.get("Quality_Control_Defect_Rate_%")),
            _to_float(r.get("Error_Rate_%")),
            _to_float(r.get("Temperature_C")),
            _to_float(r.get("Vibration_Hz")),
            _to_float(r.get("Power_Consumption_kW")),
            _to_float(r.get("Network_Latency_ms")),
            _to_float(r.get("Packet_Loss_%")),
            _to_float(r.get("Predictive_Maintenance_Score")),
            float(STATUS_MAP.get(r.get("Efficiency_Status"),0)),
            float(MODE_MAP.get(r.get("Operation_Mode"),0)),
        ]
    @staticmethod
    def weak_label(r: Dict[str, Any],
                   speed_min=200.0, defect_hi=5.0, err_hi=2.0,
                   temp_hi=75.0, vib_hi=120.0, net_lat_hi=150.0, pkt_hi=2.0) -> int:
        speed = _to_float(r.get("Production_Speed_units_per_hr"))
        stress = (
            _to_float(r.get("Quality_Control_Defect_Rate_%"))>=defect_hi or
            _to_float(r.get("Error_Rate_%"))>=err_hi or
            _to_float(r.get("Temperature_C"))>=temp_hi or
            _to_float(r.get("Vibration_Hz"))>=vib_hi or
            _to_float(r.get("Network_Latency_ms"))>=net_lat_hi or
            _to_float(r.get("Packet_Loss_%"))>=pkt_hi or
            STATUS_MAP.get(r.get("Efficiency_Status"),0) >= 2
        )
        return 1 if (speed < speed_min and stress) else 0

# Logistics extractor (your exact headers)
class FeatureExtractorLogistics:
    feature_names = [
        "delay_dev","eta_var","traffic","port_cong","customs","load_unload",
        "equip_avail","weather","inv_level","ship_cost","fuel","driver_score",
        "fatigue","route_risk","supplier_rel","lead_time","hist_demand",
        "iot_temp","delay_prob","disrupt_like","risk_class","lat_bin","lon_bin"
    ]
    RISK_CLASS = {"LOW":0,"MEDIUM":1,"HIGH":2}
    @staticmethod
    def _bin(v): 
        try: return round(float(v), 1)
        except: return 0.0
    @staticmethod
    def _risk(s):
        if s is None: return 1.0
        return float(FeatureExtractorLogistics.RISK_CLASS.get(str(s).strip().upper(),1))
    @staticmethod
    def row_to_features(r: Dict[str, Any]) -> List[float]:
        delay = _to_float(r.get("delivery_time_deviation"))
        eta   = _to_float(r.get("eta_variation_hours"))
        traffic = _to_float(r.get("traffic_congestion_level"))
        portc   = _to_float(r.get("port_congestion_level"))
        customs = _to_float(r.get("customs_clearance_time"))
        load    = _to_float(r.get("loading_unloading_time"))
        equip   = _to_float(r.get("handling_equipment_availability"))
        weather = _to_float(r.get("weather_condition_severity"))
        inv     = _to_float(r.get("warehouse_inventory_level"))
        ship    = _to_float(r.get("shipping_costs"))
        fuel    = _to_float(r.get("fuel_consumption_rate"))
        driver  = _to_float(r.get("driver_behavior_score"))
        fatigue = _to_float(r.get("fatigue_monitoring_score"))
        route_risk = _to_float(r.get("route_risk_level"))
        supplier   = _to_float(r.get("supplier_reliability_score"))
        lead    = _to_float(r.get("lead_time_days"))
        demand  = _to_float(r.get("historical_demand"))
        temp    = _to_float(r.get("iot_temperature"))
        dprob   = _to_float(r.get("delay_probability"))
        disrupt = _to_float(r.get("disruption_likelihood_score"))
        risk_cls = FeatureExtractorLogistics._risk(r.get("risk_classification"))
        latb = FeatureExtractorLogistics._bin(r.get("vehicle_gps_latitude"))
        lonb = FeatureExtractorLogistics._bin(r.get("vehicle_gps_longitude"))
        return [delay,eta,traffic,portc,customs,load,equip,weather,inv,ship,
                fuel,driver,fatigue,route_risk,supplier,lead,demand,temp,
                dprob,disrupt,risk_cls,latb,lonb]
    # weak labels for tasks
    @staticmethod
    def weak_label_delay(r: Dict[str, Any], delay_hi=2.0, eta_hi=1.0) -> int:
        return 1 if (_to_float(r.get("delivery_time_deviation"))>=delay_hi or
                     _to_float(r.get("eta_variation_hours"))>=eta_hi) else 0
    @staticmethod
    def weak_label_inefficiency(r: Dict[str, Any], fuel_hi=15.0, delay_ok=1.0) -> int:
        return 1 if (_to_float(r.get("fuel_consumption_rate"))>=fuel_hi and
                     _to_float(r.get("delivery_time_deviation"))<=delay_ok) else 0

# -----------------------------
# IO helpers
# -----------------------------
def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# -----------------------------
# Model bundle (JSON)
# -----------------------------
@dataclass
class TrainedModelBundle:
    scaler: StandardScaler
    model_type: str  # "rf" | "logreg"
    model: Union[RandomForestClassifier, LogisticRegressionOVR]
    features: List[str]
    classes: List[int]
    domain: str
    task: str

    def save(self, path: str) -> None:
        write_json(path, {
            "scaler": self.scaler.to_dict(),
            "model_type": self.model_type,
            "model": self.model.to_dict(),
            "features": self.features,
            "classes": self.classes,
            "domain": self.domain,
            "task": self.task,
        })

    @staticmethod
    def load(path: str) -> "TrainedModelBundle":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        scaler = StandardScaler.from_dict(d["scaler"])
        mtype = d["model_type"]
        if mtype == "rf":
            model = RandomForestClassifier.from_dict(d["model"])  # type: ignore
        else:
            model = LogisticRegressionOVR.from_dict(d["model"])  # type: ignore
        return TrainedModelBundle(
            scaler, mtype, model,
            list(d.get("features", [])),
            list(d.get("classes", [0,1])),
            d.get("domain","manufacturing"),
            d.get("task","bottleneck"),
        )

# -----------------------------
# Training / prediction
# -----------------------------
def _metrics(y_true: List[int], y_pred: List[int]) -> Dict[str,float]:
    tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==0)
    acc = (tp+tn)/max(1,len(y_true))
    prec = tp/max(1,tp+fp)
    rec = tp/max(1,tp+fn)
    f1 = 2*prec*rec/max(1e-12,prec+rec) if (prec+rec)>0 else 0.0
    return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4)}

def _extract(domain: str, rows: List[Dict[str,Any]], task: str) -> Tuple[List[List[float]], List[int], List[str]]:
    X, y = [], []
    if domain == "manufacturing":
        for r in rows:
            X.append(FeatureExtractorManufacturing.row_to_features(r))
            y.append(FeatureExtractorManufacturing.weak_label(r))
        feats = FeatureExtractorManufacturing.feature_names
    else:
        for r in rows:
            X.append(FeatureExtractorLogistics.row_to_features(r))
            if task == "inefficiency":
                y.append(FeatureExtractorLogistics.weak_label_inefficiency(r))
            else:
                y.append(FeatureExtractorLogistics.weak_label_delay(r))
        feats = FeatureExtractorLogistics.feature_names
    return X, y, feats

def train(domain: str, rows: List[Dict[str,Any]], task: str, algo: str) -> Tuple[TrainedModelBundle, Dict[str,Any]]:
    X, y, feats = _extract(domain, rows, task)
    scaler = StandardScaler.fit(X)
    Xs = scaler.transform(X)
    if algo == "logreg":
        model = LogisticRegressionOVR.fit(Xs, y, reg_lambda=1e-3, lr=0.1, epochs=200)
    else:
        model = RandomForestClassifier(n_trees=100, max_depth=12, min_samples_split=8,
                                       max_features="sqrt", bootstrap=True, random_state=42).fit(Xs, y)
    # in-sample metrics (simple)
    if isinstance(model, RandomForestClassifier):
        y_pred = model.predict(Xs)
        classes = model.classes_
    else:
        y_pred = model.predict(Xs)
        classes = model.classes_
    metrics = _metrics(y, y_pred)
    bundle = TrainedModelBundle(scaler, "rf" if isinstance(model, RandomForestClassifier) else "logreg",
                                model, feats, classes, domain, task)
    return bundle, {"metrics":metrics, "features":feats, 
                    "feature_importances": getattr(model,"feature_importances_", None)}

def _features_for_domain(rows: List[Dict[str,Any]], domain: str) -> List[List[float]]:
    return ([FeatureExtractorManufacturing.row_to_features(r) for r in rows] if domain=="manufacturing"
            else [FeatureExtractorLogistics.row_to_features(r) for r in rows])

def predict_rows(bundle: TrainedModelBundle, rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    X = _features_for_domain(rows, bundle.domain)
    Xs = bundle.scaler.transform(X)
    probs = (bundle.model.predict_proba(Xs))
    preds = (bundle.model.predict(Xs))
    try:
        idx1 = bundle.classes.index(1)
    except ValueError:
        idx1 = 0
    out = []
    for r, p, pr in zip(rows, preds, probs):
        d = dict(r)
        d["pred_label"] = int(p)
        d["pred_score"] = float(pr[idx1]) if 0 <= idx1 < len(pr) else 0.0
        out.append(d)
    return out

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train & predict for Manufacturing and Logistics (stdlib)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train model(s)")
    tr.add_argument("--domain", choices=["manufacturing","logistics","both"], required=True)
    tr.add_argument("--mfg_csv")
    tr.add_argument("--log_csv")
    tr.add_argument("--log_task", choices=["delay","inefficiency"], default="delay")
    tr.add_argument("--algo", choices=["rf","logreg"], default="rf")
    tr.add_argument("--outdir", required=True)

    pr = sub.add_parser("predict", help="Predict with a saved model")
    pr.add_argument("--model", required=True)
    pr.add_argument("--csv", required=True)
    pr.add_argument("--out_csv", required=True)

    args = ap.parse_args()
    os.makedirs(getattr(args, "outdir", "."), exist_ok=True)

    if args.cmd == "train":
        if args.domain in ("manufacturing","both"):
            if not args.mfg_csv: raise SystemExit("--mfg_csv required for manufacturing training")
            m_rows = read_csv(args.mfg_csv)
            m_bundle, m_info = train("manufacturing", m_rows, task="bottleneck", algo=args.algo)
            m_path = os.path.join(args.outdir, "manufacturing_bottleneck.json")
            m_bundle.save(m_path)
            write_json(os.path.join(args.outdir, "manufacturing_modelcard.json"), m_info)
            print("[TRAIN] manufacturing:", m_info, "->", m_path)
        if args.domain in ("logistics","both"):
            if not args.log_csv: raise SystemExit("--log_csv required for logistics training")
            l_rows = read_csv(args.log_csv)
            l_bundle, l_info = train("logistics", l_rows, task=args.log_task, algo=args.algo)
            l_path = os.path.join(args.outdir, f"logistics_{args.log_task}.json")
            l_bundle.save(l_path)
            write_json(os.path.join(args.outdir, f"logistics_{args.log_task}_modelcard.json"), l_info)
            print(f"[TRAIN] logistics ({args.log_task}):", l_info, "->", l_path)

    elif args.cmd == "predict":
        bundle = TrainedModelBundle.load(args.model)
        rows = read_csv(args.csv)
        preds = predict_rows(bundle, rows)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(preds[0].keys()) if preds else [])
            if preds: w.writeheader()
            for r in preds: w.writerow(r)
        print("[PREDICT] wrote", args.out_csv)

if __name__ == "__main__":
    main()
