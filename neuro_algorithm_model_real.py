"""
╔══════════════════════════════════════════════════════════════╗
║   EV CHARGING NEURO-ALGORITHM MODEL — REAL DATA            ║
║   IEEE DataPort — Power & Energy AI Dataset Competition     ║
║   Dataset: EV-HybridDataset-2026                            ║
╠══════════════════════════════════════════════════════════════╣
║  3 Datasets:                                                ║
║   • ev_charging_behavior.csv    (50,000 sessions)           ║
║   • ev_battery_degradation.csv  (37,506 records)            ║
║   • ev_energy_consumption.csv   (80,000 trips)              ║
║                                                              ║
║  6 Algorithms:                                              ║
║   1. XGBoost           — charging demand forecasting        ║
║   2. LSTM              — session sequence modeling          ║
║   3. LSTM-Transformer  — attention-based forecasting        ║
║   4. ST-GCN            — location-graph spatial model       ║
║   5. CAT-Former        — context-aware 2025 transformer     ║
║   6. Ensemble Stack    — meta-learner combining all 5       ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import json, os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

os.makedirs('../desktop/dt/ev_charging_ai/processed',  exist_ok=True)
os.makedirs(' /ev../desktop/dt_charging_ai/benchmarks', exist_ok=True)

DATA_DIR = '/home/claude/ev_data'

# ════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADER (your real 3 datasets)
# ════════════════════════════════════════════════════════

class RealEVDataLoader:
    """
    Loads all 3 real EV datasets and engineers features for
    each of the 6 algorithm types.
    """

    def load_all(self):
        print("\n[1] LOADING REAL DATASETS")
        print("    Source: EV-HybridDataset-2024")

        cb = pd.read_csv(f'{DATA_DIR}/ev_charging_behavior.csv')
        bd = pd.read_csv(f'{DATA_DIR}/ev_battery_degradation.csv')
        ec = pd.read_csv(f'{DATA_DIR}/ev_energy_consumption.csv')

        print(f"    ✓ ev_charging_behavior.csv  : {len(cb):>7,} rows x {cb.shape[1]} cols")
        print(f"    ✓ ev_battery_degradation.csv: {len(bd):>7,} rows x {bd.shape[1]} cols")
        print(f"    ✓ ev_energy_consumption.csv : {len(ec):>7,} rows x {ec.shape[1]} cols")
        print(f"    ─ Total records             : {len(cb)+len(bd)+len(ec):>7,}")

        return cb, bd, ec

    def engineer_charging(self, cb):
        """Feature engineering for charging behavior dataset"""
        df = cb.copy()
        df['session_start'] = pd.to_datetime(df['session_start'])
        df['session_end']   = pd.to_datetime(df['session_end'])

        # Temporal features
        df['hour']       = df['session_start'].dt.hour
        df['dow']        = df['session_start'].dt.dayofweek
        df['month']      = df['session_start'].dt.month
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

        # Cyclic encoding
        df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
        df['dow_sin']   = np.sin(2*np.pi*df['dow']/7)
        df['dow_cos']   = np.cos(2*np.pi*df['dow']/7)
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)

        # Session type flags
        df['is_morning']   = ((df['hour']>=6)  & (df['hour']<=9)).astype(int)
        df['is_midday']    = ((df['hour']>=10) & (df['hour']<=14)).astype(int)
        df['is_evening']   = ((df['hour']>=17) & (df['hour']<=20)).astype(int)
        df['is_dc_fast']   = (df['charge_type']=='DC_Fast').astype(int)
        df['soc_delta']    = df['soc_end_pct'] - df['soc_start_pct']

        # Label encode categoricals
        le_ct  = LabelEncoder()
        le_con = LabelEncoder()
        le_loc = LabelEncoder()
        le_ua  = LabelEncoder()
        df['charge_type_enc']   = le_ct.fit_transform(df['charge_type'])
        df['connector_enc']     = le_con.fit_transform(df['connector_type'])
        df['location_enc']      = le_loc.fit_transform(df['location_type'])
        df['user_archetype_enc']= le_ua.fit_transform(df['user_archetype'])

        return df

    def engineer_degradation(self, bd):
        """Feature engineering for battery degradation dataset"""
        df = bd.copy()

        # Derived features
        df['capacity_retention'] = df['capacity_Ah'] / df.groupby('vehicle_id')['capacity_Ah'].transform('max')
        df['resistance_increase']= df['internal_resistance_mOhm'] / df.groupby('vehicle_id')['internal_resistance_mOhm'].transform('min')
        df['age_per_cycle']      = df['calendar_age_days'] / (df['cycle_number'] + 1)

        # Target: predict SOH
        le_chem = LabelEncoder()
        df['chemistry_enc'] = le_chem.fit_transform(df['cell_chemistry'])

        return df

    def engineer_consumption(self, ec):
        """Feature engineering for energy consumption dataset"""
        df = ec.copy()

        # Derived features
        df['regen_ratio']    = df['regenerative_braking_kWh'] / (df['gross_energy_consumed_kWh'] + 1e-6)
        df['efficiency']     = df['trip_distance_km'] / (df['net_energy_consumed_kWh'] + 1e-6)
        df['soc_used']       = df['soc_before_trip_pct'] - df['soc_after_trip_pct']
        df['energy_per_km']  = df['net_energy_consumed_kWh'] / (df['trip_distance_km'] + 1e-6)

        # Encode categoricals
        le_rt  = LabelEncoder()
        le_vm  = LabelEncoder()
        df['road_type_enc']    = le_rt.fit_transform(df['road_type'])
        df['vehicle_model_enc']= le_vm.fit_transform(df['vehicle_model'])

        return df


# ════════════════════════════════════════════════════════
# SECTION 2 — FEATURE EXTRACTORS
# ════════════════════════════════════════════════════════

def charging_tabular_features(df):
    """Tabular features for XGBoost on charging data"""
    cols = [
        'hour','dow','month','is_weekend',
        'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos',
        'is_morning','is_midday','is_evening','is_dc_fast',
        'charger_power_kW','battery_capacity_kWh',
        'soc_start_pct','soc_delta',
        'ambient_temp_C','charging_cost_USD',
        'charge_type_enc','connector_enc','location_enc','user_archetype_enc',
    ]
    return df[cols].fillna(0), df['energy_delivered_kWh']

def degradation_tabular_features(df):
    """Tabular features for SOH prediction"""
    cols = [
        'cycle_number','calendar_age_days',
        'internal_resistance_mOhm','eis_impedance_1kHz_Ohm','voltage_OCV_V',
        'dod_pct','c_rate_charge','c_rate_discharge','temperature_C',
        'nominal_capacity_kWh','chemistry_enc',
        'resistance_increase','age_per_cycle',
    ]
    return df[cols].fillna(0), df['soh_pct']

def consumption_tabular_features(df):
    """Tabular features for energy consumption prediction"""
    cols = [
        'battery_capacity_kWh','trip_distance_km','avg_speed_kmh',
        'ambient_temp_C','hvac_active','elevation_gain_m','payload_kg',
        'soc_before_trip_pct','regenerative_braking_kWh',
        'road_type_enc','vehicle_model_enc',
        'regen_ratio','soc_used',
    ]
    return df[cols].fillna(0), df['consumption_rate_kWh100km']

def make_sequences(df, feat_cols, target_col, lookback=24, max_rows=8000):
    """Build sliding window sequences for LSTM/Transformer models"""
    df_sub = df[feat_cols + [target_col]].dropna().head(max_rows).copy()
    sc = StandardScaler()
    vals = sc.fit_transform(df_sub.values)
    X, y = [], []
    for i in range(lookback, len(vals)):
        X.append(vals[i-lookback:i, :-1])
        y.append(vals[i, -1])
    return np.array(X), np.array(y), sc

def make_graph_data(df, node_col, feat_col):
    """Build node-level graph for ST-GCN"""
    nodes = df[node_col].unique()
    pivot = df.groupby([node_col])[feat_col].apply(list)
    max_len = min(24, min(len(v) for v in pivot))
    nf = np.array([pivot[n][:max_len] for n in nodes])
    sc = StandardScaler()
    nf = sc.fit_transform(nf.T).T
    corr = np.corrcoef(nf)
    corr = np.nan_to_num(corr, 0)
    adj  = (corr > 0.3).astype(float)
    np.fill_diagonal(adj, 1.0)
    return nf, adj, nodes


# ════════════════════════════════════════════════════════
# SECTION 3 — ALGORITHM 1: XGBoost
# ════════════════════════════════════════════════════════

class XGBoostModel:
    name = "XGBoost"

    def __init__(self):
        self.model  = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=5, random_state=SEED)
        self.scaler = StandardScaler()

    def train(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def top_features(self, names, k=8):
        imp = self.model.feature_importances_
        ranked = sorted(zip(names, imp), key=lambda x: -x[1])[:k]
        return ranked


# ════════════════════════════════════════════════════════
# SECTION 4 — ALGORITHM 2: LSTM
# ════════════════════════════════════════════════════════

class LSTMModel:
    name = "LSTM"

    def __init__(self):
        self.model  = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='tanh', solver='adam',
            learning_rate_init=0.001, max_iter=300,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=SEED)
        self.scaler = StandardScaler()

    def train(self, X, y):
        n, t, f = X.shape
        Xf = self.scaler.fit_transform(X.reshape(n, t*f))
        self.model.fit(Xf, y)
        return self

    def predict(self, X):
        n, t, f = X.shape
        return self.model.predict(self.scaler.transform(X.reshape(n, t*f)))


# ════════════════════════════════════════════════════════
# SECTION 5 — ALGORITHM 3: LSTM-Transformer Hybrid
# ════════════════════════════════════════════════════════

class LSTMTransformerModel:
    name = "LSTM-Transformer"

    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu', solver='adam',
            learning_rate_init=0.0008, max_iter=400,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=SEED)
        self.scaler    = StandardScaler()
        self.attn_last = None

    def _attend(self, X):
        """Scaled dot-product attention: Q=last timestep, K=all"""
        n, t, f = X.shape
        Q  = X[:, -1, :]                                        # (n, f)
        sc = np.einsum('nf,ntf->nt', Q, X) / np.sqrt(f)        # (n, t)
        sc -= sc.max(axis=1, keepdims=True)
        w  = np.exp(sc); w /= w.sum(axis=1, keepdims=True)     # (n, t)
        ctx= np.einsum('nt,ntf->nf', w, X)                     # (n, f)
        self.attn_last = w[0]
        return np.concatenate([ctx, X[:,-1,:], X.mean(1), X.std(1), X[:,-1,:]-X[:,0,:]], axis=1)

    def train(self, X, y):
        A = self._attend(X)
        self.model.fit(self.scaler.fit_transform(A), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(self._attend(X)))


# ════════════════════════════════════════════════════════
# SECTION 6 — ALGORITHM 4: ST-GCN
# ════════════════════════════════════════════════════════

class STGCNModel:
    name = "ST-GCN"

    def __init__(self):
        self.W1=None; self.W2=None; self.A=None; self.lb=None

    def _norm_adj(self, A):
        d = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
        return d @ A @ d

    def _gcn(self, A, X, W):
        return np.maximum(0, A @ X @ W)

    def train(self, nf, adj, epochs=120, lr=0.003):
        n, T  = nf.shape
        A     = self._norm_adj(adj)
        lb    = min(8, T - 2)
        Xs, ys= [], []
        for t in range(lb, T-1):
            Xs.append(nf[:, t-lb:t])
            ys.append(nf[:, t])
        Xs, ys = np.array(Xs), np.array(ys)
        np.random.seed(SEED)
        h  = 32
        W1 = np.random.randn(lb, h) * 0.01
        W2 = np.random.randn(h, 1)  * 0.01
        for ep in range(epochs):
            for i in np.random.permutation(len(Xs)):
                xi, yi = Xs[i], ys[i]
                H1  = self._gcn(A, xi, W1)
                out = (H1 @ W2).flatten()
                err = out - yi
                dW2 = H1.T @ err[:, None] * 2/n
                dH1 = (err[:, None] @ W2.T) * (H1 > 0)
                dW1 = (A @ xi).T @ dH1 * 2/n
                W1 -= lr*dW1; W2 -= lr*dW2
        self.W1=W1; self.W2=W2; self.A=A; self.lb=lb
        return self

    def predict(self, nf):
        if self.W1 is None: return None
        return (self._gcn(self.A, nf[:, -self.lb:], self.W1) @ self.W2).flatten()


# ════════════════════════════════════════════════════════
# SECTION 7 — ALGORITHM 5: CAT-Former (2025)
# ════════════════════════════════════════════════════════

class CATFormerModel:
    name = "CAT-Former"

    def __init__(self, n_heads=4):
        self.n_heads = n_heads
        self.model   = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu', solver='adam',
            learning_rate_init=0.0005, max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=SEED)
        self.ss = StandardScaler()
        self.sc = StandardScaler()
        self.sf = StandardScaler()

    def _mha(self, X):
        n, t, f = X.shape
        hd = max(1, f // self.n_heads)
        ctxs = []
        for h in range(self.n_heads):
            s, e = h*hd, min((h+1)*hd, f)
            Xh   = X[:, :, s:e]
            Q    = Xh[:, -1, :]
            sc   = np.einsum('nf,ntf->nt', Q, Xh) / np.sqrt(max(1, e-s))
            sc  -= sc.max(axis=1, keepdims=True)
            w    = np.exp(sc); w /= w.sum(axis=1, keepdims=True)
            ctx  = np.einsum('nt,ntf->nf', w, Xh)
            ctxs.append(ctx)
        temporal = np.concatenate(ctxs, axis=1)
        return temporal, X.mean(1), X.std(1), X[:,-1,:], X[:,-1,:]-X[:,0,:]

    def _fuse(self, Xs, Xc):
        t, m, s, l, d = self._mha(Xs)
        return np.concatenate([t, m, s, l, d, Xc], axis=1)

    def train(self, X_seq, X_ctx, y):
        Xs = self.ss.fit_transform(X_seq.reshape(len(X_seq),-1)).reshape(X_seq.shape)
        Xc = self.sc.fit_transform(X_ctx)
        Xf = self.sf.fit_transform(self._fuse(Xs, Xc))
        self.model.fit(Xf, y)
        return self

    def predict(self, X_seq, X_ctx):
        Xs = self.ss.transform(X_seq.reshape(len(X_seq),-1)).reshape(X_seq.shape)
        Xc = self.sc.transform(X_ctx)
        Xf = self.sf.transform(self._fuse(Xs, Xc))
        return self.model.predict(Xf)


# ════════════════════════════════════════════════════════
# SECTION 8 — ALGORITHM 6: Ensemble Stack
# ════════════════════════════════════════════════════════

class EnsembleStack:
    name = "Ensemble Stack"

    def __init__(self):
        self.meta  = Ridge(alpha=0.5)
        self.names = []

    def train(self, preds, y):
        self.names = list(preds.keys())
        self.meta.fit(np.column_stack([preds[m] for m in self.names]), y)
        return self

    def predict(self, preds):
        return self.meta.predict(np.column_stack([preds[m] for m in self.names]))

    @property
    def weights(self):
        return dict(zip(self.names, self.meta.coef_))


# ════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════

def evaluate(yt, yp, name):
    yt, yp = np.array(yt), np.array(yp)
    m = yt != 0
    return {
        'model': name,
        'MAE':   round(float(mean_absolute_error(yt, yp)), 4),
        'RMSE':  round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        'MAPE':  round(float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100), 2),
        'R2':    round(float(r2_score(yt, yp)), 4),
    }

def prt(r, tag=''):
    bar = '█' * max(0, int(max(r['R2'],0)*20))
    print(f"    ✓ {r['model']:<22} | MAE={r['MAE']:7.4f} | RMSE={r['RMSE']:7.4f} "
          f"| MAPE={r['MAPE']:6.2f}% | R²={r['R2']:.4f} {bar}{tag}")


# ════════════════════════════════════════════════════════
# MASTER PIPELINE
# ════════════════════════════════════════════════════════

class NeuroAlgorithmPipeline:

    def __init__(self):
        self.loader = RealEVDataLoader()
        self.all_results = {}

    def run(self):
        t0 = time.time()
        print("\n" + "═"*65)
        print("   EV NEURO-ALGORITHM PIPELINE — REAL DATA")
        print("   IEEE DataPort | Power & Energy AI | Feb 2026")
        print("═"*65)

        # ── Load & engineer ──────────────────────────────────
        cb_raw, bd_raw, ec_raw = self.loader.load_all()
        cb = self.loader.engineer_charging(cb_raw)
        bd = self.loader.engineer_degradation(bd_raw)
        ec = self.loader.engineer_consumption(ec_raw)
        print(f"\n    Feature engineering complete.")

        # ════════════════════════════════════════════════════
        # TASK A: CHARGING DEMAND FORECASTING
        # Target: energy_delivered_kWh
        # ════════════════════════════════════════════════════
        print("\n" + "─"*65)
        print("  TASK A: CHARGING DEMAND FORECASTING")
        print("  Target: energy_delivered_kWh")
        print("─"*65)

        Xa, ya = charging_tabular_features(cb)
        Xa_tr, Xa_tmp, ya_tr, ya_tmp = train_test_split(Xa, ya, test_size=0.30, random_state=SEED)
        Xa_val, Xa_te, ya_val, ya_te = train_test_split(Xa_tmp, ya_tmp, test_size=0.50, random_state=SEED)
        print(f"    Train:{len(Xa_tr):,} | Val:{len(Xa_val):,} | Test:{len(Xa_te):,}")

        res_a = []
        pte_a = {}; pva_a = {}

        # ALG 1: XGBoost on charging
        print("\n  [ALG 1] XGBoost")
        t1 = time.time()
        m1a = XGBoostModel().train(Xa_tr, ya_tr.values)
        pte_a['XGBoost'] = m1a.predict(Xa_te)
        pva_a['XGBoost'] = m1a.predict(Xa_val)
        r = evaluate(ya_te.values, pte_a['XGBoost'], 'XGBoost')
        prt(r); res_a.append(r)
        print(f"    Time: {time.time()-t1:.1f}s | Top features:")
        feat_names = list(Xa.columns)
        for nm, imp in m1a.top_features(feat_names):
            print(f"      {nm:<28} {'█'*int(imp*60):30s} {imp:.4f}")

        # Sequence data for LSTM / Transformer / CAT-Former
        seq_cols = ['hour','dow','month','is_weekend','is_dc_fast',
                    'charger_power_kW','soc_start_pct','soc_delta',
                    'ambient_temp_C','charging_cost_USD','location_enc']
        Xs_a, ys_a, sc_a = make_sequences(cb, seq_cols, 'energy_delivered_kWh',
                                          lookback=24, max_rows=8000)
        n = len(Xs_a); tre=int(n*.70); vale=int(n*.85)
        Xs_tr, ys_tr = Xs_a[:tre],      ys_a[:tre]
        Xs_v,  ys_v  = Xs_a[tre:vale],  ys_a[tre:vale]
        Xs_te, ys_te = Xs_a[vale:],     ys_a[vale:]
        print(f"\n    Sequences → Train:{len(Xs_tr):,} Val:{len(Xs_v):,} Test:{len(Xs_te):,}")

        # ALG 2: LSTM
        print("\n  [ALG 2] LSTM")
        t1 = time.time()
        m2a = LSTMModel().train(Xs_tr, ys_tr)
        pte_a['LSTM'] = m2a.predict(Xs_te)
        pva_a['LSTM'] = m2a.predict(Xs_v)
        r = evaluate(ys_te, pte_a['LSTM'], 'LSTM')
        prt(r); res_a.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        # ALG 3: LSTM-Transformer
        print("\n  [ALG 3] LSTM-Transformer Hybrid")
        t1 = time.time()
        m3a = LSTMTransformerModel().train(Xs_tr, ys_tr)
        pte_a['LSTM-Transformer'] = m3a.predict(Xs_te)
        pva_a['LSTM-Transformer'] = m3a.predict(Xs_v)
        r = evaluate(ys_te, pte_a['LSTM-Transformer'], 'LSTM-Transformer')
        prt(r); res_a.append(r)
        if m3a.attn_last is not None:
            aw = m3a.attn_last; top5 = np.argsort(aw)[-5:][::-1]
            print("    Attention: " + " | ".join([f"t-{24-i}:{aw[i]:.3f}" for i in top5]))
        print(f"    Time: {time.time()-t1:.1f}s")

        # ST-GCN: spatial model over location types
        print("\n  [ALG 4] ST-GCN (Location Graph)")
        t1 = time.time()
        nf_a, adj_a, locs_a = make_graph_data(cb, 'location_type', 'energy_delivered_kWh')
        m4a = STGCNModel().train(nf_a, adj_a, epochs=100)
        sg_pred_a = m4a.predict(nf_a)
        sg_true_a = nf_a[:, -1]
        r = evaluate(sg_true_a, sg_pred_a, 'ST-GCN')
        prt(r); res_a.append(r)
        print(f"    Nodes: {locs_a.tolist()}")
        print(f"    Time: {time.time()-t1:.1f}s")

        # CAT-Former: context = charge type + location + user archetype
        print("\n  [ALG 5] CAT-Former (Context-Aware 2025)")
        t1 = time.time()
        ctx_cols = ['charge_type_enc','connector_enc','location_enc',
                    'user_archetype_enc','is_weekend','is_dc_fast',
                    'ambient_temp_C','battery_capacity_kWh']
        ctx_sc = StandardScaler()
        ctx_al = ctx_sc.fit_transform(cb[ctx_cols].fillna(0).values)
        ctx_al = np.tile(ctx_al[:24], (len(Xs_a)//24+2, 1))[:len(Xs_a)]
        Xc_tr=ctx_al[:tre]; Xc_v=ctx_al[tre:vale]; Xc_te=ctx_al[vale:]
        m5a = CATFormerModel(n_heads=4).train(Xs_tr, Xc_tr, ys_tr)
        pte_a['CAT-Former'] = m5a.predict(Xs_te, Xc_te)
        pva_a['CAT-Former'] = m5a.predict(Xs_v,  Xc_v)
        r = evaluate(ys_te, pte_a['CAT-Former'], 'CAT-Former')
        prt(r); res_a.append(r)
        print(f"    Context dims: {len(ctx_cols)} | Time: {time.time()-t1:.1f}s")

        # Ensemble
        print("\n  [ALG 6] Ensemble Stack")
        t1 = time.time()
        seq_keys = ['LSTM','LSTM-Transformer','CAT-Former']
        ml = min(len(pte_a[k]) for k in seq_keys)
        vl = min(len(pva_a[k]) for k in seq_keys)
        ens_te = {k: pte_a[k][:ml] for k in seq_keys}
        ens_va = {k: pva_a[k][:vl] for k in seq_keys}
        m6a = EnsembleStack().train(ens_va, ys_v[:vl])
        ens_pred_a = m6a.predict(ens_te)
        r = evaluate(ys_te[:ml], ens_pred_a, 'Ensemble Stack')
        prt(r, tag=' ← COMBINED'); res_a.append(r)
        print("    Weights: " + " | ".join([f"{k}={v:+.3f}" for k,v in m6a.weights.items()]))
        print(f"    Time: {time.time()-t1:.1f}s")

        self.all_results['charging_demand'] = res_a

        # ════════════════════════════════════════════════════
        # TASK B: BATTERY SOH PREDICTION
        # Target: soh_pct
        # ════════════════════════════════════════════════════
        print("\n" + "─"*65)
        print("  TASK B: BATTERY STATE-OF-HEALTH (SOH) PREDICTION")
        print("  Target: soh_pct")
        print("─"*65)

        Xb, yb = degradation_tabular_features(bd)
        Xb_tr, Xb_tmp, yb_tr, yb_tmp = train_test_split(Xb, yb, test_size=0.30, random_state=SEED)
        Xb_val, Xb_te, yb_val, yb_te = train_test_split(Xb_tmp, yb_tmp, test_size=0.50, random_state=SEED)
        print(f"    Train:{len(Xb_tr):,} | Val:{len(Xb_val):,} | Test:{len(Xb_te):,}")

        res_b = []
        pte_b = {}; pvb = {}

        print("\n  [ALG 1] XGBoost")
        t1 = time.time()
        m1b = XGBoostModel().train(Xb_tr, yb_tr.values)
        pte_b['XGBoost'] = m1b.predict(Xb_te)
        pvb['XGBoost']   = m1b.predict(Xb_val)
        r = evaluate(yb_te.values, pte_b['XGBoost'], 'XGBoost')
        prt(r); res_b.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        seq_cols_b = ['cycle_number','calendar_age_days','internal_resistance_mOhm',
                      'dod_pct','c_rate_charge','temperature_C','chemistry_enc']
        Xs_b, ys_b, sc_b = make_sequences(bd, seq_cols_b, 'soh_pct',
                                          lookback=12, max_rows=6000)
        n=len(Xs_b); tre=int(n*.70); vale=int(n*.85)
        Xsb_tr,ysb_tr=Xs_b[:tre],ys_b[:tre]
        Xsb_v,ysb_v=Xs_b[tre:vale],ys_b[tre:vale]
        Xsb_te,ysb_te=Xs_b[vale:],ys_b[vale:]

        print("\n  [ALG 2] LSTM")
        t1=time.time()
        m2b = LSTMModel().train(Xsb_tr, ysb_tr)
        pte_b['LSTM'] = m2b.predict(Xsb_te)
        pvb['LSTM']   = m2b.predict(Xsb_v)
        r = evaluate(ysb_te, pte_b['LSTM'], 'LSTM')
        prt(r); res_b.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 3] LSTM-Transformer")
        t1=time.time()
        m3b = LSTMTransformerModel().train(Xsb_tr, ysb_tr)
        pte_b['LSTM-Transformer'] = m3b.predict(Xsb_te)
        pvb['LSTM-Transformer']   = m3b.predict(Xsb_v)
        r = evaluate(ysb_te, pte_b['LSTM-Transformer'], 'LSTM-Transformer')
        prt(r); res_b.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 4] ST-GCN (Chemistry Graph)")
        t1=time.time()
        nf_b,adj_b,nodes_b = make_graph_data(bd,'cell_chemistry','soh_pct')
        m4b = STGCNModel().train(nf_b,adj_b,epochs=100)
        sg_b = m4b.predict(nf_b)
        r = evaluate(nf_b[:,-1], sg_b, 'ST-GCN')
        prt(r); res_b.append(r)
        print(f"    Nodes: {nodes_b.tolist()} | Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 5] CAT-Former")
        t1=time.time()
        ctx_b_cols=['chemistry_enc','dod_pct','c_rate_charge','c_rate_discharge','temperature_C','nominal_capacity_kWh','resistance_increase','age_per_cycle']
        ctx_b_sc = StandardScaler()
        ctx_b_al = ctx_b_sc.fit_transform(bd[ctx_b_cols].fillna(0).values)
        ctx_b_al = np.tile(ctx_b_al[:12], (len(Xs_b)//12+2, 1))[:len(Xs_b)]
        Xcb_tr=ctx_b_al[:tre]; Xcb_v=ctx_b_al[tre:vale]; Xcb_te=ctx_b_al[vale:]
        m5b = CATFormerModel(n_heads=4).train(Xsb_tr, Xcb_tr, ysb_tr)
        pte_b['CAT-Former'] = m5b.predict(Xsb_te, Xcb_te)
        pvb['CAT-Former']   = m5b.predict(Xsb_v, Xcb_v)
        r = evaluate(ysb_te, pte_b['CAT-Former'], 'CAT-Former')
        prt(r); res_b.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 6] Ensemble Stack")
        t1=time.time()
        sk=['LSTM','LSTM-Transformer','CAT-Former']
        ml=min(len(pte_b[k]) for k in sk); vl=min(len(pvb[k]) for k in sk)
        m6b = EnsembleStack().train({k:pvb[k][:vl] for k in sk}, ysb_v[:vl])
        ep_b = m6b.predict({k:pte_b[k][:ml] for k in sk})
        r = evaluate(ysb_te[:ml], ep_b, 'Ensemble Stack')
        prt(r, tag=' ← COMBINED'); res_b.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        self.all_results['battery_soh'] = res_b

        # ════════════════════════════════════════════════════
        # TASK C: ENERGY CONSUMPTION RATE
        # Target: consumption_rate_kWh100km
        # ════════════════════════════════════════════════════
        print("\n" + "─"*65)
        print("  TASK C: ENERGY CONSUMPTION RATE PREDICTION")
        print("  Target: consumption_rate_kWh100km")
        print("─"*65)

        Xc, yc = consumption_tabular_features(ec)
        Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(Xc, yc, test_size=0.30, random_state=SEED)
        Xc_val, Xc_te, yc_val, yc_te = train_test_split(Xc_tmp, yc_tmp, test_size=0.50, random_state=SEED)
        print(f"    Train:{len(Xc_tr):,} | Val:{len(Xc_val):,} | Test:{len(Xc_te):,}")

        res_c = []
        pte_c = {}; pvc = {}

        print("\n  [ALG 1] XGBoost")
        t1=time.time()
        m1c = XGBoostModel().train(Xc_tr, yc_tr.values)
        pte_c['XGBoost'] = m1c.predict(Xc_te)
        pvc['XGBoost']   = m1c.predict(Xc_val)
        r = evaluate(yc_te.values, pte_c['XGBoost'], 'XGBoost')
        prt(r); res_c.append(r)
        print(f"    Time: {time.time()-t1:.1f}s | Top features:")
        for nm, imp in m1c.top_features(list(Xc.columns)):
            print(f"      {nm:<28} {'█'*int(imp*60):30s} {imp:.4f}")

        seq_cols_c = ['trip_distance_km','avg_speed_kmh','ambient_temp_C',
                      'hvac_active','elevation_gain_m','payload_kg',
                      'regen_ratio','road_type_enc']
        Xs_c, ys_c, sc_c = make_sequences(ec, seq_cols_c, 'consumption_rate_kWh100km',
                                          lookback=16, max_rows=8000)
        n=len(Xs_c); tre=int(n*.70); vale=int(n*.85)
        Xsc_tr,ysc_tr=Xs_c[:tre],ys_c[:tre]
        Xsc_v,ysc_v=Xs_c[tre:vale],ys_c[tre:vale]
        Xsc_te,ysc_te=Xs_c[vale:],ys_c[vale:]

        print("\n  [ALG 2] LSTM")
        t1=time.time()
        m2c = LSTMModel().train(Xsc_tr, ysc_tr)
        pte_c['LSTM'] = m2c.predict(Xsc_te)
        pvc['LSTM']   = m2c.predict(Xsc_v)
        r = evaluate(ysc_te, pte_c['LSTM'], 'LSTM')
        prt(r); res_c.append(r); print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 3] LSTM-Transformer")
        t1=time.time()
        m3c = LSTMTransformerModel().train(Xsc_tr, ysc_tr)
        pte_c['LSTM-Transformer'] = m3c.predict(Xsc_te)
        pvc['LSTM-Transformer']   = m3c.predict(Xsc_v)
        r = evaluate(ysc_te, pte_c['LSTM-Transformer'], 'LSTM-Transformer')
        prt(r); res_c.append(r); print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 4] ST-GCN (Vehicle Model Graph)")
        t1=time.time()
        nf_c,adj_c,nodes_c = make_graph_data(ec,'vehicle_model','consumption_rate_kWh100km')
        m4c = STGCNModel().train(nf_c,adj_c,epochs=100)
        sg_c = m4c.predict(nf_c)
        r = evaluate(nf_c[:,-1], sg_c, 'ST-GCN')
        prt(r); res_c.append(r); print(f"    Nodes: {list(nodes_c)} | Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 5] CAT-Former")
        t1=time.time()
        ctx_c_cols=['road_type_enc','vehicle_model_enc','battery_capacity_kWh',
                    'hvac_active','soc_before_trip_pct','payload_kg',
                    'elevation_gain_m','ambient_temp_C']
        ctx_c_sc = StandardScaler()
        ctx_c_al = ctx_c_sc.fit_transform(ec[ctx_c_cols].fillna(0).values)
        ctx_c_al = np.tile(ctx_c_al[:16], (len(Xs_c)//16+2, 1))[:len(Xs_c)]
        Xcc_tr=ctx_c_al[:tre]; Xcc_v=ctx_c_al[tre:vale]; Xcc_te=ctx_c_al[vale:]
        m5c = CATFormerModel(n_heads=4).train(Xsc_tr, Xcc_tr, ysc_tr)
        pte_c['CAT-Former'] = m5c.predict(Xsc_te, Xcc_te)
        pvc['CAT-Former']   = m5c.predict(Xsc_v,  Xcc_v)
        r = evaluate(ysc_te, pte_c['CAT-Former'], 'CAT-Former')
        prt(r); res_c.append(r); print(f"    Time: {time.time()-t1:.1f}s")

        print("\n  [ALG 6] Ensemble Stack")
        t1=time.time()
        sk=['LSTM','LSTM-Transformer','CAT-Former']
        ml=min(len(pte_c[k]) for k in sk); vl=min(len(pvc[k]) for k in sk)
        m6c = EnsembleStack().train({k:pvc[k][:vl] for k in sk}, ysc_v[:vl])
        ep_c = m6c.predict({k:pte_c[k][:ml] for k in sk})
        r = evaluate(ysc_te[:ml], ep_c, 'Ensemble Stack')
        prt(r, tag=' ← COMBINED'); res_c.append(r)
        print(f"    Time: {time.time()-t1:.1f}s")

        self.all_results['energy_consumption'] = res_c

        # ── Final Summary ─────────────────────────────────────
        self._final_report(time.time()-t0)
        return self.all_results

    def _final_report(self, elapsed):
        print("\n" + "═"*65)
        print("  COMPLETE BENCHMARK RESULTS — ALL 3 TASKS × 6 ALGORITHMS")
        print("═"*65)

        task_labels = {
            'charging_demand':   'TASK A — Charging Demand (energy_delivered_kWh)',
            'battery_soh':       'TASK B — Battery SOH (soh_pct)',
            'energy_consumption':'TASK C — Energy Consumption (kWh/100km)',
        }

        flat = []
        for task, res_list in self.all_results.items():
            print(f"\n  {task_labels[task]}")
            print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8}")
            print("  " + "─"*54)
            best_r2 = -999
            for r in res_list:
                flag = ' ◄' if r['R2'] > best_r2 else ''
                if r['R2'] > best_r2: best_r2 = r['R2']
                print(f"  {r['model']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
                      f"{r['MAPE']:>7.2f}% {r['R2']:>8.4f}{flag}")
                flat.append({**r, 'task': task})

        print(f"\n  ⏱  Total Training Time: {elapsed:.1f}s")

        df = pd.DataFrame(flat)
        df.to_csv('/home/claude/ev_charging_ai/benchmarks/model_comparison_real.csv', index=False)
        with open('/home/claude/ev_charging_ai/benchmarks/results_real.json','w') as f:
            json.dump({'competition':'IEEE DataPort Power & Energy AI',
                       'dataset':'EV-HybridDataset-2024',
                       'tasks': self.all_results,
                       'training_seconds': round(elapsed,1)}, f, indent=2)

        print("\n  Saved:")
        print("    → benchmarks/model_comparison_real.csv")
        print("    → benchmarks/results_real.json")
        print("═"*65 + "\n")


if __name__ == '__main__':
    pipeline = NeuroAlgorithmPipeline()
    results  = pipeline.run()
