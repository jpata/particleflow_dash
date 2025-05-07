import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import uproot
import numpy as np
import pandas as pd
import awkward
import vector

import torch
import torch.nn as nn
import pickle as pkl
import os
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional, Any, Union
import functools # For functools.partial

# Assuming mlpf library is correctly installed and structured
import mlpf
from mlpf.model.mlpf import MLPF # Assuming MLPF is the correct class name
from mlpf.model.utils import unpack_predictions, unpack_target
from mlpf.jet_utils import match_jets, to_p4_sph
from mlpf.plotting.plot_utils import cms_label, sample_label

# --- Type Aliases ---
NpArrayFloat = np.typing.NDArray[np.float64]
NpArrayInt = np.typing.NDArray[np.int64]
NpArrayBool = np.typing.NDArray[np.bool]
AkArray = awkward.Array

# --- Configuration ---
config = SimpleNamespace(
    ROOT_FILE_PATH="data/reco_p8_ee_tt_ecm380_1.root",
    PARQUET_FILE_PATH="data/reco_p8_ee_tt_ecm380_1.parquet",
    DEFAULT_B_FIELD_TESLA=-4.0,
    DEFAULT_SCALE_FACTOR=1000.0,
    DEFAULT_EVENT_INDEX=0,
    # DEFAULT_PT_CUT_GEV=1.0, # Removed pt cut
    PION_MASS_GEV=0.139570,
    C_LIGHT=3e8,
    DEFAULT_TRACK_COLLECTION_NAME="SiTracks_Refitted",
    DEFAULT_TRACK_STATE_BRANCH_NAME="SiTracks_1",
    HIT_COLLECTIONS_TO_PLOT=[
        "VXDTrackerHits", "VXDEndcapTrackerHits", "ITrackerHits", "OTrackerHits",
        "ECALBarrel", "ECALEndcap", "ECALOther",
        "HCALBarrel", "HCALEndcap", "HCALOther", "MUON"
    ],
    HIT_FEATURES_STD=["position.x", "position.y", "position.z", "energy", "type"],
    PANDORA_CLUSTER_COLLECTION_NAME="PandoraClusters",
    PANDORA_CLUSTER_FEATURES=[
        "type", "energy", "energyError",
        "position.x", "position.y", "position.z"
    ],
    DEFAULT_SHOW_PANDORA_CLUSTERS=True, # This will be superseded by the new checklist
    PANDORA_CLUSTER_COLOR="cyan",
    TRACK_FEATURE_ORDER=[
        "elemtype", "pt", "eta", "sin_phi", "cos_phi", "p", "chi2", "ndf",
        "dEdx", "dEdxError", "radiusOfInnermostHit", "tanLambda", "D0",
        "omega", "Z0", "time",
    ],
    CLUSTER_FEATURE_ORDER=[
        "elemtype", "et", "eta", "sin_phi", "cos_phi", "energy", "position.x",
        "position.y", "position.z", "iTheta", "energy_ecal", "energy_hcal",
        "energy_other", "num_hits", "sigma_x", "sigma_y", "sigma_z",
    ],
    MODEL_DIR="data",
    TORCH_DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    HIT_LABELS={0: "Raw ECAL hit", 1: "Raw HCAL hit", 2: "Raw Muon chamber hit", 3: "Raw tracker hit"},
    HIT_SUBDETECTOR_COLOR={0: "steelblue", 1: "green", 2: "orange", 3: "red"}
)
config.MODEL_KWARGS_PATH = os.path.join(config.MODEL_DIR, "model_kwargs.pkl")
config.MODEL_CHECKPOINT_PATH = os.path.join(config.MODEL_DIR, "checkpoint-10-1.932789.pth")


# --- Globals for Attention Extraction via Monkey-Patching ---
captured_attentions_for_event: List[Tuple[str, NpArrayFloat]] = []
original_mha_forwards_map: Dict[nn.MultiheadAttention, Any] = {}

# --- Globals for Embedding Extraction via Hooks ---
captured_embeddings_for_event: List[Tuple[str, NpArrayFloat]] = []


def mha_forward_wrapper(layer_name: str, original_forward_method: callable, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
    """
    Wrapper for nn.MultiheadAttention.forward method.
    Forces need_weights=True, captures attention, and returns only attn_output.
    """
    global captured_attentions_for_event

    kwargs['need_weights'] = True
    kwargs['average_attn_weights'] = True


    attn_output, attn_output_weights = original_forward_method(query, key, value, **kwargs)

    if attn_output_weights is not None:
        if attn_output_weights.dim() == 3 and attn_output_weights.shape[0] == 1: # (1, L, S)
            attn_map_np = attn_output_weights[0].detach().cpu().numpy()
            captured_attentions_for_event.append((layer_name, attn_map_np))
        elif attn_output_weights.dim() == 2: # Already (L, S)
             attn_map_np = attn_output_weights.detach().cpu().numpy()
             captured_attentions_for_event.append((layer_name, attn_map_np))
    return attn_output

# --- Hook function for capturing embeddings ---
def embedding_capture_hook(layer_name: str, module: nn.Module, input_tensor: Any, output_tensor: Any):
    global captured_embeddings_for_event
    if isinstance(output_tensor, torch.Tensor):
        if output_tensor.dim() == 3 and output_tensor.shape[0] == 1: # (1, Seq_Len, Embedding_Dim)
            embedding_np = output_tensor[0].detach().cpu().numpy()
            captured_embeddings_for_event.append((layer_name, embedding_np))
        elif output_tensor.dim() == 2: # Already (Seq_Len, Embedding_Dim)
            embedding_np = output_tensor.detach().cpu().numpy()
            captured_embeddings_for_event.append((layer_name, embedding_np))

# --- Utility Functions ---
def pad_and_concatenate_arrays(arr1: NpArrayFloat, arr2: NpArrayFloat) -> NpArrayFloat:
    if arr1.ndim != 2 and arr1.shape[0] != 0 : arr1 = arr1.reshape(0,0)
    if arr2.ndim != 2 and arr2.shape[0] != 0 : arr2 = arr2.reshape(0,0)
    shape1_cols = arr1.shape[1] if arr1.shape[0] > 0 else 0
    shape2_cols = arr2.shape[1] if arr2.shape[0] > 0 else 0
    if arr1.shape[0] == 0 and arr2.shape[0] == 0:
        return np.empty((0, max(shape1_cols, shape2_cols)))
    max_cols = max(shape1_cols, shape2_cols)
    def pad_array(arr: NpArrayFloat, target_cols: int) -> NpArrayFloat:
        if arr.shape[0] == 0:
            return np.empty((0, target_cols))
        if arr.shape[1] < target_cols:
            padding = ((0, 0), (0, target_cols - arr.shape[1]))
            return np.pad(arr, padding, mode='constant', constant_values=0)
        return arr
    arr1_padded = pad_array(arr1, max_cols)
    arr2_padded = pad_array(arr2, max_cols)
    return np.concatenate((arr1_padded, arr2_padded), axis=0)

def calculate_track_pt(omega_data: AkArray, b_field_tesla: float) -> AkArray:
    omega_np = awkward.to_numpy(omega_data)
    pt_np = np.zeros_like(omega_np, dtype=float)
    non_zero_mask = omega_np != 0
    if abs(b_field_tesla) < 1e-9:
         pt_np[non_zero_mask] = np.inf
    else:
        pt_np[non_zero_mask] = (3e-4 * np.abs(b_field_tesla) / np.abs(omega_np[non_zero_mask]))
    pt_np[~non_zero_mask] = 0
    return awkward.from_numpy(pt_np)

# --- Data Loading Functions ---
LoadedRootData = Tuple[Optional[uproot.ReadOnlyDirectory], Optional[uproot.TTree], Dict[str, int], Dict[int, str], int]
def load_root_data(file_path: str) -> LoadedRootData:
    fi = None; ev_tree = None; collectionIDs: Dict[str, int] = {}; collectionIDs_reverse: Dict[int, str] = {}; max_events_root = 0
    try:
        fi = uproot.open(file_path); ev_tree = fi.get("events"); metadata_tree = fi.get("metadata")
        if metadata_tree:
            collectionIDs_data = metadata_tree.arrays("CollectionIDs")
            if "CollectionIDs" in collectionIDs_data.fields and \
               "m_names" in collectionIDs_data["CollectionIDs"].fields and \
               "m_collectionIDs" in collectionIDs_data["CollectionIDs"].fields and \
               len(collectionIDs_data["CollectionIDs"]["m_names"]) > 0:
                names = collectionIDs_data["CollectionIDs"]["m_names"][0]; ids = collectionIDs_data["CollectionIDs"]["m_collectionIDs"][0]
                collectionIDs = {k: v for k, v in zip(names, ids)}; collectionIDs_reverse = {v: k for k, v in collectionIDs.items()}
            else: print("Warning: 'CollectionIDs' structure in 'metadata' tree is not as expected or empty.")
        else: print("Warning: 'metadata' tree not found in ROOT file. CollectionIDs will be empty.")
        if ev_tree: max_events_root = int(ev_tree.num_entries)
        else: print("Warning: 'events' tree not found in ROOT file.")
    except FileNotFoundError: print(f"ERROR: ROOT file not found at {file_path}."); return None, None, {}, {}, 0
    except Exception as e: print(f"Error loading ROOT file or metadata from {file_path}: {e}"); return None, None, {}, {}, 0
    return fi, ev_tree, collectionIDs, collectionIDs_reverse, max_events_root

LoadedParquetData = Tuple[Optional[AkArray], int]
def load_parquet_data(file_path: str) -> LoadedParquetData:
    pq_data = None; max_events_pq = 0
    try:
        pq_data = awkward.from_parquet(file_path)
        if 'X_track' in pq_data.fields: max_events_pq = len(pq_data['X_track']); print(f"Successfully loaded Parquet file: {file_path}. Max events: {max_events_pq}")
        else: print(f"Warning: Parquet file {file_path} loaded, but 'X_track' field is missing."); pq_data = None
    except FileNotFoundError: print(f"ERROR: Parquet file not found at {file_path}. Heatmaps will be empty.")
    except Exception as e: print(f"Error loading Parquet file {file_path}: {e}. Heatmaps will be empty.")
    return pq_data, max_events_pq

LoadedMLPFModel = Tuple[Optional[MLPF], Optional[Dict[str, Any]]]
def load_mlpf_model(kwargs_path: str, checkpoint_path: str, device: torch.device) -> LoadedMLPFModel:
    model_kwargs = None; model_instance = None
    try:
        if not os.path.exists(kwargs_path): print(f"ERROR: Model kwargs file not found at {kwargs_path}"); return None, None
        if not os.path.exists(checkpoint_path): print(f"ERROR: Model checkpoint file not found at {checkpoint_path}"); return None, None
        with open(kwargs_path, "rb") as f: model_kwargs = pkl.load(f); print("Successfully loaded model_kwargs.pkl")
        if not isinstance(model_kwargs, dict): print("ERROR: model_kwargs.pkl did not contain a dictionary."); return None, None
        model_instance = MLPF(**model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model_instance.load_state_dict(model_state); model_instance.eval(); model_instance.to(device)
        if hasattr(model_instance, 'conv_id') and hasattr(model_instance, 'conv_reg'):
            for conv_list_attr in ['conv_id', 'conv_reg']:
                conv_list = getattr(model_instance, conv_list_attr)
                if conv_list is not None:
                    for conv_layer in conv_list:
                        if hasattr(conv_layer, 'enable_ctx_manager'): conv_layer.enable_ctx_manager = False
        print(f"MLPF model loaded from {checkpoint_path} and configured for inference on {device}.")
        return model_instance, model_kwargs
    except FileNotFoundError as e_fnf: print(f"ERROR: Model file not found. {e_fnf}.")
    except Exception as e_model_load: print(f"Error loading MLPF model: {e_model_load}")
    return None, None

# --- Feature Extraction Functions ---
def extract_hit_features(event_tree: uproot.TTree, event_index: int, collection_name: str, features_to_extract: List[str]) -> AkArray:
    feat_arr_for_event: Dict[str, AkArray] = {}; feature_map: List[Tuple[str, str]] = []
    is_tracker_hit = "TrackerHit" in collection_name
    for feat_key in features_to_extract: file_feat_name = "eDep" if is_tracker_hit and feat_key == "energy" else feat_key; feature_map.append((feat_key, file_feat_name))
    branches_to_load = [f"{collection_name}.{ff[1]}" for ff in feature_map]
    missing_pos_branches = any(b.endswith((".position.x", ".position.y", ".position.z")) and b not in event_tree for b in branches_to_load)
    if missing_pos_branches: return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})
    try: event_collection_data = event_tree.arrays(branches_to_load, entry_start=event_index, entry_stop=event_index + 1, library="ak")
    except Exception: return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})
    found_data_for_event = any(bnk in event_collection_data.fields and len(event_collection_data[bnk][0]) > 0 for bnk in branches_to_load if bnk in event_collection_data.fields)
    if not found_data_for_event: return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})
    num_hits = 0; first_branch_key = f"{collection_name}.{feature_map[0][1]}"
    if first_branch_key in event_collection_data.fields and len(event_collection_data[first_branch_key]) > 0: num_hits = len(event_collection_data[first_branch_key][0])
    for original_feat_name, feat_name_in_file in feature_map:
        branch_full_name_key = f"{collection_name}.{feat_name_in_file}"
        if branch_full_name_key in event_collection_data.fields and len(event_collection_data[branch_full_name_key]) > 0: feat_arr_for_event[original_feat_name] = event_collection_data[branch_full_name_key][0]
        else: feat_arr_for_event[original_feat_name] = awkward.from_numpy(np.zeros(num_hits))
    subdetector_val = 3;
    if collection_name.startswith("ECAL"): subdetector_val = 0
    elif collection_name.startswith("HCAL"): subdetector_val = 1
    elif collection_name.startswith("MUON"): subdetector_val = 2
    feat_arr_for_event["subdetector"] = awkward.from_numpy(np.full(num_hits, subdetector_val, dtype=np.int32))
    if feat_arr_for_event:
        lengths = [len(arr) for arr in feat_arr_for_event.values()]
        if len(set(lengths)) > 1:
            min_len = min(l for l in lengths if l > 0) if any(l > 0 for l in lengths) else 0
            for k_f in feat_arr_for_event: feat_arr_for_event[k_f] = feat_arr_for_event[k_f][:min_len]
    return awkward.Array(feat_arr_for_event)

TrackFeatures = Tuple[NpArrayFloat, NpArrayFloat, NpArrayFloat, NpArrayInt, NpArrayFloat]
def extract_track_features(event_tree: uproot.TTree, event_index: int, track_collection: str, track_state_branch: str, b_field_tesla: float) -> TrackFeatures:
    empty_result: TrackFeatures = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    track_base_feats = ["type", "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit"]; track_state_params = ["tanLambda", "D0", "phi", "omega", "Z0", "time"]
    track_branches = [f"{track_collection}/{track_collection}.{feat}" for feat in track_base_feats]; track_branches.append(f"{track_collection}/{track_collection}.trackStates_begin")
    state_branches = [f"{track_state_branch}/{track_state_branch}.{param}" for param in track_state_params]
    all_needed_branches = track_branches + state_branches
    available_branches = set(event_tree.keys(filter_name=f"{track_collection}/*") + event_tree.keys(filter_name=f"{track_state_branch}/*"))
    if any(b.startswith(f"{track_collection}.") and b not in available_branches for b in track_branches) or any(b not in available_branches for b in state_branches): return empty_result
    try: event_track_data = event_tree.arrays(all_needed_branches, entry_start=event_index, entry_stop=event_index + 1, library="ak")
    except Exception: return empty_result
    base_type_branch_key = f"{track_collection}/{track_collection}.type"
    if not (base_type_branch_key in event_track_data.fields and len(event_track_data[base_type_branch_key][0]) > 0): return empty_result
    track_features_ak: Dict[str, AkArray] = {}
    for feat in track_base_feats:
        branch_key = f"{track_collection}/{track_collection}.{feat}"
        if branch_key in event_track_data.fields and len(event_track_data[branch_key]) > 0: track_features_ak[feat] = event_track_data[branch_key][0]
        else: return empty_result
    num_tracks = len(track_features_ak["type"]);
    if num_tracks == 0: return empty_result
    track_state_indices_branch_key = f"{track_collection}/{track_collection}.trackStates_begin"
    if track_state_indices_branch_key not in event_track_data.fields or len(event_track_data[track_state_indices_branch_key]) == 0: return empty_result
    track_state_indices = event_track_data[track_state_indices_branch_key][0]
    if len(track_state_indices) != num_tracks or (len(track_state_indices) > 0 and awkward.any(track_state_indices < 0)): return empty_result
    for param_name in track_state_params:
        param_branch_key = f"{track_state_branch}/{track_state_branch}.{param_name}"
        if param_branch_key not in event_track_data.fields or len(event_track_data[param_branch_key]) == 0: return empty_result
        all_states_for_param = event_track_data[param_branch_key][0]
        if len(track_state_indices) == 0: track_features_ak[param_name] = awkward.Array([])
        elif awkward.any(track_state_indices >= len(all_states_for_param)): return empty_result
        else: track_features_ak[param_name] = all_states_for_param[track_state_indices]
    track_features_ak["pt"] = calculate_track_pt(track_features_ak["omega"], b_field_tesla)
    phi_np: NpArrayFloat = awkward.to_numpy(track_features_ak["phi"]); pt_np: NpArrayFloat = awkward.to_numpy(track_features_ak["pt"])
    tanLambda_np: NpArrayFloat = awkward.to_numpy(track_features_ak["tanLambda"]); omega_np: NpArrayFloat = awkward.to_numpy(track_features_ak["omega"])
    px_np = np.cos(phi_np) * pt_np; py_np = np.sin(phi_np) * pt_np; pz_np = tanLambda_np * pt_np; charge_np = np.sign(omega_np).astype(int)
    return px_np, py_np, pz_np, charge_np, pt_np

def extract_pandora_cluster_features(event_tree: uproot.TTree, event_index: int, collection_name: str, features_to_extract: List[str]) -> Optional[pd.DataFrame]:
    if collection_name not in event_tree: return None
    pandora_branches_to_request = [f"{collection_name}/{collection_name}.{feat_name}" for feat_name in features_to_extract]
    available_branches_in_tree = [b for b in pandora_branches_to_request if b in event_tree.keys(filter_name=f"{collection_name}/*")]
    if not available_branches_in_tree: return None
    try: pandora_event_data = event_tree.arrays(available_branches_in_tree, entry_start=event_index, entry_stop=event_index + 1, library="ak")
    except Exception: return None
    num_clusters = 0
    first_valid_branch = next((b for b in pandora_branches_to_request if b in pandora_event_data.fields and len(pandora_event_data[b]) > 0), None)
    if first_valid_branch: num_clusters = len(pandora_event_data[first_valid_branch][0])
    if num_clusters == 0: return pd.DataFrame()
    df_pandora_dict: Dict[str, NpArrayFloat] = {}; all_essential_cols_present = True
    pandora_feature_to_df_col_map = {"type": "type", "energy": "energy", "energyError": "energyError", "position.x": "x", "position.y": "y", "position.z": "z"}
    for original_feat_name in features_to_extract:
        branch_key_to_check = f"{collection_name}/{collection_name}.{original_feat_name}"; df_col_name = pandora_feature_to_df_col_map.get(original_feat_name, original_feat_name)
        if branch_key_to_check in pandora_event_data.fields and len(pandora_event_data[branch_key_to_check]) > 0: df_pandora_dict[df_col_name] = awkward.to_numpy(pandora_event_data[branch_key_to_check][0])
        else: df_pandora_dict[df_col_name] = np.zeros(num_clusters);
        if df_col_name in ["x", "y", "z", "energy"]: all_essential_cols_present = False # This condition seems wrong, should be if essential cols are MISSING
    return pd.DataFrame(df_pandora_dict)

# --- ML Model Inference (Modified with Monkey-Patching and Hooks) ---
MLPFOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
MLPFInferenceResult = Tuple[Optional[MLPFOutput], str, List[Tuple[str, NpArrayFloat]], List[Tuple[str, NpArrayFloat]]] # preds, msg, attentions, embeddings

def run_mlpf_inference(
    model: MLPF,
    model_kwargs: Dict[str, Any],
    x_track_event_np: NpArrayFloat,
    x_cluster_event_np: NpArrayFloat,
    device: torch.device
) -> MLPFInferenceResult:
    """Runs MLPF model inference, monkey-patching MHA for attention & using hooks for embeddings."""
    global captured_attentions_for_event, original_mha_forwards_map
    global captured_embeddings_for_event # For hooks
    captured_attentions_for_event = []
    original_mha_forwards_map.clear()
    captured_embeddings_for_event = [] # Clear for current event

    model_input_dim = model_kwargs.get("input_dim")
    if model_input_dim is None:
        return None, "input_dim not found in MODEL_KWARGS", [], []

    def _ensure_2d_shape(arr: NpArrayFloat, target_dim: int) -> NpArrayFloat:
        if arr.ndim == 0: return np.empty((0, target_dim))
        if arr.ndim == 1:
            if arr.shape[0] == target_dim : return np.expand_dims(arr, axis=0)
            elif arr.shape[0] == 0 : return np.empty((0, target_dim))
            else: return np.empty((0, target_dim))
        if arr.shape[0] == 0: return np.empty((0, target_dim))
        return arr[..., :target_dim]

    x_track_np = _ensure_2d_shape(x_track_event_np, model_input_dim)
    x_cluster_np = _ensure_2d_shape(x_cluster_event_np, model_input_dim)

    if x_track_np.shape[0] == 0 and x_cluster_np.shape[0] == 0:
        return None, "No track or cluster features to run MLPF model.", [], []

    combined_features_np = pad_and_concatenate_arrays(x_track_np, x_cluster_np)
    if combined_features_np.shape[0] == 0:
        return None, "No combined features after processing for MLPF model.", [], []

    model_input_tensor = torch.tensor(combined_features_np, dtype=torch.float32).unsqueeze(0).to(device)
    if model_input_tensor.shape[2] == 0:
        return None, "Combined features tensor has 0 features, cannot create mask.", [], []
    mask = (model_input_tensor[:, :, 0] != 0).to(device)

    # --- Setup MHA Monkey-Patching ---
    mha_layers_to_patch: List[Tuple[str, nn.MultiheadAttention]] = []
    module_lists_names = ['conv_id', 'conv_reg']
    for list_name in module_lists_names:
        if hasattr(model, list_name):
            module_list = getattr(model, list_name)
            if module_list is not None:
                for i, layer in enumerate(module_list):
                    if hasattr(layer, 'mha') and isinstance(layer.mha, nn.MultiheadAttention):
                        mha_layers_to_patch.append((f"{list_name}_{i}", layer.mha))

    for layer_name, mha_instance in mha_layers_to_patch:
        if mha_instance not in original_mha_forwards_map:
            original_mha_forwards_map[mha_instance] = mha_instance.forward
            mha_instance.forward = functools.partial(mha_forward_wrapper, layer_name, original_mha_forwards_map[mha_instance])

    # --- Setup Forward Hooks for Embeddings ---
    hooks = []
    if hasattr(model, 'conv_id') and model.conv_id is not None and len(model.conv_id) > 0:
        # Hook the last layer of the conv_id ModuleList
        hook = model.conv_id[-1].register_forward_hook(
            functools.partial(embedding_capture_hook, "final_conv_id_output")
        )
        hooks.append(hook)

    if hasattr(model, 'conv_reg') and model.conv_reg is not None and len(model.conv_reg) > 0:
        # Hook the last layer of the conv_reg ModuleList
        hook = model.conv_reg[-1].register_forward_hook(
            functools.partial(embedding_capture_hook, "final_conv_reg_output")
        )
        hooks.append(hook)
    # --- End of Hook Setup ---

    preds_tuple: Optional[MLPFOutput] = None
    try:
        with torch.no_grad():
            raw_model_output = model(model_input_tensor, mask)
            if isinstance(raw_model_output, tuple) and len(raw_model_output) == 4 and \
               all(isinstance(p, torch.Tensor) for p in raw_model_output):
                preds_tuple = raw_model_output
            elif isinstance(raw_model_output, tuple) and len(raw_model_output) >= 4:
                 preds_tuple = (raw_model_output[0], raw_model_output[1], raw_model_output[2], raw_model_output[3])

    finally:
        # --- Restore Original MHA Forward Methods ---
        for mha_instance, original_forward in original_mha_forwards_map.items():
            mha_instance.forward = original_forward
        original_mha_forwards_map.clear()
        # --- Remove Forward Hooks ---
        for hook in hooks:
            hook.remove()

    attentions_to_return = list(captured_attentions_for_event)
    embeddings_to_return = list(captured_embeddings_for_event) # Make a copy

    if preds_tuple is None:
        return None, "Model prediction was not in the expected format or failed.", attentions_to_return, embeddings_to_return

    preds_binary_particle, preds_pid, preds_momentum, preds_pu = preds_tuple
    output_msg = (
        f"MLPF Model Inference Output:\n"
        f"  preds_binary_particle shape: {preds_binary_particle.shape}\n"
        f"  preds_pid shape: {preds_pid.shape}\n"
        f"  preds_momentum shape: {preds_momentum.shape}\n"
        f"  preds_pu shape: {preds_pu.shape}\n"
        f"  Number of input elements to model: {combined_features_np.shape[0]}"
    )
    if not attentions_to_return: output_msg += "\n  No attention matrices captured."
    else:
        output_msg += f"\n  Captured {len(attentions_to_return)} attention matrices:"
        for name, attn in attentions_to_return: output_msg += f"\n    - {name}: shape {attn.shape}"

    if not embeddings_to_return: output_msg += "\n  No intermediate embeddings captured."
    else:
        output_msg += f"\n  Captured {len(embeddings_to_return)} intermediate embedding matrices:"
        for name, emb in embeddings_to_return: output_msg += f"\n    - {name}: shape {emb.shape}"

    return preds_tuple, output_msg, attentions_to_return, embeddings_to_return

# --- Plotting Helper Functions ---
def create_empty_heatmap_figure(title: str = "No Data / Error") -> go.Figure:
    fig = go.Figure(); fig.update_layout(title=title, xaxis_title="Features", yaxis_title="Index")
    fig.add_annotation(text="Data unavailable or error in loading.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig

def create_feature_heatmap(data_np: NpArrayFloat, feature_order: List[str], title: str, y_axis_prefix: str, colorscale: str = 'Viridis') -> go.Figure:
    if data_np.ndim == 1: data_np = np.expand_dims(data_np, axis=0)
    if data_np.shape[0] == 0:
        fig = go.Figure().update_layout(title=f"{title} - No Data", xaxis_title="Features", yaxis_title=f"{y_axis_prefix} Index")
        fig.add_annotation(text=f"No {y_axis_prefix.lower()} data for this event.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return fig
    if data_np.shape[1] != len(feature_order): return create_empty_heatmap_figure(f"{title} - Feature Mismatch (Data: {data_np.shape[1]}, Expected: {len(feature_order)})")
    fig = go.Figure(data=go.Heatmap(z=data_np, x=feature_order, y=[f'{y_axis_prefix} {i}' for i in range(data_np.shape[0])], colorscale=colorscale, colorbar=dict(title='Normalized Value'), xgap=1, ygap=0))
    fig.update_layout(title=title, xaxis_title="Features", yaxis_title=f"{y_axis_prefix} Index", xaxis_tickangle=-45, margin=dict(l=70, r=20, b=120, t=50))
    return fig

def create_attention_heatmap_figure(
    attn_matrix_np: NpArrayFloat,
    title: str,
    colorscale: str = 'Blues'
) -> go.Figure:
    if not isinstance(attn_matrix_np, np.ndarray) or attn_matrix_np.ndim != 2 or \
       attn_matrix_np.shape[0] == 0 or attn_matrix_np.shape[1] == 0:
        fig = go.Figure().update_layout(title=f"{title} - No/Invalid Data (Shape: {attn_matrix_np.shape if isinstance(attn_matrix_np, np.ndarray) else 'N/A'})")
        fig.add_annotation(text="Attention data not available or invalid for heatmap.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    target_seq_len, source_seq_len = attn_matrix_np.shape
    y_labels = [f'{i}' for i in range(target_seq_len)]
    x_labels = [f'{i}' for i in range(source_seq_len)]

    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix_np, x=x_labels, y=y_labels, colorscale=colorscale,
        colorbar=dict(title='Attention Weight'), xgap=0, ygap=0
    ))
    fig.update_layout(
        title=title, xaxis_title="Source Sequence Elements (Key/Value)",
        yaxis_title="Target Sequence Elements (Query)", xaxis_tickangle=-45,
        yaxis_autorange='reversed',
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'),
        xaxis=dict(constrain='domain'),
        margin=dict(l=100, r=20, b=120, t=50),
    )
    return fig

def create_embedding_heatmap_figure(
    embedding_matrix_np: NpArrayFloat,
    title: str,
    colorscale: str = 'RdBu'
) -> go.Figure:
    """Creates a heatmap figure for an embedding matrix."""
    if not isinstance(embedding_matrix_np, np.ndarray) or embedding_matrix_np.ndim != 2 or \
       embedding_matrix_np.shape[0] == 0 or embedding_matrix_np.shape[1] == 0:
        fig = go.Figure().update_layout(title=f"{title} - No/Invalid Data (Shape: {embedding_matrix_np.shape if isinstance(embedding_matrix_np, np.ndarray) else 'N/A'})")
        fig.add_annotation(text="Embedding data not available or invalid for heatmap.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    seq_len, embedding_dim = embedding_matrix_np.shape
    y_labels = [f' {i}' for i in range(seq_len)]
    # Display only a subset of x_labels if embedding_dim is too large to avoid clutter
    if embedding_dim > 50: # Heuristic: if more than 50 dims, label every 10th
        x_labels_ticks = [f'Dim {j*10}' for j in range(embedding_dim // 10)]
        x_labels_values = [j*10 for j in range(embedding_dim // 10)]
        if embedding_dim % 10 != 0 : # Add last label if not perfectly divisible
            x_labels_ticks.append(f'Dim {embedding_dim-1}')
            x_labels_values.append(embedding_dim-1)
    else:
        x_labels_ticks = [f'Dim {j}' for j in range(embedding_dim)]
        x_labels_values = list(range(embedding_dim))


    fig = go.Figure(data=go.Heatmap(
        z=embedding_matrix_np,
        # x=x_labels, # Using all labels can be too dense for large embedding_dim
        y=y_labels,
        colorscale=colorscale,
        colorbar=dict(title='Embedding Value'),
        xgap=0, ygap=0
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Embedding Dimension",
        yaxis_title="Sequence Element",
        xaxis_tickangle=-45,
        #xaxis=dict(tickmode='array', tickvals=x_labels_values, ticktext=x_labels_ticks),
        margin=dict(l=100, r=20, b=120, t=50),
    )
    return fig


HelixTraceData = Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]
def generate_helix_trace_data(px_values: NpArrayFloat, py_values: NpArrayFloat, pz_values: NpArrayFloat, charge_values: NpArrayInt, particle_mass: float, b_field: float, scale_factor: float, num_points_helix: int = 70, num_points_straight: int = 50) -> HelixTraceData:
    helix_x, helix_y, helix_z = [], [], []
    for i in range(len(px_values)):
        if not all(np.isfinite(val) for val in [px_values[i], py_values[i], pz_values[i]]): continue
        particle_vec = vector.obj(px=px_values[i], py=py_values[i], pz=pz_values[i], mass=particle_mass)
        beta = particle_vec.beta
        if not (np.isfinite(beta) and beta > 1e-9): continue
        path_len_m = 2.0
        if charge_values[i] == 0 or abs(b_field) < 1e-9 or particle_vec.pt < 1e-6:
            time_param = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_straight); current_path = config.C_LIGHT * beta * time_param
            p_abs = particle_vec.p; dir_x = particle_vec.px / p_abs if p_abs > 1e-9 else 0; dir_y = particle_vec.py / p_abs if p_abs > 1e-9 else 0; dir_z = particle_vec.pz / p_abs if p_abs > 1e-9 else 0
            path_x = scale_factor * dir_x * current_path; path_y = scale_factor * dir_y * current_path; path_z = scale_factor * dir_z * current_path
        else:
            abs_b_field = abs(b_field)
            if abs_b_field < 1e-9: # Should be caught by outer if, but for safety
                time_param = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_straight); current_path = config.C_LIGHT * beta * time_param
                p_abs = particle_vec.p; dir_x = particle_vec.px / p_abs if p_abs > 1e-9 else 0; dir_y = particle_vec.py / p_abs if p_abs > 1e-9 else 0; dir_z = particle_vec.pz / p_abs if p_abs > 1e-9 else 0
                path_x = scale_factor * dir_x * current_path; path_y = scale_factor * dir_y * current_path; path_z = scale_factor * dir_z * current_path
            else:
                radius_m = particle_vec.pt / (abs(charge_values[i]) * 0.299792458 * abs_b_field)
                time_vals = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_helix)
                s_xy_t = (particle_vec.pt / particle_vec.p if particle_vec.p > 1e-9 else 0) * (config.C_LIGHT * beta * time_vals)
                arg_angle_helix = s_xy_t / radius_m if radius_m > 1e-9 else np.zeros_like(time_vals)
                signed_arg_angle_helix = np.sign(charge_values[i] * b_field) * arg_angle_helix; initial_phase_angle = particle_vec.phi - np.pi/2.0
                path_x = scale_factor * radius_m * (np.cos(initial_phase_angle + signed_arg_angle_helix) - np.cos(initial_phase_angle))
                path_y = scale_factor * radius_m * (np.sin(initial_phase_angle + signed_arg_angle_helix) - np.sin(initial_phase_angle))
                path_z = scale_factor * (particle_vec.pz / particle_vec.p if particle_vec.p > 1e-9 else 0) * (config.C_LIGHT * beta * time_vals)
        helix_x.extend(list(path_x)); helix_y.extend(list(path_y)); helix_z.extend(list(path_z))
        helix_x.append(None); helix_y.append(None); helix_z.append(None)
    return helix_x, helix_y, helix_z

def create_3d_event_display_figure(traces: List[go.Scatter3d], event_index: int, b_field: float, scale_factor: float, root_data_error: bool = False) -> go.Figure:
    fig = go.Figure(data=traces)
    title = f"3D Event Display: Event {event_index} (B={b_field:.1f}T, Scale={scale_factor:.0f})" # Removed pT cut from title
    if root_data_error: title = f"Error loading 3D data for Event {event_index}. Display may be incomplete."
    fig.update_layout(title=title, scene=dict(xaxis=dict(title='X (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-2500, 2500]), yaxis=dict(title='Y (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-2500, 2500]), zaxis=dict(title='Z (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-4500, 4500]), camera=dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)), aspectmode='cube'), legend=dict(x=0.75, y=0.95, font=dict(size=10)), margin=dict(l=0, r=0, b=0, t=40), uirevision="persistent_view")
    fig.update_traces(marker_line_width=0, selector=dict(type='scatter3d'))
    if not traces: fig.add_annotation(text="No 3D data to display for this event or parameters.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
    return fig

# --- Global Data (Loaded once) ---
root_file_obj: Optional[uproot.ReadOnlyDirectory] = None; event_tree_global: Optional[uproot.TTree] = None
collectionIDs_global: Dict[str, int] = {}; collectionIDs_reverse_global: Dict[int, str] = {}; max_events_root_global: int = 0
parquet_data_global: Optional[AkArray] = None; max_events_pq_global: int = 0
mlpf_model_global: Optional[MLPF] = None; model_kwargs_global: Optional[Dict[str, Any]] = None
MAX_EVENTS_AVAILABLE: int = 0

def initialize_global_data() -> None:
    global root_file_obj, event_tree_global, collectionIDs_global, collectionIDs_reverse_global, max_events_root_global, parquet_data_global, max_events_pq_global, mlpf_model_global, model_kwargs_global, MAX_EVENTS_AVAILABLE
    print(f"Using PyTorch device: {config.TORCH_DEVICE}")
    root_file_obj, event_tree_global, collectionIDs_global, collectionIDs_reverse_global, max_events_root_global = load_root_data(config.ROOT_FILE_PATH)
    parquet_data_global, max_events_pq_global = load_parquet_data(config.PARQUET_FILE_PATH)
    mlpf_model_global, model_kwargs_global = load_mlpf_model(config.MODEL_KWARGS_PATH, config.MODEL_CHECKPOINT_PATH, config.TORCH_DEVICE)
    if max_events_root_global > 0 and max_events_pq_global > 0: MAX_EVENTS_AVAILABLE = min(max_events_root_global, max_events_pq_global);
    if max_events_root_global != max_events_pq_global and MAX_EVENTS_AVAILABLE > 0: print(f"Warning: Mismatch event counts. ROOT: {max_events_root_global}, Parquet: {max_events_pq_global}. Using min: {MAX_EVENTS_AVAILABLE}")
    elif max_events_root_global > 0: MAX_EVENTS_AVAILABLE = max_events_root_global; print(f"Warning: Parquet data has no events or failed to load. Using ROOT event count: {MAX_EVENTS_AVAILABLE}")
    elif max_events_pq_global > 0: MAX_EVENTS_AVAILABLE = max_events_pq_global; print(f"Warning: ROOT data has no events or failed to load. Using Parquet event count: {MAX_EVENTS_AVAILABLE}")
    else: MAX_EVENTS_AVAILABLE = 0; print("Warning: No events found in either ROOT or Parquet file.")

# --- Dash Application ---
app = dash.Dash(__name__); app.title = "CLIC Event Display & Feature Inspector"
def create_app_layout() -> html.Div:
    max_event_val = max(0, MAX_EVENTS_AVAILABLE -1) if MAX_EVENTS_AVAILABLE > 0 else 0; input_disabled = MAX_EVENTS_AVAILABLE == 0
    return html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
        html.H1("Interactive Particle Event Display & Feature Inspector", style={'textAlign': 'center', 'color': '#333'}),
        html.Div(style={'width': '95%', 'margin': 'auto', 'padding': '15px', 'backgroundColor': '#f0f0f0', 'borderRadius': '8px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '15px'}, children=[
                html.Div(style={'margin': '5px'}, children=[html.Label("Event Index (iev):", style={'fontWeight': 'bold', 'marginRight': '5px'}), dcc.Input(id='iev-input', type='number', value=config.DEFAULT_EVENT_INDEX, min=0, max=max_event_val, step=1, disabled=input_disabled, style={'padding': '5px', 'width': '80px'})]),
                html.Div(style={'margin': '5px'}, children=[html.Label("B-Field (T):", style={'fontWeight': 'bold', 'marginRight': '5px'}), dcc.Input(id='b-field-input', type='number', value=config.DEFAULT_B_FIELD_TESLA, step=0.1, style={'padding': '5px', 'width': '80px'})]),
                html.Div(style={'margin': '5px'}, children=[html.Label("Scale Factor:", style={'fontWeight': 'bold', 'marginRight': '5px'}), dcc.Input(id='scale-input', type='number', value=config.DEFAULT_SCALE_FACTOR, step=100, style={'padding': '5px', 'width': '80px'})]),
                # html.Div(style={'margin': '5px'}, children=[html.Label("Min Track pT (GeV):", style={'fontWeight': 'bold', 'marginRight': '5px'}), dcc.Input(id='pt-cut-input', type='number', value=config.DEFAULT_PT_CUT_GEV, min=0.0, step=0.1, style={'padding': '5px', 'width': '80px'})]), # Removed pt-cut-input
                html.Div(style={'margin': '10px'}, children=[
                    dcc.Checklist(
                        id='display-toggle-checklist',
                        options=[
                            {'label': 'Show Hits', 'value': 'show_hits'},
                            {'label': 'Show Tracks', 'value': 'show_tracks'},
                            {'label': 'Show Pandora Clusters', 'value': 'show_pandora_clusters'}
                        ],
                        value=['show_hits', 'show_tracks', 'show_pandora_clusters'], # Default values
                        inline=True, # Display options horizontally
                        style={'fontWeight': 'bold'}
                    )
                ]),
            ]),
            html.Div(id='error-message-display', style={'marginTop': '10px', 'marginBottom': '10px', 'color': 'red', 'textAlign': 'center', 'minHeight': '20px', 'whiteSpace': 'pre-wrap'}),
            html.Div(id='model-inference-output-display', style={'marginTop': '10px', 'marginBottom': '10px', 'color': 'blue', 'textAlign': 'left', 'minHeight': '20px', 'whiteSpace': 'pre-wrap', 'backgroundColor': '#e6f7ff', 'padding': '10px', 'borderRadius': '5px', 'fontFamily': 'monospace', 'fontSize':'0.9em', 'maxHeight': '200px', 'overflowY':'auto'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'gap': '15px',  'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '2 1 600px', 'minWidth': '60%'}, children=[dcc.Loading(id="loading-graph", type="circle", children=[dcc.Graph(id='particle-graph-display', style={'height': '75vh', 'minHeight': '500px'})])]),
                html.Div(style={'flex': '1 1 300px', 'minWidth': '35%', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}, children=[
                    html.H3("Track Features (X_track) - Normalized", style={'textAlign': 'center', 'marginBlock': '5px'}),
                    dcc.Loading(id="loading-xtrack-heatmap", type="circle", children=[dcc.Graph(id='xtrack-heatmap-display', style={'height': '32vh', 'minHeight': '220px'})]),
                    html.H3("Cluster Features (X_cluster) - Normalized", style={'textAlign': 'center', 'marginBlock': '5px'}),
                    dcc.Loading(id="loading-xcluster-heatmap", type="circle", children=[dcc.Graph(id='xcluster-heatmap-display', style={'height': '32vh', 'minHeight': '220px'})]),
                ])
            ]),
            html.Div(id='attention-heatmaps-panel', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'}, children=[]),
            html.Div(id='embedding-heatmaps-panel', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}, children=[]), # New panel for embeddings
        ])
    ])

@app.callback(
    [Output('particle-graph-display', 'figure'),
     Output('xtrack-heatmap-display', 'figure'),
     Output('xcluster-heatmap-display', 'figure'),
     Output('error-message-display', 'children'),
     Output('model-inference-output-display', 'children'),
     Output('attention-heatmaps-panel', 'children'),
     Output('embedding-heatmaps-panel', 'children')], # New output
    [Input('iev-input', 'value'),
     Input('b-field-input', 'value'),
     Input('scale-input', 'value'),
     # Input('pt-cut-input', 'value'), # Removed pt-cut-input
     Input('display-toggle-checklist', 'value')] # New input for toggles
)
def update_graph_and_heatmaps(current_event_idx: Optional[int], b_field_val: Optional[float], scale_factor_val: Optional[float], display_options: List[str]) -> Tuple[go.Figure, go.Figure, go.Figure, str, str, List[Any], List[Any]]: # Added List[Any] for embeddings
    error_messages: List[str] = []; model_output_msg: str = "MLPF Model not run or no data.";
    attention_figures_children: List[Any] = []
    embedding_figures_children: List[Any] = [] # For new embedding heatmaps

    if current_event_idx is None: current_event_idx = config.DEFAULT_EVENT_INDEX
    if b_field_val is None: b_field_val = config.DEFAULT_B_FIELD_TESLA
    if scale_factor_val is None: scale_factor_val = config.DEFAULT_SCALE_FACTOR
    # if min_pt_cut_val is None: min_pt_cut_val = config.DEFAULT_PT_CUT_GEV # Removed
    # show_pandora_clusters = 'show' in pandora_toggle_val # Replaced by display_options
    show_hits = 'show_hits' in display_options
    show_tracks = 'show_tracks' in display_options
    show_pandora_clusters = 'show_pandora_clusters' in display_options


    empty_heatmap = create_empty_heatmap_figure(); xtrack_fig = empty_heatmap; xcluster_fig = empty_heatmap
    fig_3d = go.Figure().update_layout(title="Error: Event index invalid or data unavailable.", scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"))

    if not (0 <= current_event_idx < MAX_EVENTS_AVAILABLE):
        err = f"Event index {current_event_idx} out of range (0-{max(0,MAX_EVENTS_AVAILABLE-1)})."; error_messages.append(err)
        fig_3d.update_layout(title=f"Error: {err}")
        return fig_3d, xtrack_fig, xcluster_fig, " ".join(error_messages), model_output_msg, [], []

    mlpf_predictions: Optional[MLPFOutput] = None
    attention_data: List[Tuple[str, NpArrayFloat]] = []
    embedding_data: List[Tuple[str, NpArrayFloat]] = [] # For new embeddings

    if mlpf_model_global and model_kwargs_global and not (parquet_data_global is None) and 'X_track' in parquet_data_global.fields and 'X_cluster' in parquet_data_global.fields and 0 <= current_event_idx < max_events_pq_global:
        try:
            x_track_event_ak = parquet_data_global['X_track'][current_event_idx]; x_cluster_event_ak = parquet_data_global['X_cluster'][current_event_idx]
            x_track_event_np = awkward.to_numpy(x_track_event_ak) if len(x_track_event_ak) > 0 else np.empty((0,0))
            x_cluster_event_np = awkward.to_numpy(x_cluster_event_ak) if len(x_cluster_event_ak) > 0 else np.empty((0,0))
            if x_track_event_np.ndim < 2 and x_track_event_np.shape != (0,0): x_track_event_np = np.atleast_2d(x_track_event_np)
            if x_cluster_event_np.ndim < 2 and x_cluster_event_np.shape != (0,0): x_cluster_event_np = np.atleast_2d(x_cluster_event_np)

            mlpf_predictions, inference_msg, attention_data, embedding_data = run_mlpf_inference( # Unpack new embedding_data
                mlpf_model_global, model_kwargs_global, x_track_event_np, x_cluster_event_np, config.TORCH_DEVICE
            )
            model_output_msg = f"(Event {current_event_idx}): {inference_msg}"
            if mlpf_predictions is None: error_messages.append(f"MLPF Inference: {inference_msg}")
        except Exception as e_infer: error_messages.append(f"Error during MLPF model inference: {str(e_infer)}"); model_output_msg = f"MLPF Model Inference Failed (Event {current_event_idx}): {str(e_infer)}"
    elif mlpf_model_global is None: model_output_msg = "MLPF Model not loaded. Cannot run inference."
    elif parquet_data_global is None or not ('X_track' in parquet_data_global.fields and 'X_cluster' in parquet_data_global.fields): model_output_msg = "Parquet data (X_track/X_cluster) not available for MLPF model."
    elif not (0 <= current_event_idx < max_events_pq_global): model_output_msg = f"Event {current_event_idx} out of range for Parquet data. MLPF model not run."

    # --- Process Attention Heatmaps ---
    if attention_data:
        attention_figures_children.append(html.H3("Attention Heatmaps", style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}))
        conv_id_attentions = [(name, matrix) for name, matrix in attention_data if name.startswith("conv_id")]
        conv_reg_attentions = [(name, matrix) for name, matrix in attention_data if name.startswith("conv_reg")]
        processed_any_attention_group = False
        if conv_id_attentions:
            processed_any_attention_group = True
            attention_figures_children.append(html.H4("Convolutional Identification Layers (conv_id)", style={'textAlign': 'center', 'marginTop': '15px', 'marginBottom': '5px'}))
            row_children = []
            for name, attn_matrix in conv_id_attentions:
                title = f"Attention: {name} (Evt {current_event_idx})"
                attn_fig = create_attention_heatmap_figure(attn_matrix, title)
                graph_style = {'height': '40vh', 'minHeight': '300px'}
                container_style = {'flex': '1 1 auto', 'margin': '5px', 'minWidth': '300px', 'maxWidth': 'calc(33.33% - 10px)'}
                row_children.append(html.Div([dcc.Graph(figure=attn_fig, style=graph_style)], style=container_style))
            attention_figures_children.append(html.Div(row_children, style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'flex-start', 'marginBottom': '15px'}))
        if conv_reg_attentions:
            processed_any_attention_group = True
            attention_figures_children.append(html.H4("Convolutional Regression Layers (conv_reg)", style={'textAlign': 'center', 'marginTop': '15px', 'marginBottom': '5px'}))
            row_children = []
            for name, attn_matrix in conv_reg_attentions:
                title = f"Attention: {name} (Evt {current_event_idx})"
                attn_fig = create_attention_heatmap_figure(attn_matrix, title)
                graph_style = {'height': '40vh', 'minHeight': '300px'}
                container_style = {'flex': '1 1 auto', 'margin': '5px', 'minWidth': '300px', 'maxWidth': 'calc(33.33% - 10px)'}
                row_children.append(html.Div([dcc.Graph(figure=attn_fig, style=graph_style)], style=container_style))
            attention_figures_children.append(html.Div(row_children, style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'flex-start', 'marginBottom': '15px'}))
        if not processed_any_attention_group and attention_data:
             attention_figures_children.append(html.P("No 'conv_id' or 'conv_reg' specific attention data captured.", style={'textAlign': 'center'}))
    else:
        attention_figures_children.append(html.P("No attention data captured or available for this event.", style={'textAlign': 'center'}))

    # --- Process Embedding Heatmaps ---
    if embedding_data:
        embedding_figures_children.append(html.H3("Intermediate Embedding Heatmaps", style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}))
        embedding_row_children = []
        for name, emb_matrix in embedding_data:
            title = f"Embedding: {name} (Evt {current_event_idx})"
            emb_fig = create_embedding_heatmap_figure(emb_matrix, title)
            graph_style = {'height': '50vh', 'minHeight': '400px'} # Embeddings might be taller
            # Allow these to take more width if fewer of them
            container_style = {'flex': '1 1 auto', 'margin': '10px', 'minWidth': '400px', 'maxWidth': 'calc(50% - 20px)'}
            embedding_row_children.append(
                html.Div([dcc.Graph(figure=emb_fig, style=graph_style)], style=container_style)
            )
        embedding_figures_children.append(html.Div(embedding_row_children, style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'flex-start', 'marginBottom': '15px'}))
    else:
        embedding_figures_children.append(html.P("No intermediate embedding data captured or available for this event.", style={'textAlign': 'center'}))


    if parquet_data_global is not None and 0 <= current_event_idx < max_events_pq_global:
        try:
            if 'X_track' in parquet_data_global.fields and len(parquet_data_global['X_track']) > current_event_idx:
                x_track_data_ak = parquet_data_global['X_track'][current_event_idx]
                if len(x_track_data_ak) > 0:
                    x_track_np_hm = awkward.to_numpy(x_track_data_ak)
                    if x_track_np_hm.ndim > 0 and x_track_np_hm.shape[0] > 0:
                        means_trk = np.mean(x_track_np_hm, axis=0, keepdims=True); stds_trk = np.std(x_track_np_hm, axis=0, keepdims=True); stds_trk[stds_trk == 0] = 1
                        x_track_np_norm = (x_track_np_hm - means_trk) / stds_trk; x_track_np_norm[np.isnan(x_track_np_norm)] = 0
                        xtrack_fig = create_feature_heatmap(x_track_np_norm, config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx})", "Trk")
                    else: xtrack_fig = create_feature_heatmap(np.array([]), config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx}) - No Tracks", "Trk")
                else: xtrack_fig = create_feature_heatmap(np.array([]), config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx}) - No Tracks", "Trk")
            else: error_messages.append(f"X_track data not found for event {current_event_idx} in Parquet.")
            if 'X_cluster' in parquet_data_global.fields and len(parquet_data_global['X_cluster']) > current_event_idx:
                x_cluster_data_ak = parquet_data_global['X_cluster'][current_event_idx]
                if len(x_cluster_data_ak) > 0:
                    x_cluster_np_hm = awkward.to_numpy(x_cluster_data_ak)
                    if x_cluster_np_hm.ndim > 0 and x_cluster_np_hm.shape[0] > 0:
                        means_cls = np.mean(x_cluster_np_hm, axis=0, keepdims=True); stds_cls = np.std(x_cluster_np_hm, axis=0, keepdims=True); stds_cls[stds_cls == 0] = 1
                        x_cluster_np_norm = (x_cluster_np_hm - means_cls) / stds_cls; x_cluster_np_norm[np.isnan(x_cluster_np_norm)] = 0
                        xcluster_fig = create_feature_heatmap(x_cluster_np_norm, config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx})", "Cls", colorscale='Plasma')
                    else: xcluster_fig = create_feature_heatmap(np.array([]), config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx}) - No Clusters", "Cls", colorscale='Plasma')
                else: xcluster_fig = create_feature_heatmap(np.array([]), config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx}) - No Clusters", "Cls", colorscale='Plasma')
            else: error_messages.append(f"X_cluster data not found for event {current_event_idx} in Parquet.")
        except Exception as e_pq_vis: error_messages.append(f"Error processing Parquet data for heatmaps: {str(e_pq_vis)}")
    elif parquet_data_global is None: error_messages.append("Parquet data not loaded. Heatmaps unavailable.")
    elif not (0 <= current_event_idx < max_events_pq_global): error_messages.append(f"Event index {current_event_idx} out of range for Parquet data. Heatmaps unavailable.")

    fig_traces_3d: List[go.Scatter3d] = []; root_data_issue_for_3d = False
    if event_tree_global is None or max_events_root_global == 0: error_messages.append("ROOT data not loaded or no events. 3D display unavailable."); root_data_issue_for_3d = True
    elif not (0 <= current_event_idx < max_events_root_global): error_messages.append(f"Event index {current_event_idx} out of range for ROOT data. 3D display unavailable."); root_data_issue_for_3d = True
    else:
        if show_hits:
            all_hits_features_list: List[AkArray] = []
            if collectionIDs_reverse_global:
                for coll_name_from_meta in collectionIDs_reverse_global.values():
                    if coll_name_from_meta in config.HIT_COLLECTIONS_TO_PLOT and coll_name_from_meta in event_tree_global:
                        hit_features = extract_hit_features(event_tree_global, current_event_idx, coll_name_from_meta, config.HIT_FEATURES_STD)
                        if len(hit_features.fields) > 0 and 'position.x' in hit_features.fields and len(hit_features['position.x']) > 0: all_hits_features_list.append(hit_features)
            if all_hits_features_list:
                try:
                    hit_feature_matrix = awkward.concatenate(all_hits_features_list, axis=0)
                    if len(hit_feature_matrix) > 0 and all(f in hit_feature_matrix.fields for f in ['position.x', 'position.y', 'position.z', 'energy', 'subdetector']):
                        df_hits = awkward.to_dataframe(hit_feature_matrix[['position.x', 'position.y', 'position.z', 'energy', 'subdetector']]); df_hits.rename(columns={'position.x': 'px', 'position.y': 'py', 'position.z': 'pz'}, inplace=True)
                        df_hits["energy"] = 1000 * df_hits["energy"]; df_hits["plotsize"] = 2.0
                        df_hits.loc[df_hits["subdetector"]==0, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==0, "energy"]/5), 1, 10)
                        df_hits.loc[df_hits["subdetector"]==1, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==1, "energy"]/10), 1, 10)
                        df_hits.loc[df_hits["subdetector"]==2, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==2, "energy"]*100), 1, 10)
                        df_hits.loc[df_hits["subdetector"]==3, "plotsize"] = 3.0
                        for subdet_idx in df_hits["subdetector"].unique():
                            sub_df = df_hits[df_hits["subdetector"] == subdet_idx]
                            fig_traces_3d.append(go.Scatter3d(x=np.clip(sub_df["px"], -4500, 4500), y=np.clip(sub_df["py"], -4500, 4500), z=np.clip(sub_df["pz"], -4500, 4500), mode='markers', marker=dict(size=sub_df["plotsize"], color=config.HIT_SUBDETECTOR_COLOR.get(subdet_idx, "grey"), opacity=0.7), name=config.HIT_LABELS.get(subdet_idx, "Unknown Hit")))
                    else: error_messages.append("Concatenated hit data is empty or missing essential fields.")
                except Exception as e_concat_hits: error_messages.append(f"Error processing/plotting ROOT hits: {e_concat_hits}")

        if show_tracks:
            track_px_all, track_py_all, track_pz_all, track_charge_all, track_pt_all = extract_track_features(event_tree_global, current_event_idx, config.DEFAULT_TRACK_COLLECTION_NAME, config.DEFAULT_TRACK_STATE_BRANCH_NAME, b_field_val)
            if track_pt_all.size > 0:
                # pt_mask = track_pt_all >= min_pt_cut_val # Removed pt cut
                # track_px, track_py, track_pz = track_px_all[pt_mask], track_py_all[pt_mask], track_pz_all[pt_mask]; track_charge = track_charge_all[pt_mask]
                track_px, track_py, track_pz, track_charge = track_px_all, track_py_all, track_pz_all, track_charge_all # Use all tracks
                if track_px.size > 0:
                    try:
                        hx, hy, hz = generate_helix_trace_data(track_px, track_py, track_pz, track_charge, config.PION_MASS_GEV, b_field_val, scale_factor_val)
                        if hx: fig_traces_3d.append(go.Scatter3d(x=hx, y=hy, z=hz, mode='lines', line=dict(color="purple", width=3), name=f"SiTracks")) # Removed pT cut from name
                    except Exception as e_helix: error_messages.append(f"Error generating track helices: {e_helix}")
                # elif track_px_all.size > 0 : error_messages.append(f"No tracks with pT >= {min_pt_cut_val:.1f} GeV.") # Commented out as pT cut removed
            elif config.DEFAULT_TRACK_COLLECTION_NAME in event_tree_global: error_messages.append("No tracks processed or found for this event from ROOT.")

        if show_pandora_clusters:
            df_pandora = extract_pandora_cluster_features(event_tree_global, current_event_idx, config.PANDORA_CLUSTER_COLLECTION_NAME, config.PANDORA_CLUSTER_FEATURES)
            if df_pandora is not None and not df_pandora.empty:
                if all(col in df_pandora for col in ["x", "y", "z", "energy"]):
                    df_pandora["plotsize"] = np.clip(2 + 2 * np.log1p(df_pandora["energy"]), 1, 12)
                    hover_texts_pandora = [f"E: {df_pandora['energy'][i]:.2f} GeV" + (f"<br>Type: {int(df_pandora['type'][i])}" if 'type' in df_pandora and pd.notna(df_pandora['type'][i]) else "") + (f"<br>E_err: {df_pandora['energyError'][i]:.2f} GeV" if 'energyError' in df_pandora and pd.notna(df_pandora['energyError'][i]) else "") for i in df_pandora.index]
                    fig_traces_3d.append(go.Scatter3d(x=df_pandora["x"], y=df_pandora["y"], z=df_pandora["z"], mode='markers', marker=dict(size=df_pandora["plotsize"], color=config.PANDORA_CLUSTER_COLOR, opacity=0.6), name="Pandora Clusters", text=hover_texts_pandora, hoverinfo="text+name"))
                else: error_messages.append("Essential Pandora Cluster data (pos/energy) missing.")
            elif df_pandora is None: error_messages.append(f"'{config.PANDORA_CLUSTER_COLLECTION_NAME}' not found or error during extraction.")
    fig_3d = create_3d_event_display_figure(fig_traces_3d, current_event_idx, b_field_val, scale_factor_val, root_data_issue_for_3d) # Removed min_pt_cut_val
    final_error_message = " | ".join(msg for msg in error_messages if msg)
    return fig_3d, xtrack_fig, xcluster_fig, final_error_message, model_output_msg, attention_figures_children, embedding_figures_children

if __name__ == '__main__':
    initialize_global_data()
    app.layout = create_app_layout()
    if MAX_EVENTS_AVAILABLE == 0:
        print("--------------------------------------------------------------------------")
        print("Failed to load data or no events found in either ROOT or Parquet file.")
        print(f"Please check ROOT_FILE_PATH: '{config.ROOT_FILE_PATH}' and PARQUET_FILE_PATH: '{config.PARQUET_FILE_PATH}'.")
        print("The Dash app will start but may not be functional.")
        print("--------------------------------------------------------------------------")
    else:
        print(f"Successfully loaded data. Max events available for display: {MAX_EVENTS_AVAILABLE}")
        if max_events_root_global > 0 and collectionIDs_global: print(f"ROOT Collections found: {list(collectionIDs_global.keys())}")
        elif max_events_root_global > 0: print("ROOT Metadata for CollectionIDs not found or empty.")
        if parquet_data_global is not None: print(f"Parquet data fields: {parquet_data_global.fields}")
    app.run(debug=True)