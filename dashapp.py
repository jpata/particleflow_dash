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

# Assuming mlpf library is correctly installed and structured
import mlpf
from mlpf.model.mlpf import MLPF # Assuming MLPF is the correct class name
from mlpf.model.utils import unpack_predictions, unpack_target # Keep if used elsewhere, not directly in this refactored snippet
from mlpf.jet_utils import match_jets, to_p4_sph # Keep if used elsewhere
from mlpf.plotting.plot_utils import cms_label, sample_label # Keep if used elsewhere

# --- Type Aliases ---
NpArrayFloat = np.typing.NDArray[np.float64]
NpArrayInt = np.typing.NDArray[np.int64]
NpArrayBool = np.typing.NDArray[np.bool]
AkArray = awkward.Array # General awkward array

# --- Configuration ---
# Using a SimpleNamespace for cleaner access to config parameters
config = SimpleNamespace(
    ROOT_FILE_PATH="data/reco_p8_ee_tt_ecm380_1.root",
    PARQUET_FILE_PATH="data/reco_p8_ee_tt_ecm380_1.parquet",
    DEFAULT_B_FIELD_TESLA=-4.0,
    DEFAULT_SCALE_FACTOR=1000.0,
    DEFAULT_EVENT_INDEX=0,
    DEFAULT_PT_CUT_GEV=1.0,
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
    DEFAULT_SHOW_PANDORA_CLUSTERS=True,
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


# --- Utility Functions ---
def pad_and_concatenate_arrays(arr1: NpArrayFloat, arr2: NpArrayFloat) -> NpArrayFloat:
    """Pads two 2D numpy arrays to have the same number of columns and concatenates them along rows."""
    if arr1.ndim != 2 and arr1.shape[0] != 0 : arr1 = arr1.reshape(0,0) # Handle 0-element case gracefully
    if arr2.ndim != 2 and arr2.shape[0] != 0 : arr2 = arr2.reshape(0,0)

    shape1_cols = arr1.shape[1] if arr1.shape[0] > 0 else 0
    shape2_cols = arr2.shape[1] if arr2.shape[0] > 0 else 0
    
    if arr1.shape[0] == 0 and arr2.shape[0] == 0:
        # If both are empty, decide on a common number of columns (e.g., from model_input_dim if available or 0)
        # For now, assuming if both are empty, the target shape will handle it or error out later if dim mismatch
        return np.empty((0, max(shape1_cols, shape2_cols)))


    max_cols = max(shape1_cols, shape2_cols)

    def pad_array(arr: NpArrayFloat, target_cols: int) -> NpArrayFloat:
        if arr.shape[0] == 0: # If array is empty (0 rows)
            return np.empty((0, target_cols))
        if arr.shape[1] < target_cols:
            padding = ((0, 0), (0, target_cols - arr.shape[1]))
            return np.pad(arr, padding, mode='constant', constant_values=0)
        return arr

    arr1_padded = pad_array(arr1, max_cols)
    arr2_padded = pad_array(arr2, max_cols)

    return np.concatenate((arr1_padded, arr2_padded), axis=0)

def calculate_track_pt(omega_data: AkArray, b_field_tesla: float) -> AkArray:
    """Calculates track pT from omega and B-field."""
    omega_np = awkward.to_numpy(omega_data)
    pt_np = np.zeros_like(omega_np, dtype=float)
    non_zero_mask = omega_np != 0
    # Ensure b_field_tesla is non-zero to avoid division by zero if omega is also non-zero
    if abs(b_field_tesla) < 1e-9: # Effectively zero B-field
         pt_np[non_zero_mask] = np.inf # Or handle as an error/very large pT
    else:
        pt_np[non_zero_mask] = (3e-4 * np.abs(b_field_tesla) / np.abs(omega_np[non_zero_mask]))
    pt_np[~non_zero_mask] = 0
    return awkward.from_numpy(pt_np)

# --- Data Loading Functions ---
LoadedRootData = Tuple[Optional[uproot.ReadOnlyDirectory], Optional[uproot.TTree], Dict[str, int], Dict[int, str], int]
def load_root_data(file_path: str) -> LoadedRootData:
    """Loads data from a ROOT file."""
    fi = None
    ev_tree = None
    collectionIDs: Dict[str, int] = {}
    collectionIDs_reverse: Dict[int, str] = {}
    max_events_root = 0
    try:
        fi = uproot.open(file_path)
        ev_tree = fi.get("events") # Use .get for safer access
        metadata_tree = fi.get("metadata")

        if metadata_tree:
            collectionIDs_data = metadata_tree.arrays("CollectionIDs")
            if "CollectionIDs" in collectionIDs_data.fields and \
               "m_names" in collectionIDs_data["CollectionIDs"].fields and \
               "m_collectionIDs" in collectionIDs_data["CollectionIDs"].fields and \
               len(collectionIDs_data["CollectionIDs"]["m_names"]) > 0: # Check if list is not empty
                names = collectionIDs_data["CollectionIDs"]["m_names"][0]
                ids = collectionIDs_data["CollectionIDs"]["m_collectionIDs"][0]
                collectionIDs = {k: v for k, v in zip(names, ids)}
                collectionIDs_reverse = {v: k for k, v in collectionIDs.items()}
            else:
                print("Warning: 'CollectionIDs' structure in 'metadata' tree is not as expected or empty.")
        else:
            print("Warning: 'metadata' tree not found in ROOT file. CollectionIDs will be empty.")

        if ev_tree:
            max_events_root = int(ev_tree.num_entries)
        else:
            print("Warning: 'events' tree not found in ROOT file.")

    except FileNotFoundError:
        print(f"ERROR: ROOT file not found at {file_path}. Please update path.")
        return None, None, {}, {}, 0
    except Exception as e:
        print(f"Error loading ROOT file or metadata from {file_path}: {e}")
        return None, None, {}, {}, 0
    return fi, ev_tree, collectionIDs, collectionIDs_reverse, max_events_root

LoadedParquetData = Tuple[Optional[AkArray], int]
def load_parquet_data(file_path: str) -> LoadedParquetData:
    """Loads data from a Parquet file."""
    pq_data = None
    max_events_pq = 0
    try:
        pq_data = awkward.from_parquet(file_path)
        if 'X_track' in pq_data.fields:
            max_events_pq = len(pq_data['X_track'])
            print(f"Successfully loaded Parquet file: {file_path}. Max events: {max_events_pq}")
        else:
            print(f"Warning: Parquet file {file_path} loaded, but 'X_track' field is missing.")
            pq_data = None # Invalidate if key data is missing
    except FileNotFoundError:
        print(f"ERROR: Parquet file not found at {file_path}. Heatmaps will be empty.")
    except Exception as e:
        print(f"Error loading Parquet file {file_path}: {e}. Heatmaps will be empty.")
    return pq_data, max_events_pq

LoadedMLPFModel = Tuple[Optional[MLPF], Optional[Dict[str, Any]]]
def load_mlpf_model(kwargs_path: str, checkpoint_path: str, device: torch.device) -> LoadedMLPFModel:
    """Loads the MLPF model and its keyword arguments."""
    model_kwargs = None
    model_instance = None
    try:
        if not os.path.exists(kwargs_path):
            print(f"ERROR: Model kwargs file not found at {kwargs_path}")
            return None, None
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Model checkpoint file not found at {checkpoint_path}")
            return None, None

        with open(kwargs_path, "rb") as f:
            model_kwargs = pkl.load(f)
            print("Successfully loaded model_kwargs.pkl")

        if not isinstance(model_kwargs, dict): # Basic check
            print("ERROR: model_kwargs.pkl did not contain a dictionary.")
            return None, None

        model_instance = MLPF(**model_kwargs) # Make sure MLPF class is correctly defined/imported
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = {}
        if "model_state_dict" in checkpoint: model_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint: model_state = checkpoint["state_dict"]
        else: model_state = checkpoint
        
        # model_state = {k.replace('module.', ''): v for k, v in model_state.items()} # If needed

        model_instance.load_state_dict(model_state)
        model_instance.eval()
        model_instance.to(device)

        # Disable attention context manager (if applicable)
        if hasattr(model_instance, 'conv_id') and hasattr(model_instance, 'conv_reg'):
            for conv_list_attr in ['conv_id', 'conv_reg']:
                conv_list = getattr(model_instance, conv_list_attr)
                if conv_list is not None:
                    for conv_layer in conv_list:
                        if hasattr(conv_layer, 'enable_ctx_manager'):
                            conv_layer.enable_ctx_manager = False
        print(f"MLPF model loaded from {checkpoint_path} and configured for inference on {device}.")
        return model_instance, model_kwargs

    except FileNotFoundError as e_fnf:
        print(f"ERROR: Model file not found. {e_fnf}.")
    except Exception as e_model_load:
        print(f"Error loading MLPF model: {e_model_load}")
    return None, None


# --- Feature Extraction Functions ---
def extract_hit_features(
    event_tree: uproot.TTree,
    event_index: int,
    collection_name: str,
    features_to_extract: List[str]
) -> AkArray:
    """Extracts hit features for a given collection and event index."""
    feat_arr_for_event: Dict[str, AkArray] = {}
    feature_map: List[Tuple[str, str]] = []
    is_tracker_hit = "TrackerHit" in collection_name

    for feat_key in features_to_extract:
        file_feat_name = "eDep" if is_tracker_hit and feat_key == "energy" else feat_key
        feature_map.append((feat_key, file_feat_name))

    branches_to_load = [f"{collection_name}.{ff[1]}" for ff in feature_map]
    
    # Check if essential position branches exist even before trying to load
    # This check might need refinement based on actual tree structure if position is nested
    # For simplicity, assuming flat branch names like "collection_name.position.x"
    missing_pos_branches = any(
        b for b in branches_to_load if b.endswith((".position.x", ".position.y", ".position.z")) and b not in event_tree
    )
    if missing_pos_branches:
         # print(f"Warning: Essential position branches for {collection_name} missing in TTree. Skipping.")
         return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})


    try:
        event_collection_data = event_tree.arrays(
            branches_to_load, entry_start=event_index, entry_stop=event_index + 1, library="ak"
        )
    except Exception as e:
        # print(f"Warning: Error loading hit data for {collection_name}, event {event_index}: {e}")
        return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})

    # Check if any data was actually found for the event
    found_data_for_event = any(
        branch_name_key in event_collection_data.fields and len(event_collection_data[branch_name_key][0]) > 0
        for branch_name_key in branches_to_load if branch_name_key in event_collection_data.fields # Check field exists before len
    )
    if not found_data_for_event:
        return awkward.Array({f_orig: awkward.Array([]) for f_orig, _ in feature_map} | {"subdetector": awkward.Array([])})

    num_hits = 0
    first_branch_key = f"{collection_name}.{feature_map[0][1]}"
    if first_branch_key in event_collection_data.fields:
        # Ensure the structure is as expected: event_collection_data[branch][event_idx_in_batch]
        if len(event_collection_data[first_branch_key]) > 0:
             num_hits = len(event_collection_data[first_branch_key][0])


    for original_feat_name, feat_name_in_file in feature_map:
        branch_full_name_key = f"{collection_name}.{feat_name_in_file}"
        if branch_full_name_key in event_collection_data.fields and len(event_collection_data[branch_full_name_key]) > 0:
            feat_arr_for_event[original_feat_name] = event_collection_data[branch_full_name_key][0]
        else:
            feat_arr_for_event[original_feat_name] = awkward.from_numpy(np.zeros(num_hits)) # Use from_numpy for explicit ak array

    subdetector_val = 3 # Default to tracker
    if collection_name.startswith("ECAL"): subdetector_val = 0
    elif collection_name.startswith("HCAL"): subdetector_val = 1
    elif collection_name.startswith("MUON"): subdetector_val = 2
    feat_arr_for_event["subdetector"] = awkward.from_numpy(np.full(num_hits, subdetector_val, dtype=np.int32))

    # Ensure all arrays have the same length (truncate to minimum if necessary)
    if feat_arr_for_event:
        lengths = [len(arr) for arr in feat_arr_for_event.values()]
        if len(set(lengths)) > 1:
            # print(f"Warning: Inconsistent hit feature lengths for {collection_name}, event {event_index}. Min length used.")
            min_len = min(l for l in lengths if l > 0) if any(l > 0 for l in lengths) else 0
            for k_f in feat_arr_for_event:
                feat_arr_for_event[k_f] = feat_arr_for_event[k_f][:min_len]
    
    return awkward.Array(feat_arr_for_event)


TrackFeatures = Tuple[NpArrayFloat, NpArrayFloat, NpArrayFloat, NpArrayInt, NpArrayFloat] # px, py, pz, charge, pt
def extract_track_features(
    event_tree: uproot.TTree,
    event_index: int,
    track_collection: str,
    track_state_branch: str,
    b_field_tesla: float
) -> TrackFeatures:
    """Extracts track features (px, py, pz, charge, pt) for an event."""
    empty_result: TrackFeatures = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    track_base_feats = ["type", "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit"]
    track_state_params = ["tanLambda", "D0", "phi", "omega", "Z0", "time"]

    track_branches = [f"{track_collection}/{track_collection}.{feat}" for feat in track_base_feats]
    track_branches.append(f"{track_collection}/{track_collection}.trackStates_begin") # Index to track states
    state_branches = [f"{track_state_branch}/{track_state_branch}.{param}" for param in track_state_params]
    
    all_needed_branches = track_branches + state_branches
    # More robust check for branch existence
    available_branches = set(event_tree.keys(filter_name=f"{track_collection}/*") + \
                             event_tree.keys(filter_name=f"{track_state_branch}/*"))
    
    missing_tree_branches = [b for b in all_needed_branches if b not in available_branches and not b.startswith(track_state_branch+"/") ] # Simpler check for now
    # A more precise check would be needed if branch names aren't exactly as constructed
    # For track_state_branch, the full path is constructed, so direct check is okay

    if any(b.startswith(f"{track_collection}.") and b not in available_branches for b in track_branches) or \
       any(b not in available_branches for b in state_branches) :
        print(f"Warning: For tracks, event {event_index}, critical branches missing. Needed: {all_needed_branches}, Available: {available_branches}")
        return empty_result

    try:
        event_track_data = event_tree.arrays(
            all_needed_branches, entry_start=event_index, entry_stop=event_index + 1, library="ak"
        )
    except Exception as e:
        print(f"Error loading track data for event {event_index}: {e}")
        return empty_result

    base_type_branch_key = f"{track_collection}/{track_collection}.type"
    if not (base_type_branch_key in event_track_data.fields and len(event_track_data[base_type_branch_key][0]) > 0):
        return empty_result

    track_features_ak: Dict[str, AkArray] = {}
    for feat in track_base_feats:
        branch_key = f"{track_collection}/{track_collection}.{feat}"
        if branch_key in event_track_data.fields and len(event_track_data[branch_key]) > 0 :
             track_features_ak[feat] = event_track_data[branch_key][0]
        else: # Should not happen if initial branch check passed, but as safeguard
             return empty_result


    num_tracks = len(track_features_ak["type"])
    if num_tracks == 0:
        return empty_result

    track_state_indices_branch_key = f"{track_collection}/{track_collection}.trackStates_begin"
    if track_state_indices_branch_key not in event_track_data.fields or len(event_track_data[track_state_indices_branch_key]) == 0 :
        return empty_result # No indices means no states
        
    track_state_indices = event_track_data[track_state_indices_branch_key][0]


    if len(track_state_indices) != num_tracks:
        # print(f"Warning: Mismatch N_tracks ({num_tracks}) vs N_trackStateIndices ({len(track_state_indices)}) in event {event_index}.")
        return empty_result
    
    if len(track_state_indices) > 0 and awkward.any(track_state_indices < 0):
        # print(f"Warning: Negative trackStateIndices found in event {event_index}.")
        return empty_result

    for param_name in track_state_params:
        param_branch_key = f"{track_state_branch}/{track_state_branch}.{param_name}"
        if param_branch_key not in event_track_data.fields or len(event_track_data[param_branch_key]) == 0:
            # print(f"Error: Track state parameter branch {param_branch_key} not found or empty for event {event_index}.")
            return empty_result
            
        all_states_for_param = event_track_data[param_branch_key][0]
        
        if len(track_state_indices) == 0: # No tracks, so no states to pick
            track_features_ak[param_name] = awkward.Array([])
        elif awkward.any(track_state_indices >= len(all_states_for_param)): # Check for out-of-bounds
            # print(f"Error: Track state index out of bounds for param {param_name}, event {event_index}.")
            return empty_result
        else:
            track_features_ak[param_name] = all_states_for_param[track_state_indices]
            
    track_features_ak["pt"] = calculate_track_pt(track_features_ak["omega"], b_field_tesla)

    # Convert to NumPy for calculations
    phi_np: NpArrayFloat = awkward.to_numpy(track_features_ak["phi"])
    pt_np: NpArrayFloat = awkward.to_numpy(track_features_ak["pt"])
    tanLambda_np: NpArrayFloat = awkward.to_numpy(track_features_ak["tanLambda"])
    omega_np: NpArrayFloat = awkward.to_numpy(track_features_ak["omega"])

    px_np = np.cos(phi_np) * pt_np
    py_np = np.sin(phi_np) * pt_np
    pz_np = tanLambda_np * pt_np
    charge_np = np.sign(omega_np).astype(int)
    
    return px_np, py_np, pz_np, charge_np, pt_np

def extract_pandora_cluster_features(
    event_tree: uproot.TTree,
    event_index: int,
    collection_name: str,
    features_to_extract: List[str]
) -> Optional[pd.DataFrame]:
    """Extracts Pandora cluster features into a Pandas DataFrame."""
    if collection_name not in event_tree:
        # print(f"Warning: Pandora collection '{collection_name}' not in TTree for event {event_index}.")
        return None

    pandora_branches_to_request = [
        f"{collection_name}/{collection_name}.{feat_name}" for feat_name in features_to_extract
    ]
    
    # Check availability of branches
    available_branches_in_tree = [
        b for b in pandora_branches_to_request if b in event_tree.keys(filter_name=f"{collection_name}/*")
    ]
    if not available_branches_in_tree:
        # print(f"Warning: No Pandora cluster branches found for {collection_name} in event {event_index}.")
        return None

    try:
        pandora_event_data = event_tree.arrays(
            available_branches_in_tree,
            entry_start=event_index,
            entry_stop=event_index + 1,
            library="ak"
        )
    except Exception as e:
        # print(f"Error loading Pandora cluster data for {collection_name}, event {event_index}: {e}")
        return None

    num_clusters = 0
    # Determine num_clusters from the first successfully loaded branch from the request list
    # (assuming all cluster feature arrays for an event have the same length)
    first_valid_branch = next((b for b in pandora_branches_to_request if b in pandora_event_data.fields and len(pandora_event_data[b]) > 0), None)

    if first_valid_branch:
        num_clusters = len(pandora_event_data[first_valid_branch][0])

    if num_clusters == 0:
        return pd.DataFrame() # Return empty DataFrame if no clusters

    df_pandora_dict: Dict[str, NpArrayFloat] = {}
    all_essential_cols_present = True

    # Map from config feature names to DataFrame column names
    pandora_feature_to_df_col_map = {
        "type": "type", "energy": "energy", "energyError": "energyError",
        "position.x": "x", "position.y": "y", "position.z": "z"
    }

    for original_feat_name in features_to_extract:
        branch_key_to_check = f"{collection_name}/{collection_name}.{original_feat_name}"
        df_col_name = pandora_feature_to_df_col_map.get(original_feat_name, original_feat_name) # Default to original_feat_name if not in map

        if branch_key_to_check in pandora_event_data.fields and len(pandora_event_data[branch_key_to_check]) > 0:
            df_pandora_dict[df_col_name] = awkward.to_numpy(pandora_event_data[branch_key_to_check][0])
        else:
            df_pandora_dict[df_col_name] = np.zeros(num_clusters)
            if df_col_name in ["x", "y", "z", "energy"]: # Check essential columns for plotting
                all_essential_cols_present = False
                # print(f"Warning: Pandora branch {branch_key_to_check} missing/empty for event {event_index}, df column '{df_col_name}' filled with zeros.")
    
    if not all_essential_cols_present:
        # print(f"Warning: Essential Pandora data (pos/energy) missing for event {event_index}. Plotting might be affected.")
        # Depending on strictness, could return None here
        pass

    return pd.DataFrame(df_pandora_dict)


# --- ML Model Inference ---
MLPFOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] # binary_particle, pid, momentum, pu
def run_mlpf_inference(
    model: MLPF,
    model_kwargs: Dict[str, Any],
    x_track_event_np: NpArrayFloat, # Expect numpy array here
    x_cluster_event_np: NpArrayFloat, # Expect numpy array here
    device: torch.device
) -> Tuple[Optional[MLPFOutput], str]:
    """Runs MLPF model inference for a single event."""
    
    import pdb;pdb.set_trace()
    model_input_dim = model_kwargs.get("input_dim")
    if model_input_dim is None:
        return None, "input_dim not found in MODEL_KWARGS"

    # Ensure inputs are 2D and correctly shaped for slicing/padding
    def _ensure_2d_shape(arr: NpArrayFloat, target_dim: int) -> NpArrayFloat:
        if arr.ndim == 0: # scalar, shouldn't happen with typical feature arrays
            return np.empty((0, target_dim))
        if arr.ndim == 1:
            if arr.shape[0] == target_dim : # A single element with all features
                 return np.expand_dims(arr, axis=0)
            elif arr.shape[0] == 0 : # An empty 1D array
                 return np.empty((0, target_dim))
            else: # Malformed single element
                 return np.empty((0, target_dim)) # Or raise error
        # arr is already 2D or more
        if arr.shape[0] == 0: # (0, N)
            return np.empty((0, target_dim))
        return arr[..., :target_dim] # Take up to model_input_dim features

    x_track_np = _ensure_2d_shape(x_track_event_np, model_input_dim)
    x_cluster_np = _ensure_2d_shape(x_cluster_event_np, model_input_dim)

    if x_track_np.shape[0] == 0 and x_cluster_np.shape[0] == 0:
        return None, "No track or cluster features to run MLPF model."

    # Pad and concatenate
    # The pad_and_concatenate_arrays function expects arrays to have at least one dimension
    # If one is (0, N) and other is (M, N), it should work.
    # If one is (0,0) due to no features, need careful handling inside pad_and_concatenate_arrays
    # or ensure model_input_dim is used to shape empty arrays.
    
    # Adjusting for pad_and_concatenate_arrays which expects at least 2D arrays
    # If an array is (0, F), it's fine. If it was (0,), it became (0,F) in _ensure_2d_shape.
    combined_features_np = pad_and_concatenate_arrays(x_track_np, x_cluster_np)

    if combined_features_np.shape[0] == 0:
        return None, "No combined features after processing for MLPF model."
    
    # The model might expect a specific number of columns after padding,
    # which pad_and_concatenate_arrays handles by padding to max of the two.
    # If model_input_dim is strictly required for combined_features_np's width:
    if combined_features_np.shape[1] != model_input_dim and combined_features_np.shape[0] > 0:
        # This case implies that one of the input arrays (track/cluster)
        # had MORE features than model_input_dim, and pad_and_concatenate_arrays
        # padded to that wider dimension. This shouldn't happen if _ensure_2d_shape
        # correctly sliced to model_input_dim.
        # OR, if one array was empty and the other was narrower than model_input_dim,
        # it would be padded to model_input_dim by _ensure_2d_shape,
        # but pad_and_concatenate might not enforce model_input_dim if it wasn't the max.
        # Let's assume _ensure_2d_shape and pad_and_concatenate work as intended:
        # all inputs to pad_and_concatenate are already sliced to model_input_dim width.
        pass


    model_input_tensor = torch.tensor(combined_features_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Mask based on the first feature (assuming non-zero for valid elements)
    # This needs to be robust if the first feature *can* be zero for valid elements.
    # The original code uses X_features_padded[:, :, 0]!=0.
    # Here, combined_features_np is already the data.
    if model_input_tensor.shape[2] == 0: # No features in tensor
        return None, "Combined features tensor has 0 features, cannot create mask."

    mask = (model_input_tensor[:, :, 0] != 0).to(device)

    with torch.no_grad():
        preds = model(model_input_tensor, mask) # Assuming model signature is (input_tensor, mask)
    
    # Assuming preds is a tuple as unpacked in original code
    preds_binary_particle, preds_pid, preds_momentum, preds_pu = preds
    
    output_msg = (
        f"MLPF Model Inference Output:\n"
        f"  preds_binary_particle shape: {preds_binary_particle.shape}\n"
        f"  preds_pid shape: {preds_pid.shape}\n"
        f"  preds_momentum shape: {preds_momentum.shape}\n"
        f"  preds_pu shape: {preds_pu.shape}\n"
        f"  Number of input elements to model: {combined_features_np.shape[0]}"
    )
    return preds, output_msg


# --- Plotting Helper Functions ---
def create_empty_heatmap_figure(title: str = "No Data / Error") -> go.Figure:
    """Creates an empty Plotly figure for heatmaps with a message."""
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title="Features", yaxis_title="Index")
    fig.add_annotation(text="Data unavailable or error in loading.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig

def create_feature_heatmap(
    data_np: NpArrayFloat, # Expect 2D numpy array
    feature_order: List[str],
    title: str,
    y_axis_prefix: str,
    colorscale: str = 'Viridis'
) -> go.Figure:
    """Creates a heatmap figure for given feature data."""
    if data_np.ndim == 1: # Single element (e.g. one track/cluster)
        data_np = np.expand_dims(data_np, axis=0)
    
    if data_np.shape[0] == 0: # No elements
        fig = go.Figure().update_layout(title=f"{title} - No Data", xaxis_title="Features", yaxis_title=f"{y_axis_prefix} Index")
        fig.add_annotation(text=f"No {y_axis_prefix.lower()} data for this event.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    if data_np.shape[1] != len(feature_order):
        error_title = f"{title} - Feature Mismatch (Data: {data_np.shape[1]}, Expected: {len(feature_order)})"
        # print(f"Error for {title}: Feature count mismatch. Data has {data_np.shape[1]} features, expected {len(feature_order)}.")
        return create_empty_heatmap_figure(error_title)

    fig = go.Figure(data=go.Heatmap(
        z=data_np,
        x=feature_order,
        y=[f'{y_axis_prefix} {i}' for i in range(data_np.shape[0])],
        colorscale=colorscale,
        colorbar=dict(title='Normalized Value'), # Assuming normalized
        xgap=1, ygap=0
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Features", yaxis_title=f"{y_axis_prefix} Index",
        xaxis_tickangle=-45,
        margin=dict(l=70, r=20, b=120, t=50) # Adjusted for long feature names
    )
    return fig

HelixTraceData = Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]
def generate_helix_trace_data(
    px_values: NpArrayFloat, py_values: NpArrayFloat, pz_values: NpArrayFloat,
    charge_values: NpArrayInt, particle_mass: float,
    b_field: float, scale_factor: float, num_points_helix: int = 70, num_points_straight: int = 50
) -> HelixTraceData:
    """Generates lists of X, Y, Z coordinates for track helices or straight lines."""
    helix_x, helix_y, helix_z = [], [], []

    for i in range(len(px_values)):
        if not all(np.isfinite(val) for val in [px_values[i], py_values[i], pz_values[i]]):
            continue
        
        particle_vec = vector.obj(px=px_values[i], py=py_values[i], pz=pz_values[i], mass=particle_mass)

        # Ensure beta is non-zero and finite for path length calculation
        beta = particle_vec.beta
        if not (np.isfinite(beta) and beta > 1e-9):
            # print(f"Warning: Non-physical beta ({beta}) for track {i}. Skipping helix.")
            continue
        
        path_len_m = 2.0 # Max path length to visualize

        if charge_values[i] == 0 or abs(b_field) < 1e-9 or particle_vec.pt < 1e-6: # Straight line
            time_param = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_straight)
            current_path = config.C_LIGHT * beta * time_param
            
            # Handle p=0 case for direction vector
            p_abs = particle_vec.p
            dir_x = particle_vec.px / p_abs if p_abs > 1e-9 else 0
            dir_y = particle_vec.py / p_abs if p_abs > 1e-9 else 0
            dir_z = particle_vec.pz / p_abs if p_abs > 1e-9 else 0

            path_x = scale_factor * dir_x * current_path
            path_y = scale_factor * dir_y * current_path
            path_z = scale_factor * dir_z * current_path
        else: # Helix
            # radius_m = particle_vec.pt / (abs(charge_values[i]) * 0.3 * abs(b_field)) # Simplified qvb = mv^2/r -> r = mv/qb = pT/ (q*0.3*B)
            # Factor of 0.3 is for B in Tesla, pT in GeV/c, q in units of e
             # Ensure b_field is not zero before division
            abs_b_field = abs(b_field)
            if abs_b_field < 1e-9: # Effectively zero B-field, should have been caught by straight line
                # This case is redundant if straight line logic is correct, but as a safeguard
                # print(f"Warning: Near-zero B-field ({b_field}) for charged particle track {i}. Treating as straight.")
                time_param = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_straight)
                current_path = config.C_LIGHT * beta * time_param
                p_abs = particle_vec.p
                dir_x = particle_vec.px / p_abs if p_abs > 1e-9 else 0
                dir_y = particle_vec.py / p_abs if p_abs > 1e-9 else 0
                dir_z = particle_vec.pz / p_abs if p_abs > 1e-9 else 0
                path_x = scale_factor * dir_x * current_path
                path_y = scale_factor * dir_y * current_path
                path_z = scale_factor * dir_z * current_path
            else:
                radius_m = particle_vec.pt / (abs(charge_values[i]) * 0.299792458 * abs_b_field) # q=1, c included in 0.299...

            time_vals = np.linspace(0, path_len_m / (config.C_LIGHT * beta), num_points_helix)
            
            # Angular frequency omega = q*B / (gamma*m) = q*B / E (if E is in units of energy, not GeV)
            # Or, omega related to cyclotron frequency: omega_c = qB/m.
            # The angle turned is phi = omega_c * t / gamma = (qB/m) * (L/beta*c) / gamma
            # Simpler: phi(t) = (q * B_field * c^2 / E) * t.
            # Or, using angular velocity in xy plane: omega_xy = v_t / r = (beta*c) / radius_m
            # Total angle turned along path s: angle = s / radius_m
            
            # Original calculation's angle argument seems like omega * t
            # omega_notebook_calc = (charge_values[i]) * 0.3 * b_field / particle_vec.E # E in GeV
            # This omega_notebook_calc has units of 1/GeV * T. To get angle (rad), needs time and physical constants.
            # Let's use geometry: d_phi = ds / R_xy. For path s_xy in xy plane.
            # s_xy = v_t * t = pt/p_tot * beta*c*t
            # path_phi = (v_t * time_vals) / radius_m
            
            # Using simpler geometric approach:
            # The path length in xy plane for a turn angle d_theta is R*d_theta.
            # The z displacement is v_z * t.
            # t = s_arc / v_t
            # phi_turn = (charge_values[i] * b_field / particle_vec.pt) * (config.C_LIGHT * beta * particle_vec.pt/particle_vec.p * time_vals) # This is getting complicated
            
            # Let's try to replicate the original structure if it was working:
            # arg_angle_helix = (omega_notebook_calc / particle_vec.beta if particle_vec.beta > 1e-9 else 0) * (config.C_LIGHT * time_vals)
            # The omega_notebook_calc was: (charge) * 0.3 * B / E_total_GeV
            # So arg_angle has units of (T / GeV) * m = T*m/GeV. This is not an angle.
            #
            # A common formula for helix:
            # x(t) = x_c + R * cos(omega*t + phi_0)
            # y(t) = y_c + R * sin(omega*t + phi_0)  (sign of R or omega depends on q*B)
            # z(t) = z_0 + v_z * t
            # where omega = qB/m_rel = qB*c^2/E. R = pT / (|q|B * const_for_units)
            # phi_0 related to initial direction.
            # Particle starts at (0,0,0). Initial direction is particle_vec.phi.
            # Center of helix xc, yc:  xc = -R * sin(phi_initial), yc = R * cos(phi_initial) (if qB > 0)
            # or xc = R * sin(phi_initial), yc = -R * cos(phi_initial) (if qB < 0)
            # Let q_eff = charge_values[i] * np.sign(b_field) -> use -q_eff for sign convention
            # For initial momentum (px,py) = (pt*cos(phi0), pt*sin(phi0))
            # And B in z-dir. Lorentz force F = q (v x B). Fy = q * vx * Bz, Fx = -q * vy * Bz
            # x(t) = (pt / (qB)) * (sin(omega*t + phi0) - sin(phi0))
            # y(t) = (pt / (qB)) * (-cos(omega*t + phi0) + cos(phi0))
            # z(t) = pz/p * (beta*c*t)
            # Here, pt/(qB) is the radius R. omega = qB/m_rel = qB*c^2/E
            
            omega_relativistic = (charge_values[i] * 0.299792458 * b_field * config.C_LIGHT) / particle_vec.energy # E in GeV, B in T
            # time_vals here represents actual time.
            
            # Original logic:
            # arg_angle_helix = (omega_notebook_calc / particle_vec.beta if particle_vec.beta > 1e-9 else 0) * (config.C_LIGHT * time_values_helix)
            # path_segments_x = scale_factor * radius_m * (np.cos(arg_angle_helix + particle_vec.phi - np.pi/2) - np.cos(particle_vec.phi - np.pi/2))
            # This seems like particle_vec.phi is initial direction, -pi/2 to rotate frame.
            # The angular speed seems to be (omega_notebook_calc / particle_vec.beta) * C_LIGHT
            # Let's try to use a more standard formulation or trust the original if it visually worked.
            # The original `arg_angle_helix` could be the angle turned in the xy-plane.
            
            # Re-evaluating original angular velocity:
            # omega_orig_angular_vel = (charge_values[i] * 0.3 * b_field / particle_vec.E) / particle_vec.beta * config.C_LIGHT
            # Units: (1/GeV * T) / (1) * m/s = T*m / (GeV*s). Still not rad/s.
            #
            # Let's use the geometric `angle = s_xy / R`.
            # s_xy is path length in xy projection. s_xy = (pt/p_total) * (beta * c * t)
            total_path_length_in_xy = (particle_vec.pt / particle_vec.p if particle_vec.p > 1e-9 else 0) * \
                                      (config.C_LIGHT * beta * time_vals)
            angle_turned = total_path_length_in_xy / radius_m if radius_m > 1e-9 else np.zeros_like(time_vals)

            # The center of the circle is at R from origin, perpendicular to pT vector
            # Initial phi is particle_vec.phi
            # Offset angle for center: particle_vec.phi - np.sign(charge_values[i]*b_field)*np.pi/2
            # Position relative to center: (R*cos(current_angle), R*sin(current_angle))
            # Need to be careful with signs and initial phase
            
            # Sticking close to original formulation:
            # The phi_offset (particle_vec.phi - np.pi/2) sets the initial phase for the cosine/sine.
            # The arg_angle_helix must represent the change in angle.
            # If arg_angle_helix = omega_true * t
            # omega_true = qB / (gamma*m) = qB*c^2 / E_GeV (with appropriate constants for units)
            # Let's assume the original arg_angle_helix calculation was aiming for this.
            # The formula for x_helix, y_helix describes a circle starting at origin with initial direction.
            # x(phi_turned) = R * (cos(phi_start + phi_turned) - cos(phi_start))
            # y(phi_turned) = R * (sin(phi_start + phi_turned) - sin(phi_start))
            # Here, phi_start relates to the initial direction in the coordinate system of the helix formula.
            # The original used phi_start = particle_vec.phi - np.pi/2.
            # And arg_angle_helix was the phi_turned.

            # Let's compute phi_turned more directly:  v_perp * t / R = (p_T / m_rel) * t / R
            # arc_length_param = config.C_LIGHT * beta * (particle_vec.pt / particle_vec.p) * time_vals # distance traveled in xy plane
            # angle_turned_geom = arc_length_param / radius_m if radius_m > 1e-9 else np.zeros_like(time_vals)

            # Using the omega that yields correct Larmor precession frequency:
            omega_larmor = (charge_values[i] * 0.299792458 * b_field * config.C_LIGHT) / particle_vec.energy # E in GeV
            arg_angle_dynamic = omega_larmor * time_vals # this is angle turned over time 't'

            phi0_helix_frame = particle_vec.phi - np.sign(charge_values[i] * b_field) * np.pi / 2.0
            # If qB > 0, particle turns left. If qB < 0, turns right.
            # x(t) = R * ( cos(phi0_helix_frame) - cos(phi0_helix_frame - np.sign(qB) * arg_angle_dynamic) )
            # y(t) = R * ( sin(phi0_helix_frame) - sin(phi0_helix_frame - np.sign(qB) * arg_angle_dynamic) )
            # This is a common form for trajectory starting at origin.
            
            # Let's try the original form but ensure arg_angle is indeed an angle.
            # The length of arc traveled in xy plane is s_xy = R * theta_turned.
            # s_xy(t) = (pt/p_total) * (speed_of_light * beta * t)
            # So, theta_turned(t) = s_xy(t) / R
            s_xy_t = (particle_vec.pt / particle_vec.p if particle_vec.p > 1e-9 else 0) * (config.C_LIGHT * beta * time_vals)
            arg_angle_helix = s_xy_t / radius_m if radius_m > 1e-9 else np.zeros_like(time_vals)
            
            # Sign of turning depends on charge * B_field direction (assuming B is along z)
            # If B is along +z, positive charge turns counter-clockwise (phi increases)
            # If B is along -z (like config.DEFAULT_B_FIELD_TESLA), positive charge turns clockwise (phi decreases)
            # The original code:
            # x_helix = scale_factor * radius_m * (np.cos(arg_angle_helix + particle_vec.phi - np.pi/2) - np.cos(particle_vec.phi - np.pi/2))
            # y_helix = scale_factor * radius_m * (np.sin(arg_angle_helix + particle_vec.phi - np.pi/2) - np.sin(particle_vec.phi - np.pi/2))
            # This implies an addition of angle. If charge*B is negative, arg_angle_helix should effectively be negative.
            # The way arg_angle_helix is defined (s_xy_t / radius_m) is always positive.
            # So we need to introduce sign of (charge * B_field)
            
            # Helix turning direction:
            # Positive charge, B > 0: turns CCW (angle increases)
            # Positive charge, B < 0: turns CW (angle decreases)
            # Negative charge, B > 0: turns CW
            # Negative charge, B < 0: turns CCW
            # So, effective angle change is proportional to sign(charge * B_field)
            
            signed_arg_angle_helix = np.sign(charge_values[i] * b_field) * arg_angle_helix

            initial_phase_angle = particle_vec.phi - np.pi/2.0 # As in original

            path_x = scale_factor * radius_m * (np.cos(initial_phase_angle + signed_arg_angle_helix) - np.cos(initial_phase_angle))
            path_y = scale_factor * radius_m * (np.sin(initial_phase_angle + signed_arg_angle_helix) - np.sin(initial_phase_angle))
            path_z = scale_factor * (particle_vec.pz / particle_vec.p if particle_vec.p > 1e-9 else 0) * \
                     (config.C_LIGHT * beta * time_vals)


        helix_x.extend(list(path_x))
        helix_y.extend(list(path_y))
        helix_z.extend(list(path_z))
        helix_x.append(None) # Break for Plotly
        helix_y.append(None)
        helix_z.append(None)
        
    return helix_x, helix_y, helix_z


def create_3d_event_display_figure(
    traces: List[go.Scatter3d],
    event_index: int, b_field: float, scale_factor: float, min_pt_cut: float,
    root_data_error: bool = False
) -> go.Figure:
    """Creates the 3D event display figure from a list of traces."""
    fig = go.Figure(data=traces)
    
    title = f"3D Event Display: Event {event_index} (B={b_field:.1f}T, Scale={scale_factor:.0f}, pT_min={min_pt_cut:.1f} GeV)"
    if root_data_error:
        title = f"Error loading 3D data for Event {event_index}. Display may be incomplete."

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-2500, 2500]),
            yaxis=dict(title='Y (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-2500, 2500]),
            zaxis=dict(title='Z (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white", range=[-4500, 4500]),
            camera=dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)),
            aspectmode='cube'
        ),
        legend=dict(x=0.75, y=0.95, font=dict(size=10)),
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision="persistent_view" # Persist camera view across updates
    )
    fig.update_traces(marker_line_width=0, selector=dict(type='scatter3d'))
    
    if not traces:
        fig.add_annotation(
            text="No 3D data to display for this event or parameters.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16
        )
    return fig


# --- Global Data (Loaded once) ---
# These are treated as effectively global after initial loading for the Dash app context
root_file_obj: Optional[uproot.ReadOnlyDirectory] = None
event_tree_global: Optional[uproot.TTree] = None
collectionIDs_global: Dict[str, int] = {}
collectionIDs_reverse_global: Dict[int, str] = {}
max_events_root_global: int = 0

parquet_data_global: Optional[AkArray] = None
max_events_pq_global: int = 0

mlpf_model_global: Optional[MLPF] = None
model_kwargs_global: Optional[Dict[str, Any]] = None

MAX_EVENTS_AVAILABLE: int = 0


def initialize_global_data() -> None:
    """Loads all necessary data into global variables."""
    global root_file_obj, event_tree_global, collectionIDs_global, collectionIDs_reverse_global, max_events_root_global
    global parquet_data_global, max_events_pq_global
    global mlpf_model_global, model_kwargs_global
    global MAX_EVENTS_AVAILABLE

    print(f"Using PyTorch device: {config.TORCH_DEVICE}")

    root_file_obj, event_tree_global, collectionIDs_global, collectionIDs_reverse_global, max_events_root_global = \
        load_root_data(config.ROOT_FILE_PATH)

    parquet_data_global, max_events_pq_global = load_parquet_data(config.PARQUET_FILE_PATH)

    mlpf_model_global, model_kwargs_global = load_mlpf_model(
        config.MODEL_KWARGS_PATH, config.MODEL_CHECKPOINT_PATH, config.TORCH_DEVICE
    )

    if max_events_root_global > 0 and max_events_pq_global > 0:
        MAX_EVENTS_AVAILABLE = min(max_events_root_global, max_events_pq_global)
        if max_events_root_global != max_events_pq_global:
            print(f"Warning: Mismatch in event counts. ROOT: {max_events_root_global}, Parquet: {max_events_pq_global}. Using minimum: {MAX_EVENTS_AVAILABLE}")
    elif max_events_root_global > 0:
        MAX_EVENTS_AVAILABLE = max_events_root_global
        print(f"Warning: Parquet data has no events or failed to load. Using ROOT event count: {MAX_EVENTS_AVAILABLE}")
    elif max_events_pq_global > 0:
        MAX_EVENTS_AVAILABLE = max_events_pq_global
        print(f"Warning: ROOT data has no events or failed to load. Using Parquet event count: {MAX_EVENTS_AVAILABLE}")
    else:
        MAX_EVENTS_AVAILABLE = 0
        print("Warning: No events found in either ROOT or Parquet file.")


# --- Dash Application ---
app = dash.Dash(__name__)
app.title = "CLIC Event Display & Feature Inspector (Refactored)"

def create_app_layout() -> html.Div:
    """Creates the Dash application layout."""
    max_event_val = max(0, MAX_EVENTS_AVAILABLE -1) if MAX_EVENTS_AVAILABLE > 0 else 0
    input_disabled = MAX_EVENTS_AVAILABLE == 0

    return html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
        html.H1("Interactive Particle Event Display & Feature Inspector", style={'textAlign': 'center', 'color': '#333'}),
        html.Div(style={'width': '95%', 'margin': 'auto', 'padding': '15px', 'backgroundColor': '#f0f0f0', 'borderRadius': '8px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '15px'}, children=[
                html.Div(style={'margin': '5px'}, children=[
                    html.Label("Event Index (iev):", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='iev-input', type='number', value=config.DEFAULT_EVENT_INDEX, min=0, max=max_event_val, step=1, disabled=input_disabled, style={'padding': '5px', 'width': '80px'})
                ]),
                html.Div(style={'margin': '5px'}, children=[
                    html.Label("B-Field (T):", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='b-field-input', type='number', value=config.DEFAULT_B_FIELD_TESLA, step=0.1, style={'padding': '5px', 'width': '80px'})
                ]),
                html.Div(style={'margin': '5px'}, children=[
                    html.Label("Scale Factor:", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='scale-input', type='number', value=config.DEFAULT_SCALE_FACTOR, step=100, style={'padding': '5px', 'width': '80px'})
                ]),
                html.Div(style={'margin': '5px'}, children=[
                    html.Label("Min Track pT (GeV):", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='pt-cut-input', type='number', value=config.DEFAULT_PT_CUT_GEV, min=0.0, step=0.1, style={'padding': '5px', 'width': '80px'})
                ]),
                html.Div(style={'margin': '10px'}, children=[
                    dcc.Checklist(
                        id='pandora-toggle',
                        options=[{'label': 'Show Pandora Clusters', 'value': 'show'}],
                        value=['show'] if config.DEFAULT_SHOW_PANDORA_CLUSTERS else [],
                        style={'fontWeight': 'bold'}
                    )
                ]),
            ]),
            html.Div(id='error-message-display', style={'marginTop': '10px', 'marginBottom': '10px', 'color': 'red', 'textAlign': 'center', 'minHeight': '20px', 'whiteSpace': 'pre-wrap'}),
            html.Div(id='model-inference-output-display', style={'marginTop': '10px', 'marginBottom': '10px', 'color': 'blue', 'textAlign': 'center', 'minHeight': '20px', 'whiteSpace': 'pre-wrap'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'gap': '15px'}, children=[
                html.Div(style={'flex': '2', 'minWidth': '60%'}, children=[
                    dcc.Loading(id="loading-graph", type="circle", children=[
                        dcc.Graph(id='particle-graph-display', style={'height': '75vh'})
                    ]),
                ]),
                html.Div(style={'flex': '1', 'minWidth': '35%', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}, children=[
                    html.H3("Track Features (X_track) - Normalized", style={'textAlign': 'center', 'marginBlock': '5px'}),
                    dcc.Loading(id="loading-xtrack-heatmap", type="circle", children=[
                        dcc.Graph(id='xtrack-heatmap-display', style={'height': '36vh'})
                    ]),
                    html.H3("Cluster Features (X_cluster) - Normalized", style={'textAlign': 'center', 'marginBlock': '5px'}),
                    dcc.Loading(id="loading-xcluster-heatmap", type="circle", children=[
                        dcc.Graph(id='xcluster-heatmap-display', style={'height': '36vh'})
                    ]),
                ])
            ]),
        ])
    ])

# Assign layout after MAX_EVENTS_AVAILABLE is known
# app.layout = create_app_layout() # This will be set in __main__


@app.callback(
    [Output('particle-graph-display', 'figure'),
     Output('xtrack-heatmap-display', 'figure'),
     Output('xcluster-heatmap-display', 'figure'),
     Output('error-message-display', 'children'),
     Output('model-inference-output-display', 'children')],
    [Input('iev-input', 'value'),
     Input('b-field-input', 'value'),
     Input('scale-input', 'value'),
     Input('pt-cut-input', 'value'),
     Input('pandora-toggle', 'value')]
)
def update_graph_and_heatmaps(
    current_event_idx: Optional[int],
    b_field_val: Optional[float],
    scale_factor_val: Optional[float],
    min_pt_cut_val: Optional[float],
    pandora_toggle_val: List[str]
) -> Tuple[go.Figure, go.Figure, go.Figure, str, str]:
    
    error_messages: List[str] = []
    model_output_msg: str = "MLPF Model not run or no data."

    # --- Input Validation and Defaults ---
    if current_event_idx is None: current_event_idx = config.DEFAULT_EVENT_INDEX
    if b_field_val is None: b_field_val = config.DEFAULT_B_FIELD_TESLA
    if scale_factor_val is None: scale_factor_val = config.DEFAULT_SCALE_FACTOR
    if min_pt_cut_val is None: min_pt_cut_val = config.DEFAULT_PT_CUT_GEV
    
    show_pandora_clusters = 'show' in pandora_toggle_val

    # Initialize empty figures
    empty_heatmap = create_empty_heatmap_figure()
    xtrack_fig = empty_heatmap
    xcluster_fig = empty_heatmap
    
    scene_dict_error = dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)")
    fig_3d = go.Figure().update_layout(title="Error: Event index invalid or data unavailable.", scene=scene_dict_error)


    if not (0 <= current_event_idx < MAX_EVENTS_AVAILABLE):
        err = f"Event index {current_event_idx} out of range (0-{max(0,MAX_EVENTS_AVAILABLE-1)})."
        error_messages.append(err)
        fig_3d.update_layout(title=f"Error: {err}")
        return fig_3d, xtrack_fig, xcluster_fig, " ".join(error_messages), model_output_msg

    # --- MLPF Model Inference ---
    if not (mlpf_model_global is None) and not (model_kwargs_global is None) and not (parquet_data_global is None) and \
       'X_track' in parquet_data_global.fields and 'X_cluster' in parquet_data_global.fields and \
       0 <= current_event_idx < max_events_pq_global:
        try:
            # Awkward to NumPy for model input
            x_track_event_ak = parquet_data_global['X_track'][current_event_idx]
            x_cluster_event_ak = parquet_data_global['X_cluster'][current_event_idx]
            
            x_track_event_np = awkward.to_numpy(x_track_event_ak) if len(x_track_event_ak) > 0 else np.empty((0,0)) # ensure 2D, even if (0,N) or (0,0)
            x_cluster_event_np = awkward.to_numpy(x_cluster_event_ak) if len(x_cluster_event_ak) > 0 else np.empty((0,0))

            # Handle cases where to_numpy might return 0-dim or 1-dim for empty/single item lists
            if x_track_event_np.ndim < 2 and x_track_event_np.shape != (0,0) : x_track_event_np = np.atleast_2d(x_track_event_np)
            if x_cluster_event_np.ndim < 2 and x_cluster_event_np.shape != (0,0): x_cluster_event_np = np.atleast_2d(x_cluster_event_np)


            model_preds, inference_msg = run_mlpf_inference(
                mlpf_model_global, model_kwargs_global,
                x_track_event_np, x_cluster_event_np,
                config.TORCH_DEVICE
            )
            model_output_msg = f"(Event {current_event_idx}): {inference_msg}"
            if model_preds is None:
                error_messages.append(f"MLPF Inference: {inference_msg}")
        except Exception as e_infer:
            err_msg = f"Error during MLPF model inference: {str(e_infer)}"
            error_messages.append(err_msg)
            model_output_msg = f"MLPF Model Inference Failed (Event {current_event_idx}): {str(e_infer)}"
            # print(f"Model inference error for event {current_event_idx}: {e_infer}")
    elif mlpf_model_global is None:
        model_output_msg = "MLPF Model not loaded. Cannot run inference."
    elif parquet_data_global is None or not ('X_track' in parquet_data_global.fields and 'X_cluster' in parquet_data_global.fields):
        model_output_msg = "Parquet data (X_track/X_cluster) not available for MLPF model."
    elif not (0 <= current_event_idx < max_events_pq_global) :
        model_output_msg = f"Event {current_event_idx} out of range for Parquet data. MLPF model not run."


    # --- Process Parquet Data for Heatmaps ---
    if parquet_data_global is not None and 0 <= current_event_idx < max_events_pq_global:
        try:
            if 'X_track' in parquet_data_global.fields and len(parquet_data_global['X_track']) > current_event_idx:
                x_track_data_ak = parquet_data_global['X_track'][current_event_idx]
                if len(x_track_data_ak) > 0 :
                    x_track_np_hm = awkward.to_numpy(x_track_data_ak)
                    # Normalize (example: z-score per feature globally, then select event)
                    # For simplicity, let's assume normalization happens here or features are already somewhat scaled.
                    # A proper normalization would compute mean/std over the whole dataset (or a batch)
                    # This is a quick per-event normalization for display, might not be ideal.
                    if x_track_np_hm.ndim > 0 and x_track_np_hm.shape[0] > 0: # only if there's data
                        means_trk = np.mean(x_track_np_hm, axis=0, keepdims=True)
                        stds_trk = np.std(x_track_np_hm, axis=0, keepdims=True)
                        stds_trk[stds_trk == 0] = 1 # Avoid division by zero
                        x_track_np_norm = (x_track_np_hm - means_trk) / stds_trk
                        x_track_np_norm[np.isnan(x_track_np_norm)] = 0
                        xtrack_fig = create_feature_heatmap(x_track_np_norm, config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx})", "Trk")
                    else: # No tracks for this event in parquet
                         xtrack_fig = create_feature_heatmap(np.array([]), config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx}) - No Tracks", "Trk")
                else: # No tracks for this event in parquet
                    xtrack_fig = create_feature_heatmap(np.array([]), config.TRACK_FEATURE_ORDER, f"X_track (Event {current_event_idx}) - No Tracks", "Trk")

            else: error_messages.append(f"X_track data not found for event {current_event_idx} in Parquet.")

            if 'X_cluster' in parquet_data_global.fields and len(parquet_data_global['X_cluster']) > current_event_idx:
                x_cluster_data_ak = parquet_data_global['X_cluster'][current_event_idx]
                if len(x_cluster_data_ak) > 0:
                    x_cluster_np_hm = awkward.to_numpy(x_cluster_data_ak)
                    if x_cluster_np_hm.ndim > 0 and x_cluster_np_hm.shape[0] > 0:
                        means_cls = np.mean(x_cluster_np_hm, axis=0, keepdims=True)
                        stds_cls = np.std(x_cluster_np_hm, axis=0, keepdims=True)
                        stds_cls[stds_cls == 0] = 1
                        x_cluster_np_norm = (x_cluster_np_hm - means_cls) / stds_cls
                        x_cluster_np_norm[np.isnan(x_cluster_np_norm)] = 0
                        xcluster_fig = create_feature_heatmap(x_cluster_np_norm, config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx})", "Cls", colorscale='Plasma')
                    else: # No clusters
                        xcluster_fig = create_feature_heatmap(np.array([]), config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx}) - No Clusters", "Cls", colorscale='Plasma')
                else: # No clusters
                     xcluster_fig = create_feature_heatmap(np.array([]), config.CLUSTER_FEATURE_ORDER, f"X_cluster (Event {current_event_idx}) - No Clusters", "Cls", colorscale='Plasma')
            else: error_messages.append(f"X_cluster data not found for event {current_event_idx} in Parquet.")

        except Exception as e_pq_vis:
            error_messages.append(f"Error processing Parquet data for heatmaps: {str(e_pq_vis)}")
            # print(f"Parquet heatmap error for event {current_event_idx}: {e_pq_vis}")
    elif parquet_data_global is None:
        error_messages.append("Parquet data not loaded. Heatmaps unavailable.")
    elif not (0 <= current_event_idx < max_events_pq_global):
        error_messages.append(f"Event index {current_event_idx} out of range for Parquet data. Heatmaps unavailable.")

    # --- Process ROOT Data for 3D Event Display ---
    fig_traces_3d: List[go.Scatter3d] = []
    root_data_issue_for_3d = False

    if event_tree_global is None or max_events_root_global == 0:
        error_messages.append("ROOT data not loaded or no events. 3D display unavailable.")
        root_data_issue_for_3d = True
    elif not (0 <= current_event_idx < max_events_root_global):
        error_messages.append(f"Event index {current_event_idx} out of range for ROOT data. 3D display unavailable.")
        root_data_issue_for_3d = True
    else:
        # 1. Hits from ROOT
        all_hits_features_list: List[AkArray] = []
        if collectionIDs_reverse_global:
            for coll_name_from_meta in collectionIDs_reverse_global.values():
                if coll_name_from_meta in config.HIT_COLLECTIONS_TO_PLOT and coll_name_from_meta in event_tree_global:
                    hit_features = extract_hit_features(event_tree_global, current_event_idx, coll_name_from_meta, config.HIT_FEATURES_STD)
                    if len(hit_features.fields) > 0 and 'position.x' in hit_features.fields and len(hit_features['position.x']) > 0:
                        all_hits_features_list.append(hit_features)
        
        if all_hits_features_list:
            try:
                # Ensure common fields for concatenation, if necessary (extract_hit_features should ensure consistent fields)
                hit_feature_matrix = awkward.concatenate(all_hits_features_list, axis=0)
                if len(hit_feature_matrix) > 0 and all(f in hit_feature_matrix.fields for f in ['position.x', 'position.y', 'position.z', 'energy', 'subdetector']):
                    df_hits = awkward.to_dataframe(hit_feature_matrix[['position.x', 'position.y', 'position.z', 'energy', 'subdetector']])
                    df_hits.rename(columns={'position.x': 'px', 'position.y': 'py', 'position.z': 'pz'}, inplace=True)
                    
                    df_hits["energy"] = 1000 * df_hits["energy"] # MeV
                    df_hits["plotsize"] = 2.0
                    df_hits.loc[df_hits["subdetector"]==0, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==0, "energy"]/5), 1, 10)
                    df_hits.loc[df_hits["subdetector"]==1, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==1, "energy"]/10), 1, 10)
                    df_hits.loc[df_hits["subdetector"]==2, "plotsize"] = np.clip(2 + 2 * np.log1p(df_hits.loc[df_hits["subdetector"]==2, "energy"]*100), 1, 10)
                    df_hits.loc[df_hits["subdetector"]==3, "plotsize"] = 3.0

                    for subdet_idx in df_hits["subdetector"].unique():
                        sub_df = df_hits[df_hits["subdetector"] == subdet_idx]
                        fig_traces_3d.append(go.Scatter3d(
                            x=np.clip(sub_df["px"], -4500, 4500), y=np.clip(sub_df["py"], -4500, 4500), z=np.clip(sub_df["pz"], -4500, 4500),
                            mode='markers',
                            marker=dict(size=sub_df["plotsize"], color=config.HIT_SUBDETECTOR_COLOR.get(subdet_idx, "grey"), opacity=0.7),
                            name=config.HIT_LABELS.get(subdet_idx, "Unknown Hit")
                        ))
                else: error_messages.append("Concatenated hit data is empty or missing essential fields.")
            except Exception as e_concat_hits:
                error_messages.append(f"Error processing/plotting ROOT hits: {e_concat_hits}")
                # print(f"Hit processing error: {e_concat_hits}")

        # 2. Tracks from ROOT
        track_px_all, track_py_all, track_pz_all, track_charge_all, track_pt_all = extract_track_features(
            event_tree_global, current_event_idx,
            config.DEFAULT_TRACK_COLLECTION_NAME, config.DEFAULT_TRACK_STATE_BRANCH_NAME,
            b_field_val
        )
        if track_pt_all.size > 0:
            pt_mask = track_pt_all >= min_pt_cut_val
            track_px, track_py, track_pz = track_px_all[pt_mask], track_py_all[pt_mask], track_pz_all[pt_mask]
            track_charge = track_charge_all[pt_mask]

            if track_px.size > 0:
                try:
                    hx, hy, hz = generate_helix_trace_data(
                        track_px, track_py, track_pz, track_charge,
                        config.PION_MASS_GEV, b_field_val, scale_factor_val
                    )
                    if hx: # If any coords were generated
                        fig_traces_3d.append(go.Scatter3d(
                            x=hx, y=hy, z=hz, mode='lines',
                            line=dict(color="purple", width=3),
                            name=f"SiTrack (pT >= {min_pt_cut_val:.1f} GeV)"
                        ))
                except Exception as e_helix:
                    error_messages.append(f"Error generating track helices: {e_helix}")
                    # print(f"Helix gen error: {e_helix}")
            elif track_px_all.size > 0 : error_messages.append(f"No tracks with pT >= {min_pt_cut_val:.1f} GeV.")
        elif config.DEFAULT_TRACK_COLLECTION_NAME in event_tree_global:
            error_messages.append("No tracks processed or found for this event from ROOT.")


        # 3. Pandora Clusters from ROOT
        if show_pandora_clusters:
            df_pandora = extract_pandora_cluster_features(
                event_tree_global, current_event_idx,
                config.PANDORA_CLUSTER_COLLECTION_NAME, config.PANDORA_CLUSTER_FEATURES
            )
            if df_pandora is not None and not df_pandora.empty:
                if all(col in df_pandora for col in ["x", "y", "z", "energy"]):
                    df_pandora["plotsize"] = np.clip(2 + 2 * np.log1p(df_pandora["energy"]), 1, 12) # Assuming energy is in GeV
                    hover_texts_pandora = [
                        f"E: {df_pandora['energy'][i]:.2f} GeV" +
                        (f"<br>Type: {int(df_pandora['type'][i])}" if 'type' in df_pandora and pd.notna(df_pandora['type'][i]) else "") +
                        (f"<br>E_err: {df_pandora['energyError'][i]:.2f} GeV" if 'energyError' in df_pandora and pd.notna(df_pandora['energyError'][i]) else "")
                        for i in df_pandora.index
                    ]
                    fig_traces_3d.append(go.Scatter3d(
                        x=df_pandora["x"], y=df_pandora["y"], z=df_pandora["z"],
                        mode='markers',
                        marker=dict(size=df_pandora["plotsize"], color=config.PANDORA_CLUSTER_COLOR, opacity=0.6),
                        name="Pandora Clusters", text=hover_texts_pandora, hoverinfo="text+name"
                    ))
                else: error_messages.append("Essential Pandora Cluster data (pos/energy) missing.")
            elif df_pandora is None:
                 error_messages.append(f"'{config.PANDORA_CLUSTER_COLLECTION_NAME}' not found or error during extraction.")


    fig_3d = create_3d_event_display_figure(
        fig_traces_3d, current_event_idx, b_field_val,
        scale_factor_val, min_pt_cut_val, root_data_issue_for_3d
    )
    
    final_error_message = " | ".join(msg for msg in error_messages if msg)
    return fig_3d, xtrack_fig, xcluster_fig, final_error_message, model_output_msg


if __name__ == '__main__':
    initialize_global_data()
    app.layout = create_app_layout() # Create layout after MAX_EVENTS_AVAILABLE is known

    if MAX_EVENTS_AVAILABLE == 0:
        print("--------------------------------------------------------------------------")
        print("Failed to load data or no events found in either ROOT or Parquet file.")
        print(f"Please check ROOT_FILE_PATH: '{config.ROOT_FILE_PATH}'")
        print(f"and PARQUET_FILE_PATH: '{config.PARQUET_FILE_PATH}'.")
        print("The Dash app will start but may not be functional.")
        print("--------------------------------------------------------------------------")
    else:
        print(f"Successfully loaded data. Max events available for display: {MAX_EVENTS_AVAILABLE}")
        if max_events_root_global > 0 and collectionIDs_global:
            print(f"ROOT Collections found: {list(collectionIDs_global.keys())}")
        elif max_events_root_global > 0:
            print("ROOT Metadata for CollectionIDs not found or empty.")
        
        if parquet_data_global is not None:
            print(f"Parquet data fields: {parquet_data_global.fields}")

    app.run(debug=True)