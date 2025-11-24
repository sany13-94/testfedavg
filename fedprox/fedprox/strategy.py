from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import os
from pathlib import Path
import base64
from collections import defaultdict

import pickle
import pandas as pd
import numpy as np
from flwr.server.client_manager import ClientManager
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Optional, Callable
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
     GetPropertiesIns, GetPropertiesRes
)
GLOBAL_FEDAVG_STRATEGY_INSTANCE = None
class FedAVGWithEval(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 3,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
          ground_truth_stragglers=None,
         total_rounds: int = 15,
         on_fit_config_fn: Optional[Callable[[int], dict]] = None,
        **kwargs,
    ) -> None:
     super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,   # ✅ make sure to include this

            **kwargs,
        )

     self.uuid_to_cid = {}     # {"8325...": "client_0"}
     self.cid_to_uuid = {}     # {"client_0": "8325..."}
     self.ground_truth_cids = set(ground_truth_stragglers)  # {"client_0","client_1",...}
     self.ground_truth_flower_ids = set()  # will be filled as clients appear
     self.total_rounds=total_rounds
     self.client_participation_count = {}  # client_id -> number of times selected
     self.min_fit_clients = min_fit_clients
     # mappings
     self.training_times = {}
     self.selection_counts = {}
     self.participated_clients = set()
     self.client_assignments = {}
     self.participated_clients = set()
     self.client_assignments = {}
     self.cluster_prototypes = {}
     self.last_round_participants = set()
     self.min_evaluate_clients=min_evaluate_clients
     self.min_available_clients=min_available_clients
     self.best_avg_accuracy=0.0
     map_path="client_id_mapping1.csv"
     # Track selection history and scores
     self.selection_history = defaultdict(list)  # round -> [client_ids]
     self.prototype_scores = defaultdict(dict)   # round -> {client_id: distance_score}        
     self.num_clients=20
     self.map_path = Path(map_path)
     expected_unique=self.min_fit_clients
     self.expected_unique = expected_unique
     # Track what we've already recorded: (client_cid, flower_node_id)
     self._seen= set()
     # expose this instance globally so main() can access it later
     global GLOBAL_FEDAVG_STRATEGY_INSTANCE
     GLOBAL_FEDAVG_STRATEGY_INSTANCE = self
     save_dir = "/kaggle/working/cluster-CDCSF/fedprox/checkpoints"
     self.save_dir = save_dir
     os.makedirs(self.save_dir, exist_ok=True)

     # If the CSV already exists, preload seen pairs (so we don't duplicate)
     if self.map_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.map_path, dtype=str)
                for _, r in df.iterrows():
                    self._seen.add((str(r["client_cid"]), str(r["flower_node_id"])))
            except Exception:
                pass  # if reading fails, start fresh in memory
         
     self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=3,  # total number of clients
        num_classes=2,           # number of classes in your dataset

save_dir="feature_visualizations"
          )
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation======= sanaaa')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

    def configure_fit(self, server_round, parameters, client_manager):
      
      sample_size, min_num_clients = self.num_fit_clients(
        client_manager.num_available()
    )
      clients = client_manager.sample(
        num_clients=min_num_clients,
        min_num_clients=min_num_clients,
    )
      print(f"[Server] Round {server_round} - num clients selected: {min_num_clients}")

    
      all_prototypes_list = []
      all_client_ids = []
      selected_clients_logical_ids = []

      for client in clients:
        try:
            # Request prototypes from client
            ins = GetPropertiesIns(config={"request": "prototypes"})
            props_res = client.get_properties(
                ins=ins,
                timeout=15.0,
                group_id=None,
            )

            # CRITICAL FIX: Convert ConfigsRecord to regular dict
            # In Flower, props_res.properties is a ConfigsRecord, not a dict
            props = {}
            try:
                # Method 1: Iterate over keys (most compatible)
                for key in props_res.properties.keys():
                    props[key] = props_res.properties[key]
            except Exception as conv_error:
                print(
                    f"[Server] Round {server_round} - "
                    f"client {client.cid} properties conversion failed: {conv_error}"
                )
                continue

            # Check if prototypes are available
            if "prototypes" not in props or "class_counts" not in props:
                print(
                    f"[Server] Round {server_round} - "
                    f"client {client.cid} has no prototypes yet, skipping"
                )
                continue

            # Decode prototypes
            try:
                prototypes_encoded = props["prototypes"]
                class_counts_encoded = props["class_counts"]
                
                proto_bytes = base64.b64decode(prototypes_encoded)
                prototypes = pickle.loads(proto_bytes)
                
                counts_bytes = base64.b64decode(class_counts_encoded)
                class_counts = pickle.loads(counts_bytes)
                
            except Exception as decode_error:
                print(
                    f"[Server] Round {server_round} - "
                    f"client {client.cid} decode error: {decode_error}"
                )
                continue

            # Validate prototypes
            if not isinstance(prototypes, dict):
                print(
                    f"[Server] Round {server_round} - "
                    f"client {client.cid} prototypes not a dict, skipping"
                )
                continue

            # Successfully collected prototypes
            all_prototypes_list.append(prototypes)

            # Get logical client ID
            client_cid = props.get("client_cid", client.cid)
            try:
                client_cid_int = int(client_cid)
            except Exception:
                # If conversion fails, use the original cid
                client_cid_int = client.cid

            all_client_ids.append(client_cid_int)
            selected_clients_logical_ids.append(client_cid_int)
            
            print(f"[Server] Round {server_round} - ✓ Client {client_cid_int}: Prototypes collected")

        except Exception as e:
            print(
                f"[Server] Round {server_round} - "
                f"error getting prototypes from client {client.cid}: {e}"
            )
            import traceback
            traceback.print_exc()
            continue

    
      if all_prototypes_list and all_client_ids:
        print(f"[Server] Round {server_round} - Computing prototype scores for {len(all_client_ids)} clients")
        self.compute_and_log_scores(
            round_num=server_round,
            selected_clients=selected_clients_logical_ids,
            all_prototypes_list=all_prototypes_list,
            all_client_ids=all_client_ids,
        )
      else:
        print(
            f"[Server] Round {server_round} - "
            "no prototypes collected, skipping score computation"
        )

    
      fit_config = {}
      if self.on_fit_config_fn is not None:
        fit_config = self.on_fit_config_fn(server_round)

    
      fit_instructions = []
      for client in clients:
        client_fit_config = dict(fit_config)
        client_fit_config["server_round"] = server_round
        client_fit_config["extract_prototypes"] = True

        fit_instructions.append((client, FitIns(parameters, client_fit_config)))

      return fit_instructions



    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
   
      #sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=10, min_num_clients=10
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
      # Create EvaluateIns for each client
   
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    
    def compute_and_log_scores(self, round_num, selected_clients, all_prototypes_list, 
                               all_client_ids):
       
        self.selection_history[round_num] = selected_clients
        
        # Compute global reference prototypes (average across all clients)
        reference_prototypes = self._compute_global_reference(all_prototypes_list)
        
        # Calculate distance scores for each client
        for client_id, prototypes in zip(all_client_ids, all_prototypes_list):
            score = self._calculate_prototype_distance(prototypes, reference_prototypes)
            self.prototype_scores[round_num][client_id] = score
        
        print(f"Round {round_num}: Computed scores for {len(selected_clients)} clients")
        print(f"  Domain diversity (avg distance): {np.mean(list(self.prototype_scores[round_num].values())):.4f}")
    
    def _compute_global_reference(self, all_prototypes_list):
        """
        Compute global reference prototypes by averaging across all clients.
        
        Args:
            all_prototypes_list: List of prototype dictionaries from all clients
            
        Returns:
            dict: Global reference prototypes
        """
        if not all_prototypes_list:
            return {}
        
        # Aggregate prototypes by class
        class_proto_aggregates = defaultdict(list)
        
        for client_prototypes in all_prototypes_list:
            for class_id, proto in client_prototypes.items():
                if proto is not None:
                    class_proto_aggregates[class_id].append(proto)
        
        # Average prototypes for each class
        global_prototypes = {}
        for class_id, proto_list in class_proto_aggregates.items():
            if proto_list:
                global_prototypes[class_id] = np.mean(np.stack(proto_list), axis=0)
        
        return global_prototypes
    
    def _calculate_prototype_distance(self, prototypes, reference_prototypes):
        """
        Calculate average Euclidean distance from client prototypes to reference prototypes.
        Higher distance indicates more distinct domain characteristics.
        
        Args:
            prototypes: Client's prototypes {class_id: prototype_vector}
            reference_prototypes: Reference prototypes (global average)
            
        Returns:
            float: Average distance to reference (domain distinctiveness score)
        """
        if not prototypes or not reference_prototypes:
            return 0.0
        
        distances = []
        
        for class_id, proto in prototypes.items():
            if proto is not None and class_id in reference_prototypes:
                ref_proto = reference_prototypes[class_id]
                if ref_proto is not None:
                    # Euclidean distance
                    dist = np.linalg.norm(proto - ref_proto)
                    distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def create_selection_matrix(self):
        """
        Create a matrix for heatmap visualization.
        Rows: Clients, Columns: Rounds
        Values: Prototype distance scores (0 if not selected)
        
        Returns:
            numpy.ndarray: Matrix of shape (num_clients, num_rounds)
        """
        if not self.selection_history:
            print("Warning: No selection history available")
            return np.zeros((self.num_clients, 1))
        
        max_round = max(self.selection_history.keys())
        matrix = np.zeros((self.num_clients, max_round))
        
        for round_num in range(1, max_round + 1):
            if round_num in self.prototype_scores:
                for client_id, score in self.prototype_scores[round_num].items():
                    # Convert client_id to integer index
                    client_idx = int(client_id) if isinstance(client_id, str) else client_id
                    if 0 <= client_idx < self.num_clients:
                        matrix[client_idx, round_num - 1] = score
        
        return matrix
    
    def plot_heatmap(self, figsize=(20, 10), cmap='viridis', save_name='fedavg_selection_heatmap.png'):
   
      print(f'==== Visualization Heatmap ====')
    
      matrix = self.create_selection_matrix()
    
      if matrix.shape[1] == 0:
        print("No data to plot")
        return
    
      num_rounds = matrix.shape[1]
      num_clients = matrix.shape[0]
    
      # Adjust figure size based on data dimensions
      figsize = (max(16, min(24, num_rounds * 0.08)), 
               max(8, min(14, num_clients * 0.4)))
    
      # Create figure with better DPI
      fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
      # Enhanced heatmap with better visual settings
      im = sns.heatmap(
        matrix, 
        cmap=cmap,
        cbar_kws={
            'label': 'Prototype score',
            'pad': 0.02,
            'aspect': 30,
            'shrink': 0.8
        },
        ax=ax,
        linewidths=0,
        rasterized=True,
        square=False,
        xticklabels=False,  # We'll set custom labels
        yticklabels=False   # We'll set custom labels
    )
    
      # Enhance colorbar
      cbar = im.collections[0].colorbar
      cbar.ax.tick_params(labelsize=11)
      cbar.set_label('Prototype score', fontsize=12, weight='bold')
    
      # Set title with better styling
      ax.set_title(
        'Prototype-based selection pattern',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
      # Y-AXIS (Clients) - Enhanced formatting
      ax.set_ylabel('Client ID (logical)', fontsize=14, fontweight='bold', labelpad=10)
    
      # Set y-ticks at center of each cell
      y_positions = np.arange(num_clients) + 0.5
      ax.set_yticks(y_positions)
    
      # Create clean y-axis labels
      y_labels = [f'Client {i}' for i in range(num_clients)]
      ax.set_yticklabels(
        y_labels,
        fontsize=10,
        rotation=0,
        va='center'
    )
    
      # X-AXIS (Rounds) - Enhanced formatting with intelligent tick placement
      ax.set_xlabel('Round', fontsize=14, fontweight='bold', labelpad=10)
    
      # Intelligent x-tick placement based on number of rounds
      if num_rounds <= 20:
        tick_interval = 1
        rotation = 45
      elif num_rounds <= 50:
        tick_interval = 2
        rotation = 45
      elif num_rounds <= 100:
        tick_interval = 5
        rotation = 60
      elif num_rounds <= 200:
        tick_interval = 10
        rotation = 70
      else:
        tick_interval = max(10, num_rounds // 30)
        rotation = 80
    
      # Generate tick positions
      x_tick_positions = np.arange(0, num_rounds, tick_interval)
    
      # Set x-ticks at center of cells
      ax.set_xticks(x_tick_positions + 0.5)
    
      # Create round labels (1-indexed for users)
      x_labels = [str(pos + 1) for pos in x_tick_positions]
      ax.set_xticklabels(
        x_labels,
        fontsize=9,
        rotation=rotation,
        ha='right',
        rotation_mode='anchor'
    )
    
      # Add subtle grid for better readability (optional)
      # ax.grid(False)  # Remove seaborn default grid if any
    
      # Improve layout
      plt.tight_layout()
    
      # Add subtle border around the heatmap
      for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')
    
      # Save figure with high quality
      save_path = self.save_dir / save_name
      plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
      print(f"Heatmap saved to: {save_path}")
    
      # Print statistics
      non_zero_values = matrix[matrix > 0]
      if len(non_zero_values) > 0:
        print(f"\nHeatmap Statistics:")
        print(f"  Total selections: {len(non_zero_values)}")
        print(f"  Score range: [{non_zero_values.min():.4f}, {non_zero_values.max():.4f}]")
        print(f"  Mean score: {non_zero_values.mean():.4f}")
        print(f"  Std deviation: {non_zero_values.std():.4f}")
    
      plt.close(fig)



    def _append_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        header = ["server_cid", "client_cid", "flower_node_id"]
        write_header = not self.map_path.exists()
        with self.map_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerows(rows)
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
      """
      Aggregate model updates and updat
      """
      if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")
        
      if not results:
            print(f"[Round {server_round}] No clients returned results. Skipping aggregation.")
            return None, {}
      try:  
        clients_params_list = []
        num_samples_list = []
        current_round_durations = []
        current_participants = set()
        new_rows: List[dict] = []

        # Process results and update tracking
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            uuid = client_proxy.cid  # Flower internal UUID            
            cid = metrics.get("client_cid")
            node = metrics.get("flower_node_id")
            self.uuid_to_cid[uuid] = cid
            self.cid_to_uuid[cid] = uuid

            print(f'===client id: {cid} and flower id {uuid} and node :{node} ===')

           
            if client_id not in self.client_participation_count:
              self.client_participation_count[client_id] = 0
            self.client_participation_count[client_id] += 1
            
            self.participated_clients.add(client_id)
            current_participants.add(client_id)
            
            # Update EMA training time - Equation (4)
          
            metrics = fit_res.metrics or {}
            if "duration" not in metrics:
                  continue
            dur = float(metrics["duration"])
            prev = self.training_times.get(uuid)
           
           
            current_round_durations.append(dur)
            if cid is None or node is None:
                continue
            key = (str(int(cid)), str(node))
            if key not in self._seen:
                self._seen.add(key)
                new_rows.append({
                    "server_cid": client_proxy.cid,        # connection id for reference
                    "client_cid": key[0],
                    "flower_node_id": key[1],
                })
            self._append_rows(new_rows)
            if new_rows:
              print(f"[Server] Recorded {len(new_rows)} new client(s). Total unique: {len(self._seen)}")
            
            
            # Collect parameters for aggregation
            clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
            num_samples_list.append(fit_res.num_examples)
           
            # The client should report its logical id once in fit metrics
            logical = fit_res.metrics.get("logical_id") if fit_res.metrics else None
            print(f"[Mapping]rtertr ====logical_id={logical} and {self.uuid_to_cid}")
            print(f"[Mapping]ddd ====: {self.cid_to_uuid}")

        
           
        self.last_round_participants = current_participants
        self.total_rounds_completed = server_round

        
        print(f"\n[Round {server_round}] Participants: {list(current_participants)}")
        print(f"[Round {server_round}] Average raw training time: {np.mean(current_round_durations):.2f}s")
        
        # Perform FedAvg aggregation
        aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
        
        if server_round == self.total_rounds :
            self.save_client_mapping()
            print("\n" + "="*80)
            print(f"[Round {server_round}] TRAINING COMPLETED - Auto-saving results...")
            print("="*80)
            self._save_all_results()

            matrix = self.create_selection_matrix()

            # Optionally inspect matrix shape
            print(f"Selection matrix shape: {matrix.shape}")

            # Save heatmap into the Hydra run directory
            
            self.plot_heatmap(
            save_name="fedavg_selection_heatmap.png",
            cmap="viridis",
        )
        return ndarrays_to_parameters(aggregated_params), {}

      except Exception as e:
        print(f"[aggregate_fit] Error processing client {getattr(client_proxy,'cid','?')}: {e}")
        # continue to next client so we still reach the mapping update

    def visualize_client_participation(self, participation_dict, save_path="participation_chart.png", 
                                   method_name="FedProto-Fair"):

      # ✅ Load UUID → cid mapping
      mapping_df = pd.read_csv("client_id_mapping1.csv")
      uuid_to_cid = dict(zip(mapping_df["flower_node_id"].astype(str),
                           mapping_df["client_cid"].astype(str)))

      # ✅ Convert participation_dict keys using mapping
      mapped_dict = {}
      for uuid, count in participation_dict.items():
        uuid_str = str(uuid)
        cid = uuid_to_cid.get(uuid_str, f"UNK-{uuid}")  # fallback: unknown
        print('======{cid}====')
        mapped_dict[cid] = count

      # ✅ Sort clients by numeric cid
      sorted_items = sorted(mapped_dict.items(), key=lambda x: int(x[0]))
      client_ids = [f"Client {item[0]}" for item in sorted_items]
      counts = [item[1] for item in sorted_items]

      # ✅ Plot exactly same as before (using client_ids now)
      fig, ax = plt.subplots(figsize=(14, 6))
      bars = ax.bar(range(len(client_ids)), counts)

      for i, count in enumerate(counts):
        if count == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.5)

      ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
      ax.set_ylabel('Number of Participations', fontsize=12, fontweight='bold')
      ax.set_title(f'Client Participation Distribution - {method_name}', fontsize=14, fontweight='bold')
      ax.set_xticks(range(len(client_ids)))
      ax.set_xticklabels(client_ids, rotation=45, ha='right')
      ax.grid(axis='y', alpha=0.3, linestyle='--')

      total_clients = len(client_ids)
      participated = sum(1 for c in counts if c > 0)
      avg_participation = np.mean(counts)
      std_participation = np.std(counts)

      stats_text = f"Total Clients: {total_clients}\nParticipated: {participated} ({participated/total_clients*100:.1f}%)\nAvg Participation: {avg_participation:.2f} ± {std_participation:.2f}"
      ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

      plt.tight_layout()
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"Visualization saved to {save_path}")
      plt.show()
    
    def _fedavg_parameters(
        self, params_list: List[List[np.ndarray]], num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters using FedAvg (weighted averaging)."""
        if not params_list:
            return []

        print("==== aggregation===")
        total_samples = sum(num_samples_list)

        # Initialize aggregated parameters with zeros
        aggregated_params = [np.zeros_like(param) for param in params_list[0]]

        # Weighted sum of parameters
        for params, num_samples in zip(params_list, num_samples_list):
            for i, param in enumerate(params):
                aggregated_params[i] += param * num_samples

        # Weighted average of parameters
        aggregated_params = [param / total_samples for param in aggregated_params]

        return aggregated_params
    
    def save_client_mapping(self):

      df = pd.DataFrame([
    {"flower_uuid": uuid, "client_cid": cid}
    for uuid, cid in self.uuid_to_cid.items()
])

      df.to_csv("client_id_mapping.csv", index=False)
      print("Saved mapping at:")

      print(df)

    def _save_all_results(self):
      
        self.save_participation_stats()
        self.visualize_client_participation(self.client_participation_count, save_path="participation_chart.png", 
                                )    

    

    def save_participation_stats(self, filename="client_participation.csv"):
        """Save participation statistics at the end of training"""
        import pandas as pd
        
        # Create dataframe
        data = []
        for client_id, count in self.client_participation_count.items():
            data.append({
                'client_id': client_id,
                'participation_count': count,
                'participation_rate': count / self.total_rounds_completed
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('participation_count', ascending=False)
        df.to_csv(filename, index=False)
        print(f"Participation stats saved to {filename}")
        return df

        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        

        if not results:
            return None, {}
        accuracies = {}
     
        self.current_features = {}
        self.current_labels = {}
        # Extract all accuracies from evaluation

        accuracies = {}
        for client_proxy, eval_res in results:
            client_id = client_proxy.cid

            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies[f"client_{client_id}"] = accuracy
            metrics = eval_res.metrics
            # Get features and labels if available
           
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        # Only visualize if we have all the data and accuracy improved
        if avg_accuracy > self.best_avg_accuracy:
          print(f'==visualization===')
          self.best_avg_accuracy = avg_accuracy
        # Only visualize if we have all the data and accuracy improved
        log_filename = "fedavg_server_accuracy_log2.csv"
        write_header = not os.path.exists(log_filename)
        with open(log_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["round", "avg_accuracy"])
                writer.writerow([server_round, avg_accuracy])
        return avg_accuracy, {"accuracy": avg_accuracy}


def weighted_loss_avg(metrics: List[Tuple[float, int]]) -> float:
    
    if not metrics:
        return 0.0

    total_examples = sum([num_examples for _, num_examples in metrics])
    weighted_losses = [loss * num_examples for loss, num_examples in metrics]

    return sum(weighted_losses) / total_examples

