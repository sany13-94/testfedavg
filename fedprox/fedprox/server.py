from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
import datetime
from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.distributions import Dirichlet, Categorical
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from flwr.common import GetPropertiesIns
import json
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from pathlib import Path
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import test,test_gpaf 
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
import csv
import requests
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import os
from fedprox.visualizeprototypes import ClusterVisualizationForConfigureFit

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

class GPAFStrategy(FedAvg):
    def __init__(
        self,
       experiment_name,
        num_classes: int=9,
        fraction_fit: float = 1.0,
        fraction_evaluate=1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients=2,
        min_available_clients=2,
        batch_size=32,
        ground_truth_stragglers=None,
         total_rounds: int = 15,
         
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.server_url = "https://add18b7094f7.ngrok-free.app/heartbeat"

        #clusters parameters
        self.warmup_rounds = 5 # Stage 1 duration
        self.num_clusters = 4
        self.client_assignments = {}  # {client_id: cluster_id}
        self.clustering_interval = 8
        # Simple participation counter
        self.client_participation_count = {}  # client_id -> number of times selected
        
        # Initialize as empty dictionaries
        self.cluster_prototypes = {i: {} for i in range(self.num_clusters)}
        self.cluster_class_counts = {i: defaultdict(int) for i in range(self.num_clusters)}
        map_path="client_id_mapping1.csv"
        
        
        self.theta = getattr(self, "theta", 0.65)          # optional threshold for s_c
        self.use_topk = getattr(self, "use_topk", True)    # prefer Top-K when you know |S_gt|

        self.uuid_to_cid = {}     # {"8325...": "client_0"}
        self.cid_to_uuid = {}     # {"client_0": "8325..."}
        self.ground_truth_cids = set(ground_truth_stragglers)  # {"client_0","client_1",...}
        self.ground_truth_flower_ids = set()  # will be filled as clients appear
 
        self._map_written = False
        
        # CSMDA Client Selection Parameters (UPDATED)
        self.training_times = defaultdict(float)
        self.selection_counts = defaultdict(int)
        self.accuracy_history = defaultdict(float)
        self._current_accuracies = {}
        # ... existing initialization ...
        # Core parameters from corrected methodology

        # Validation tracking
        self.validation_history = []  # Track predictions vs ground truth per round
               
        true_domain_labels = np.array([0]*5 + [1]*5 + [2]*4 + [0]*1)  # Adjust to your setup
        self.visualizer = ClusterVisualizationForConfigureFit(
            save_dir="./clustering_visualizations",
            true_domain_labels=true_domain_labels
        )
        self.virtual_cluster_id = 999
        
        # Tracking
        self.training_times = {}
        self.selection_counts = {}
        self.participated_clients = set()
        self.client_assignments = {}
        self.participated_clients = set()
        self.client_assignments = {}
        self.cluster_prototypes = {}
        self.last_round_participants = set()
        # Virtual cluster configuration
        self.use_virtual_cluster = True  # Enable virtual cluster for never-participated clients
        ema_alpha: float = 0.3  # EMA smoothing for training times
        beta: float = 1.5  # Penalty strength for reliability score
        initial_alpha1: float = 0.6  # Initial reliability weight
        initial_alpha2: float = 0.4  # Initial fairness weight
        phase_threshold: int = 20  # Round to switch weight emphasis
       
        self.total_rounds=total_rounds

        # Store ground truth straggler labels
        self.ground_truth_stragglers = ground_truth_stragglers  # Set of client IDs
        
        # NEW/MODIFIED FAIRNESS ATTRIBUTES
      
        reliability_lambda = 0.05
        acc_drop_threshold  = 0.005
       

        # NEW RELIABILITY ATTRIBUTE
        self.reliability_lambda = reliability_lambda

        self.phase_threshold = 30
        
        # CSMDA Hyperparameters
        self.alpha = 0.3  # EMA decay for training time
        self.epsilon = 0.1  # straggler tolerance (10% of T_max)
        self.phase_threshold = 30  # switch from reliability to fairness focus
        # EMA Training Time Tracking
        self.ema_alpha = ema_alpha
        self.training_times = {}  # T_c(i) - EMA of training times
        
        # Reliability Score Parameters
        self.beta = beta  # Penalty strength parameter
        
        # Fairness Tracking
        self.selection_counts = {}  # v_c - number of times client selected
        self.total_rounds_completed = 0  # T - total rounds
        
        # Weight Adaptation
        self.initial_alpha1 = initial_alpha1
        self.initial_alpha2 = initial_alpha2
        self.phase_threshold = phase_threshold
        #self.total_rounds = total_rounds
        
        print(f"[Init] Strategy initialized with α={ema_alpha}, β={beta}")

        # Initialize other components
        self.stat_util = {}
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy = 0.0
        self.batch_size = batch_size
        self.save_dir = "visualizations"

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
         experiment_id = mlflow.create_experiment(experiment_name)
         print(f"Created new experiment with ID: {experiment_id}")
         experiment = mlflow.get_experiment(experiment_id)
        else:
         print(f"Using existing experiment with ID: {experiment.experiment_id}")
      
        # Store MLflow reference
        self.mlflow = mlflow
        self.client_to_domain={}
        self.num_domains = self.min_fit_clients
        self.batch_size=batch_size
        self.save_dir="visualizations"
        
        print(f'num domain : {self.min_fit_clients}')
       
        #experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="server") as run:
         self.server_run_id = run.info.run_id
         # Log server parameters
         mlflow.log_params({
                "num_classes": num_classes,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
         
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
       
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.map_path = Path(map_path)
        expected_unique=self.min_fit_clients
        self.expected_unique = expected_unique
        # Track what we've already recorded: (client_cid, flower_node_id)
        self._seen= set()

        # If the CSV already exists, preload seen pairs (so we don't duplicate)
        if self.map_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.map_path, dtype=str)
                for _, r in df.iterrows():
                    self._seen.add((str(r["client_cid"]), str(r["flower_node_id"])))
            except Exception:
                pass  # if reading fails, start fresh in memory
         
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    
   
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
      try:
        """
        Aggregate model updates and updat
        """
        if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")
        
        if not results:
            print(f"[Round {server_round}] No clients returned results. Skipping aggregation.")
            return None, {}
        
        clients_params_list = []
        num_samples_list = []
        current_round_durations = []
        current_participants = set()
        new_rows: List[dict] = []

        # Create mapping dict if first time
        if not hasattr(self, "uuid_to_cid"):
          self.uuid_to_cid = {}

        # Process results and update tracking
        # Process results and update tracking
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            uuid = client_proxy.cid  # Flower internal UUID            
            cid = metrics.get("client_cid")
            node = metrics.get("flower_node_id")
            self.uuid_to_cid[uuid] = cid

            print(f'===client id: {cid} and flower id {uuid} and node :{node} ===')

           
            if client_id not in self.client_participation_count:
              self.client_participation_count[client_id] = 0
            self.client_participation_count[client_id] += 1
            
            self.participated_clients.add(client_id)
            current_participants.add(client_id)
          
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
        return ndarrays_to_parameters(aggregated_params), {}
      except Exception as e:
        print(f"[aggregate_fit] Error processing client {getattr(client_proxy,'cid','?')}: {e}")
        # continue to next client so we still reach the mapping update
        
   
    #mapping clients id in stragglers  
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
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.EvaluateRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes], Exception]],
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        print(f"[Server] Round {server_round}: {len(results)} clients evaluated, {len(failures)} failed evaluation.")
        
        aggregated_accuracy = 0.0
        if results:
            self._current_accuracies = {}
            for client_proxy, res in results:
                client_id = client_proxy.cid
                if "accuracy" in res.metrics:
                    client_accuracy = float(res.metrics["accuracy"])
                    self._current_accuracies[client_id] = client_accuracy
                    
                    with self.mlflow.start_run(run_id=self.server_run_id):
                        self.mlflow.log_metrics({
                            f"accuracy_client_{client_id}": client_accuracy
                        }, step=server_round)
                else:
                    print(f"[Warning] Client {client_id} did not report 'accuracy' in eval_res.metrics.")

                if "features" in res.metrics and "labels" in res.metrics:
                    try:
                        features_np = pickle.loads(base64.b64decode(res.metrics.get("features").encode('utf-8')))
                        labels_np = pickle.loads(base64.b64decode(res.metrics.get("labels").encode('utf-8')))
                        pass
                    except Exception as e:
                        print(f"[Warning] Failed to decode features/labels for client {client_id}: {e}")

            if self._current_accuracies:
                aggregated_accuracy = sum(self._current_accuracies.values()) / len(self._current_accuracies)
                print(f"[Server] Round {server_round}: Aggregated Average Accuracy: {aggregated_accuracy:.4f}")
                with self.mlflow.start_run(run_id=self.server_run_id):
                    self.mlflow.log_metrics({"avg_accuracy_global": aggregated_accuracy}, step=server_round)

            if aggregated_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = aggregated_accuracy
            
            log_filename = "server_accuracy_log2.csv"
            write_header = not os.path.exists(log_filename)
            with open(log_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["round", "avg_accuracy"])
                writer.writerow([server_round, aggregated_accuracy])

        else:
            print(f"[Server] Round {server_round}: No evaluation results received.")
            aggregated_accuracy = 0.0
            
        return None, {"accuracy": aggregated_accuracy}
   


    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
     
      sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
     
      print(f"Server sending round number: {server_round}")  # Debug print
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation=======')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
      

  

