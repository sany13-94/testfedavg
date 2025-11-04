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
import csv
import os
import base64
import pickle
import numpy as np
from flwr.server.client_manager import ClientManager
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

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
        **kwargs,
    ) -> None:
     super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs,
        )

     self.uuid_to_cid = {}     # {"8325...": "client_0"}
     self.cid_to_uuid = {}     # {"client_0": "8325..."}
     self.ground_truth_cids = set(ground_truth_stragglers)  # {"client_0","client_1",...}
     self.ground_truth_flower_ids = set()  # will be filled as clients appear
     self.total_rounds=total_rounds
     # mappings
     self.min_evaluate_clients=min_evaluate_clients
     self.min_available_clients=min_available_clients
     self.best_avg_accuracy=0.0
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
    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
   
      #sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=self.min_available_clients, min_num_clients=self.min_evaluate_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
      # Create EvaluateIns for each client
   
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    
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
            if prev is None:
                  ema = dur
                  print(f"[EMA Init] {uuid}: T_c(0) = {dur:.2f}s")
            else:
                ema = self.alpha * dur + (1.0 - self.alpha) * prev
                print(f"[EMA Update] {uuid}: {prev:.2f}s → {ema:.2f}s (raw: {dur:.2f}s)")
            self.training_times[uuid] = ema

           
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

        self._validate_straggler_predictions(server_round, results)
        
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

   
    def _predict_stragglers_from_score(self, T_max, client_ids):
      """Return set of predicted stragglers using s_c=1-As."""
      # compute scores for current participants only
      scores = {}
      for cid in client_ids:
        T_c = self.training_times.get(cid, 0.0)
        As = T_max / (T_c + self.beta * T_max) if (T_c > 0 and T_max > 0) else 0.0
        s_c = 1.0 - As
        scores[cid] = s_c
      """
      if self.use_topk:
        # Predict exactly as many as we injected (good for clean evaluation)
        k = len(self.ground_truth_flower_ids)  # see mapping below
        # sort by highest score (slowest)
        predicted = set(sorted(scores, key=scores.get, reverse=True)[:k])
      else:
      """
      # Thresholded prediction
      predicted = {cid for cid, s in scores.items() if s >= self.theta}
      return predicted, scores

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
        self.save_validation_results()
    def _norm(self,label: str) -> str:
      s = str(label).strip()
      return s.replace("client_", "")   # "client_0" -> "0"
    def _validate_straggler_predictions(self, server_round, results):
      # participants
      participants, round_dur = [], {}
      for client_proxy, fit_res in results:
        uuid = client_proxy.cid
        participants.append(uuid)
        if "duration" in fit_res.metrics:
            round_dur[uuid] = float(fit_res.metrics["duration"])

      # compute T_max from EMAs (assume you already updated EMA this round)
      valid_times = [t for t in self.training_times.values() if t is not None]
      if not valid_times:
        return
      T_max = float(np.mean(valid_times))

      # predict (your existing code)
      predicted_set, scores = self._predict_stragglers_from_score(T_max, participants)

      # robust ground-truth check: UUID OR logical label
      gt_uuid_set = self.cid_to_uuid    # UUIDs
      
      gt_logical_set = self.ground_truth_cids             # {"client_0","client_1",...}
      gt_idx_set = {
    int(cid.split("_", 1)[1])
    for cid in gt_logical_set
    if cid.startswith("client_") and cid.split("_", 1)[1].isdigit()
}  
    
      for uuid in participants:
          val = self.uuid_to_cid.get(uuid)  # could be "0" or 0 or None
          try:
            logical_idx = int(val) if val is not None else None
          except (TypeError, ValueError):
            logical_idx = None

          is_gt = (logical_idx is not None) and (logical_idx in gt_idx_set)
          print(f'===== {is_gt} and {self._norm(logical_idx)} and {gt_logical_set}')
          print(f'===== {gt_uuid_set} and {uuid} and {gt_idx_set}')

          rec = {
            "round": server_round,
            "client_id": uuid,
            "logical_id": logical_idx,
            "T_c": self.training_times.get(uuid, float("nan")),
            "T_max": T_max,
            "s_c": scores.get(uuid, float("nan")),
            "actual_duration": round_dur.get(uuid, float("nan")),
            "predicted_straggler": uuid in predicted_set,
            "ground_truth_straggler": is_gt,                         # <-- now correct
        }
          rec["prediction_type"] = self._classify_prediction(rec["predicted_straggler"], rec["ground_truth_straggler"])
          self.validation_history.append(rec)

            
    #strqgglers 

    def _classify_prediction(self, predicted, actual):
        """Classify prediction type for confusion matrix"""
        if predicted and actual:
            return 'True Positive'  # Correctly identified straggler
        elif not predicted and not actual:
            return 'True Negative'  # Correctly identified fast client
        elif predicted and not actual:
            return 'False Positive'  # Wrongly labeled fast client as straggler
        else:  # not predicted and actual
            return 'False Negative'  # Missed a straggler
    
    def save_validation_results(self, filename="validation_results.csv"):
        """Save validation results"""
        
        
        df = pd.DataFrame(self.validation_history)
        df.to_csv(filename, index=False)
        print(f"Validation results saved to {filename}")
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

