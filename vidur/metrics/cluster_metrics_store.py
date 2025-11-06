import copy
import json
import os
from typing import Dict, List

import pandas as pd
import wandb

from vidur.config import SimulationConfig
from vidur.entities.batch import Batch
from vidur.entities.replica import Replica
from vidur.entities.request import Request
from vidur.metrics.data_structures.cdf_sketch import CDFSketch
from vidur.metrics.data_structures.data_series import DataSeries
from vidur.metrics.metrics import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    BatchMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    RequestMetricsTimeDistributions,
    RequestMetricsTimeSeries,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeSeries,
)
from vidur.metrics.replica_metrics_store import ReplicaMetricsStore
from vidur.types.replica_id import ReplicaId


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
TIME_STR_MS = "Time (ms)"
TIME_STR = "Time (sec)"
COUNT_STR = "Count"
MEMORY_USAGE_STR = "Memory Usage (%)"


class ClusterMetricsStore:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
        global_scheduler=None,
    ):
        self._simulation_config = simulation_config
        self._config = self._simulation_config.metrics_config
        self._replicas = replicas
        self._global_scheduler = global_scheduler

        """
        TODO: We need to use a minimal `ClusterMetricsStore`.
        Explicitly not collect batch, utilization, operation metrics at cluster level as they are replica specific.
        """
        simulation_config_copy = copy.deepcopy(simulation_config)
        simulation_config_copy.metrics_config.store_batch_metrics = False
        simulation_config_copy.metrics_config.store_utilization_metrics = False
        simulation_config_copy.metrics_config.store_operation_metrics = False
        self._cluster_metric_store = ReplicaMetricsStore(
            simulation_config=simulation_config_copy,
        )

        # We use str(replica_id) as the key to avoid JSON encoding issues inside wandb.log
        self._replica_metric_stores = {
            str(replica_id): ReplicaMetricsStore(
                simulation_config=self._simulation_config,
                replica_id=replica_id,
            )
            for replica_id in replicas.keys()
        }

        self._wandb_project = self._config.wandb_project
        self._wandb_group = self._config.wandb_group
        self._wandb_run_name = self._config.wandb_run_name

        self._init_wandb()

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._wandb_project
            or not self._wandb_group
        ):
            return

        wandb.init(
            project=self._wandb_project,
            group=self._wandb_group,
            name=self._wandb_run_name,
            config=self._simulation_config.to_dict(),
        )

    def _save_as_csv(
        self,
        df: pd.DataFrame,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)
        # Print only upto 6 decimal places (micros precision) to reduce csv size
        df.to_csv(f"{base_path}/{file_name}.csv", float_format="%.6f", index=False)
        if wandb.run and self._config.save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{file_name}_table": wand_table}, step=0)

    def _save_as_json(self, data, base_path: str, file_name: str):
        os.makedirs(base_path, exist_ok=True)
        with open(f"{base_path}/{file_name}.json", "w") as f:
            json.dump(data, f)

        if wandb.run and self._config.save_table_to_wandb:
            wandb.log({f"{file_name}": data}, step=0)

    def _store_request_metrics(self, base_plot_path: str):
        if not self._config.store_request_metrics:
            return

        request_metrics_df = pd.DataFrame()
        for replica_id, store in self._replica_metric_stores.items():
            request_metrics_df = pd.concat(
                [request_metrics_df, store.get_request_metrics_df()]
            )
        request_metrics_df.sort_values(by=REQUEST_ID_STR, inplace=True)
        self._save_as_csv(
            df=request_metrics_df,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        # Log prefix cache metrics
        prefix_cache_metrics = {}
        for replica_id, store in self._replica_metric_stores.items():
            prefix_cache_stats = store.get_prefix_cache_stats()
            prefix_cache_metrics[replica_id] = prefix_cache_stats
        self._save_as_json(
            data=prefix_cache_metrics,
            base_path=base_plot_path,
            file_name="replica_prefix_cache_metrics",
        )

        # Print replica wise metrics
        if len(self._replica_metric_stores) > 1:
            for metric_name in RequestMetricsTimeDistributions:
                replica_wise_dict = {}
                for replica_id, store in self._replica_metric_stores.items():
                    replica_wise_dict[replica_id] = (
                        store._request_metrics_time_distributions[metric_name]
                    )
                DataSeries.plot_cdfs(
                    replica_wise_dict,
                    base_plot_path,
                    f"{metric_name.value}_replicawise",
                    y_axis_label=TIME_STR,
                    save_plot=self._config.store_plots,
                )

            for metric_name in RequestMetricsTimeSeries:
                replica_wise_dict = {}
                for replica_id, store in self._replica_metric_stores.items():
                    replica_wise_dict[replica_id] = store._request_metrics_time_series[
                        metric_name
                    ]
                DataSeries.plot_steps(
                    replica_wise_dict,
                    base_plot_path,
                    f"{metric_name.value}_timeseries_replicawise",
                    y_axis_label=TIME_STR,
                    save_plot=self._config.store_plots,
                )

    def _store_outsourcing_metrics(self, base_plot_path: str):
        """Store outsourcing statistics and details."""
        # Check if we have access to the global scheduler
        if not self._global_scheduler:
            return
            
        # Collect outsourcing statistics from all replicas
        outsourcing_stats = {}
        all_outsourced_details = []
        
        for replica_id in self._replicas.keys():
            # Get the replica scheduler from global scheduler
            replica_scheduler = self._global_scheduler.get_replica_scheduler(replica_id)
            
            # Check if replica scheduler has outsourcing methods
            if hasattr(replica_scheduler, 'get_outsourcing_statistics'):
                stats = replica_scheduler.get_outsourcing_statistics()
                outsourcing_stats[str(replica_id)] = stats
                
                # Collect detailed outsourced request information
                if hasattr(replica_scheduler, 'get_outsourced_request_details'):
                    details = replica_scheduler.get_outsourced_request_details()
                    all_outsourced_details.extend(details)
        
        # Save outsourcing statistics as JSON
        if outsourcing_stats:
            self._save_as_json(
                data=outsourcing_stats,
                base_path=base_plot_path,
                file_name="outsourcing_statistics",
            )
            
            # Calculate hypothetical cost if all requests were outsourced
            # Get all request metrics to calculate total tokens across ALL requests (including outsourced)
            total_prefill_tokens = 0
            total_decode_tokens = 0
            total_requests_completed_locally = 0
            
            # First, get tokens from locally completed requests
            for replica_id, store in self._replica_metric_stores.items():
                # Get request metrics dataframe
                request_df = store.get_request_metrics_df()
                if not request_df.empty:
                    # Sum up tokens from all requests
                    if 'request_num_prefill_tokens' in request_df.columns:
                        total_prefill_tokens += request_df['request_num_prefill_tokens'].sum()
                    if 'request_num_decode_tokens' in request_df.columns:
                        total_decode_tokens += request_df['request_num_decode_tokens'].sum()
                    total_requests_completed_locally += len(request_df)
            
            # Add tokens from outsourced requests
            outsourced_prefill_tokens = sum(s['total_input_tokens'] for s in outsourcing_stats.values())
            outsourced_decode_tokens = sum(s['total_output_tokens'] for s in outsourcing_stats.values())
            outsourced_count = sum(s['total_outsourced'] for s in outsourcing_stats.values())
            
            total_prefill_tokens += outsourced_prefill_tokens
            total_decode_tokens += outsourced_decode_tokens
            total_requests = total_requests_completed_locally + outsourced_count
            
            # Calculate hypothetical OpenAI API cost using same pricing model (for ChatGPT-5)
            # Input: $1.25 per 1M tokens, Output: $10.00 per 1M tokens
            input_price_per_million = 1.25
            output_price_per_million = 10.00
            
            hypothetical_total_cost = (
                (total_prefill_tokens / 1_000_000) * input_price_per_million +
                (total_decode_tokens / 1_000_000) * output_price_per_million
            )
            
            # Calculate and save cluster-wide statistics
            actual_cost = sum(s['total_api_cost_usd'] for s in outsourcing_stats.values())
            
            cluster_stats = {
                'total_outsourced': sum(s['total_outsourced'] for s in outsourcing_stats.values()),
                'total_api_cost_usd': actual_cost,
                'total_input_tokens': sum(s['total_input_tokens'] for s in outsourcing_stats.values()),
                'total_output_tokens': sum(s['total_output_tokens'] for s in outsourcing_stats.values()),
                'outsourced_from_waiting': sum(s['outsourced_from_waiting'] for s in outsourcing_stats.values()),
                'outsourced_from_running': sum(s['outsourced_from_running'] for s in outsourcing_stats.values()),
                # Hypothetical cost calculations
                'hypothetical_all_outsourced': {
                    'total_requests': total_requests,
                    'total_input_tokens': int(total_prefill_tokens),
                    'total_output_tokens': int(total_decode_tokens),
                    'total_api_cost_usd': hypothetical_total_cost,
                },
                # Cost savings
                'cost_savings_usd': hypothetical_total_cost - actual_cost,
                'cost_savings_percent': ((hypothetical_total_cost - actual_cost) / hypothetical_total_cost * 100) if hypothetical_total_cost > 0 else 0,
            }
            
            self._save_as_json(
                data=cluster_stats,
                base_path=base_plot_path,
                file_name="cluster_outsourcing_statistics",
            )
        
        # Save detailed outsourced request information as CSV
        if all_outsourced_details:
            outsourced_df = pd.DataFrame(all_outsourced_details)
            self._save_as_csv(
                df=outsourced_df,
                base_path=self._config.output_dir,
                file_name="outsourced_requests",
            )
            
            # Plot token distribution histograms comparing outsourced vs local requests
            self._plot_token_distribution_comparison(base_plot_path, all_outsourced_details)

    def _plot_token_distribution_comparison(self, base_plot_path: str, outsourced_details: List[dict]):
        """Plot histograms comparing token distributions between outsourced and local requests."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            # matplotlib not available, skip plotting
            return
        
        if not self._config.store_plots:
            return
            
        # Collect local request data
        local_prefill_tokens = []
        local_decode_tokens = []
        
        for replica_id, store in self._replica_metric_stores.items():
            request_df = store.get_request_metrics_df()
            print("request_df in plotting:", request_df)
            if not request_df.empty:
                if 'request_num_prefill_tokens' in request_df.columns:
                    local_prefill_tokens.extend(request_df['request_num_prefill_tokens'].tolist())
                if 'request_num_decode_tokens' in request_df.columns:
                    local_decode_tokens.extend(request_df['request_num_decode_tokens'].tolist())
        
        # Collect outsourced request data
        outsourced_prefill_tokens = [d['num_prefill_tokens'] for d in outsourced_details]
        outsourced_decode_tokens = [d['num_decode_tokens'] for d in outsourced_details]
        
        # Only plot if we have data
        if not (local_prefill_tokens or outsourced_prefill_tokens):
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Prefill Tokens
        if local_prefill_tokens or outsourced_prefill_tokens:
            bins = np.linspace(
                min((min(local_prefill_tokens) if local_prefill_tokens else float('inf')),
                    (min(outsourced_prefill_tokens) if outsourced_prefill_tokens else float('inf'))),
                max((max(local_prefill_tokens) if local_prefill_tokens else 0),
                    (max(outsourced_prefill_tokens) if outsourced_prefill_tokens else 0)),
                50
            )
            
            if local_prefill_tokens:
                ax1.hist(local_prefill_tokens, bins=bins, alpha=0.6, label=f'Local (n={len(local_prefill_tokens)})', color='blue', edgecolor='black')
            if outsourced_prefill_tokens:
                ax1.hist(outsourced_prefill_tokens, bins=bins, alpha=0.6, label=f'Outsourced (n={len(outsourced_prefill_tokens)})', color='red', edgecolor='black')
            
            ax1.set_xlabel('Number of Prefill Tokens', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Distribution of Prefill Tokens: Local vs Outsourced Requests', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Decode Tokens
        if local_decode_tokens or outsourced_decode_tokens:
            bins = np.linspace(
                min((min(local_decode_tokens) if local_decode_tokens else float('inf')),
                    (min(outsourced_decode_tokens) if outsourced_decode_tokens else float('inf'))),
                max((max(local_decode_tokens) if local_decode_tokens else 0),
                    (max(outsourced_decode_tokens) if outsourced_decode_tokens else 0)),
                50
            )
            
            if local_decode_tokens:
                ax2.hist(local_decode_tokens, bins=bins, alpha=0.6, label=f'Local (n={len(local_decode_tokens)})', color='blue', edgecolor='black')
            if outsourced_decode_tokens:
                ax2.hist(outsourced_decode_tokens, bins=bins, alpha=0.6, label=f'Outsourced (n={len(outsourced_decode_tokens)})', color='red', edgecolor='black')
            
            ax2.set_xlabel('Number of Decode Tokens', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Distribution of Decode Tokens: Local vs Outsourced Requests', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = f"{base_plot_path}/token_distribution_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual plots for better detail
        self._plot_individual_token_histograms(base_plot_path, local_prefill_tokens, local_decode_tokens,
                                               outsourced_prefill_tokens, outsourced_decode_tokens)
    
    def _plot_individual_token_histograms(self, base_plot_path: str, 
                                         local_prefill: List, local_decode: List,
                                         outsourced_prefill: List, outsourced_decode: List):
        """Create individual histogram plots for each token type."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return
        
        # Prefill tokens only
        if local_prefill or outsourced_prefill:
            fig, ax = plt.subplots(figsize=(10, 6))
            bins = np.linspace(
                min((min(local_prefill) if local_prefill else float('inf')),
                    (min(outsourced_prefill) if outsourced_prefill else float('inf'))),
                max((max(local_prefill) if local_prefill else 0),
                    (max(outsourced_prefill) if outsourced_prefill else 0)),
                50
            )
            
            if local_prefill:
                ax.hist(local_prefill, bins=bins, alpha=0.6, label=f'Local (n={len(local_prefill)}, mean={np.mean(local_prefill):.0f})', color='blue', edgecolor='black')
            if outsourced_prefill:
                ax.hist(outsourced_prefill, bins=bins, alpha=0.6, label=f'Outsourced (n={len(outsourced_prefill)}, mean={np.mean(outsourced_prefill):.0f})', color='red', edgecolor='black')
            
            ax.set_xlabel('Number of Prefill Tokens', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Prefill Token Distribution: Local vs Outsourced', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{base_plot_path}/prefill_tokens_histogram.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Decode tokens only
        if local_decode or outsourced_decode:
            fig, ax = plt.subplots(figsize=(10, 6))
            bins = np.linspace(
                min((min(local_decode) if local_decode else float('inf')),
                    (min(outsourced_decode) if outsourced_decode else float('inf'))),
                max((max(local_decode) if local_decode else 0),
                    (max(outsourced_decode) if outsourced_decode else 0)),
                50
            )
            
            if local_decode:
                ax.hist(local_decode, bins=bins, alpha=0.6, label=f'Local (n={len(local_decode)}, mean={np.mean(local_decode):.0f})', color='blue', edgecolor='black')
            if outsourced_decode:
                ax.hist(outsourced_decode, bins=bins, alpha=0.6, label=f'Outsourced (n={len(outsourced_decode)}, mean={np.mean(outsourced_decode):.0f})', color='red', edgecolor='black')
            
            ax.set_xlabel('Number of Decode Tokens', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Decode Token Distribution: Local vs Outsourced', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{base_plot_path}/decode_tokens_histogram.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _store_running_request_events(self, base_plot_path: str):
        """Store running request execution events."""
        all_running_events = []
        
        for replica_id, store in self._replica_metric_stores.items():
            if hasattr(store, 'get_running_request_events'):
                events = store.get_running_request_events()
                all_running_events.extend(events)
        
        # Save running request events as CSV
        if all_running_events:
            running_df = pd.DataFrame(all_running_events)
            # Sort by event time for better readability
            running_df.sort_values(by='event_time', inplace=True)
            self._save_as_csv(
                df=running_df,
                base_path=self._config.output_dir,
                file_name="running_requests",
            )

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        if self._config.keep_individual_batch_metrics:
            batch_metrics_df = pd.DataFrame()
            for replica_id, store in self._replica_metric_stores.items():
                batch_metrics_df = pd.concat(
                    [batch_metrics_df, store.get_batch_metrics_df()]
                )
            self._save_as_csv(
                df=batch_metrics_df,
                base_path=self._config.output_dir,
                file_name="batch_metrics",
            )

        for metric_name in BatchMetricsTimeDistribution:
            y_axis_label = (
                TIME_STR_MS if "model_execution" in metric_name.value else TIME_STR
            )

            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._batch_metrics_time_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label,
                save_plot=self._config.store_plots,
            )

        for metric_name in BatchMetricsCountDistribution:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._batch_metrics_count_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label=COUNT_STR,
                save_plot=self._config.store_plots,
            )

        # if self._config.keep_individual_batch_metrics:
        #     for metric_name in BatchMetricsTimeSeries:
        #         replica_wise_dict = {}
        #         for replica_id, store in self._replica_metric_stores.items():
        #             replica_wise_dict[replica_id] = store._batch_metrics_time_series[
        #                 metric_name
        #             ]
        #         DataSeries.plot_steps(
        #             replica_wise_dict,
        #             base_plot_path,
        #             f"{metric_name.value}_replicawise",
        #             y_axis_label=metric_name.value,
        #             save_plot=self._config.store_plots,
        #             y_cumsum=False,
        #         )

    def _store_token_metrics(self, base_plot_path: str):
        if not self._config.store_token_completion_metrics:
            return
        for metric_name in TokenMetricsTimeDistribution:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._token_metrics_time_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label=TIME_STR,
                save_plot=self._config.store_plots,
            )

        for metric_name in TokenMetricsTimeSeries:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._token_metrics_time_series[
                    metric_name
                ]
            DataSeries.plot_steps(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_timeseries_replicawise",
                y_axis_label=COUNT_STR,
                save_plot=self._config.store_plots,
            )

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:
            return

        if self._config.keep_individual_batch_metrics:
            op_metrics_df = pd.DataFrame()
            for replica_id, store in self._replica_metric_stores.items():
                op_metrics_df = pd.concat(
                    [op_metrics_df, store.get_operation_metrics_df()]
                )
            self._save_as_csv(
                df=op_metrics_df,
                base_path=self._config.output_dir,
                file_name="operation_metrics",
            )

        for metric_name in OperationMetrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._operation_metrics[metric_name]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_execution_time",
                y_axis_label=TIME_STR_MS,
                save_plot=self._config.store_plots,
            )

        for metric_name in CpuOperationMetrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._cpu_operation_metrics[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_execution_time",
                y_axis_label=TIME_STR_MS,
                save_plot=self._config.store_plots,
            )

    def _store_utilization_metrics(self, base_plot_path: str):
        if not self._config.store_utilization_metrics:
            return

        replica_memory_usage = {}
        replica_busy_time = {}
        replica_mfu = {}
        for replica_id, store in self._replica_metric_stores.items():
            replica_memory_usage[str(replica_id)] = (
                store._replica_memory_usage.get_stats("replica_memory_usage")
            )
            replica_busy_time[str(replica_id)] = store.get_replica_busy_time()
            replica_mfu[str(replica_id)] = store.get_replica_mfu()

        self._save_as_json(replica_memory_usage, base_plot_path, "replica_memory_usage")
        self._save_as_json(replica_busy_time, base_plot_path, "replica_busy_time")
        self._save_as_json(replica_mfu, base_plot_path, "replica_mfu")

        if self._config.keep_individual_batch_metrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._replica_memory_usage_per_batch
            # TODO: Fix perf and enable plotting the memory usage wrt time
            DataSeries.plot_steps(
                replica_wise_dict,
                base_plot_path,
                "replica_memory_usage_time_series",
                y_axis_label=MEMORY_USAGE_STR,
                save_plot=False,
                y_cumsum=False,
            )

    @if_write_metrics
    def plot(self, sim_time: float) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._cluster_metric_store.store_metrics(dir_plot_path, sim_time)
        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)
        self._store_token_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics(dir_plot_path)
        self._store_outsourcing_metrics(dir_plot_path)
        self._store_running_request_events(dir_plot_path)

    def on_batch_end(
        self, time: float, batch, replica_id: ReplicaId, memory_usage_percent: float
    ):
        self._cluster_metric_store.on_batch_end(time, batch, memory_usage_percent)
        self._replica_metric_stores[str(replica_id)].on_batch_end(
            time, batch, memory_usage_percent
        )

    def on_batch_stage_end(
        self, batch_stage, time: float, replica_id: ReplicaId, stage_id: int
    ):
        self._replica_metric_stores[str(replica_id)].on_batch_stage_end(
            batch_stage, time, stage_id
        )

    def on_replica_schedule(
        self,
        time: float,
        replica_id: ReplicaId,
        batches: List[Batch],
        memory_usage_percent: float,
    ):
        self._replica_metric_stores[str(replica_id)].on_replica_schedule(
            time, memory_usage_percent
        )
        newly_scheduled_requests = [
            request
            for batch in batches
            for request in batch.requests
            if request.scheduled_at == time
        ]
        for request in newly_scheduled_requests:
            self._replica_metric_stores[str(request.replica_id)].on_request_arrival(
                request
            )

    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: ReplicaId,
        stage_id: int,
        batch_stage,
        execution_time: float,
    ):
        self._replica_metric_stores[str(replica_id)].on_replica_stage_schedule(
            time, stage_id, batch_stage, execution_time
        )

    def on_request_arrival(self, request: Request):
        self._cluster_metric_store.on_request_arrival(request)

    def on_request_end(self, request: Request):
        self._cluster_metric_store.on_request_end(request)
        self._replica_metric_stores[str(request.replica_id)].on_request_end(request)
