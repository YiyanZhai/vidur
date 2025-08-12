from typing import List

from vidur.events import BaseEvent, ApiRequestCompletionEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler, ApiBackend
from vidur.types import EventType

logger = init_logger(__name__)


class GlobalScheduleEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

        self._replica_set = []
        self._request_mapping = []

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        
        self._replica_set = set()
        self._request_mapping = scheduler.schedule()
        
        # threshold_ms = scheduler.config.router_config_ttft_threshold_ms  # add to your scheduler/config
        threshold_ms = 10
        # use_api = getattr(scheduler.config, "router_config_use_api", False)
        use_api = True
        if use_api:
            api = ApiBackend(
                fixed_overhead_ms=450, # scheduler.config.router_config_api_fixed_overhead_ms,
                per_token_ms=100, # scheduler.config.router_config_api_per_token_ms,
            )
            
        followup_events = []

        for replica_id, request in self._request_mapping:
            est_ms = scheduler.get_replica_scheduler(replica_id)\
                              .estimate_ttft_ms_if_enqueued_now(request)

            if use_api and est_ms > threshold_ms:
                completion_time = api.schedule(self.time, request)  # returns absolute time (sec)
                # followup_events.append(ApiRequestCompletionEvent(completion_time, request, ttft_ms=est_ms))
                continue
            
            # Enqueue locally
            self._replica_set.add(replica_id)
            scheduler.get_replica_scheduler(replica_id).add_request(request)

        # Trigger local replicas that got work
        followup_events.extend(
            [ReplicaScheduleEvent(self.time, rid) for rid in self._replica_set]
        )
        return followup_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_set": self._replica_set,
            "request_mapping": [
                (replica_id, request.id)
                for replica_id, request in self._request_mapping
            ],
        }
