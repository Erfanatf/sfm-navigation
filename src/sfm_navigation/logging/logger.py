from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


def _mood_name(mood):
    """Return a string name for a mood (enum or custom string)."""
    return mood if isinstance(mood, str) else mood.name

@dataclass
class CollisionEvent:
    time: float
    pedestrian_id: int
    collision_type: str
    other_id: Optional[int] = None
    position: Tuple[float, float] = (0.0, 0.0)

@dataclass
class TemporalWindowLog:
    window_id: int
    start_time: float
    end_time: float
    pedestrians_in_vicinity: List[int]
    pedestrian_moods: Dict[int, str]
    pedestrian_positions: Dict[int, Tuple[float, float]]
    user_position: Tuple[float, float]
    collisions_in_window: List[CollisionEvent]
    n_pedestrians: int = 0
    mood_distribution: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.n_pedestrians = len(self.pedestrians_in_vicinity)
        if not self.mood_distribution:
            self.mood_distribution = {}
            for mood in self.pedestrian_moods.values():
                self.mood_distribution[mood] = self.mood_distribution.get(mood, 0) + 1


class SimulationLogger:
    def __init__(self, window_duration: float = 5.0, vicinity_radius: float = 10.0):
        self.window_duration = window_duration
        self.vicinity_radius = vicinity_radius
        self.window_logs: List[TemporalWindowLog] = []
        self.current_window_id = 0
        self.current_window_start = 0.0
        self.current_window_collisions: List[CollisionEvent] = []
        self.pedestrian_collision_counts: Dict[int, Dict[str, int]] = {}
        self.collision_distance_user = 0.5
        self.collision_distance_robot = 0.5
        self.collision_distance_obstacle = 0.1
        self.collision_distance_pedestrian = 0.1
        self.last_collision_time: Dict[int, Dict[str, float]] = {}
        self.collision_cooldown = 0.5
        self.total_simulation_time = 0.0

    def _get_or_create_collision_tracker(self, ped_id: int) -> Dict[str, int]:
        if ped_id not in self.pedestrian_collision_counts:
            self.pedestrian_collision_counts[ped_id] = {
                'user': 0, 'robot': 0, 'static_obstacle': 0, 'pedestrian': 0, 'total': 0
            }
        if ped_id not in self.last_collision_time:
            self.last_collision_time[ped_id] = {}
        return self.pedestrian_collision_counts[ped_id]

    def _can_register_collision(self, ped_id: int, collision_type: str, current_time: float) -> bool:
        if ped_id not in self.last_collision_time:
            return True
        last_time = self.last_collision_time[ped_id].get(collision_type, -999)
        return (current_time - last_time) >= self.collision_cooldown

    def check_collisions(self, current_time: float, pedestrians: List,
                         user_pos: Tuple[float, float],
                         robot_pos: Optional[Tuple[float, float]] = None,
                         static_obstacles: List = None):
        for ped in pedestrians:
            pid = ped.pedestrian_id
            tracker = self._get_or_create_collision_tracker(pid)
            dist_to_user = np.sqrt((ped.x - user_pos[0])**2 + (ped.y - user_pos[1])**2)
            if dist_to_user < (ped.radius + self.collision_distance_user):
                if self._can_register_collision(pid, 'user', current_time):
                    tracker['user'] += 1
                    tracker['total'] += 1
                    self.last_collision_time[pid]['user'] = current_time
                    self.current_window_collisions.append(CollisionEvent(
                        time=current_time, pedestrian_id=pid, collision_type='user',
                        position=(ped.x, ped.y)
                    ))
            if robot_pos is not None:
                dist_to_robot = np.sqrt((ped.x - robot_pos[0])**2 + (ped.y - robot_pos[1])**2)
                if dist_to_robot < (ped.radius + self.collision_distance_robot):
                    if self._can_register_collision(pid, 'robot', current_time):
                        tracker['robot'] += 1
                        tracker['total'] += 1
                        self.last_collision_time[pid]['robot'] = current_time
                        self.current_window_collisions.append(CollisionEvent(
                            time=current_time, pedestrian_id=pid, collision_type='robot',
                            position=(ped.x, ped.y)
                        ))
            if static_obstacles:
                for obs in static_obstacles:
                    ox = obs.x if hasattr(obs, 'x') else obs[0]
                    oy = obs.y if hasattr(obs, 'y') else obs[1]
                    orad = obs.radius if hasattr(obs, 'radius') else obs[2]
                    oid = obs.obstacle_id if hasattr(obs, 'obstacle_id') else -1
                    dist_to_obs = np.sqrt((ped.x - ox)**2 + (ped.y - oy)**2)
                    if dist_to_obs < (ped.radius + orad + self.collision_distance_obstacle):
                        collision_key = f'static_{oid}'
                        if self._can_register_collision(pid, collision_key, current_time):
                            tracker['static_obstacle'] += 1
                            tracker['total'] += 1
                            self.last_collision_time[pid][collision_key] = current_time
                            self.current_window_collisions.append(CollisionEvent(
                                time=current_time, pedestrian_id=pid,
                                collision_type='static_obstacle', other_id=oid,
                                position=(ped.x, ped.y)
                            ))
            for other_ped in pedestrians:
                if other_ped.pedestrian_id == pid:
                    continue
                dist = np.sqrt((ped.x - other_ped.x)**2 + (ped.y - other_ped.y)**2)
                if dist < (ped.radius + other_ped.radius + self.collision_distance_pedestrian):
                    collision_key = f'ped_{other_ped.pedestrian_id}'
                    if self._can_register_collision(pid, collision_key, current_time):
                        tracker['pedestrian'] += 1
                        tracker['total'] += 1
                        self.last_collision_time[pid][collision_key] = current_time
                        self.current_window_collisions.append(CollisionEvent(
                            time=current_time, pedestrian_id=pid,
                            collision_type='pedestrian', other_id=other_ped.pedestrian_id,
                            position=(ped.x, ped.y)
                        ))

    def update(self, current_time: float, pedestrians: List,
               user_pos: Tuple[float, float],
               robot_pos: Optional[Tuple[float, float]] = None,
               static_obstacles: List = None):
        self.total_simulation_time = current_time
        self.check_collisions(current_time, pedestrians, user_pos, robot_pos, static_obstacles)
        if current_time >= self.current_window_start + self.window_duration:
            self._close_current_window(current_time, pedestrians, user_pos)


    def _close_current_window(self, current_time: float, pedestrians: List,
                               user_pos: Tuple[float, float]):
        peds_in_vicinity = []
        ped_moods = {}
        ped_positions = {}
        for ped in pedestrians:
            dist_to_user = np.sqrt((ped.x - user_pos[0])**2 + (ped.y - user_pos[1])**2)
            if dist_to_user <= self.vicinity_radius:
                peds_in_vicinity.append(ped.pedestrian_id)
                ped_moods[ped.pedestrian_id] = _mood_name(ped.mood)
                ped_positions[ped.pedestrian_id] = (ped.x, ped.y)
        window_log = TemporalWindowLog(
            window_id=self.current_window_id,
            start_time=self.current_window_start,
            end_time=min(current_time, self.current_window_start + self.window_duration),
            pedestrians_in_vicinity=peds_in_vicinity,
            pedestrian_moods=ped_moods,
            pedestrian_positions=ped_positions,
            user_position=user_pos,
            collisions_in_window=self.current_window_collisions.copy()
        )
        self.window_logs.append(window_log)
        self.current_window_id += 1
        self.current_window_start = self.current_window_start + self.window_duration
        self.current_window_collisions = []

    def finalize(self, final_time: float, pedestrians: List, user_pos: Tuple[float, float]):
        if self.current_window_start < final_time:
            self._close_current_window(final_time, pedestrians, user_pos)

    def get_summary(self) -> dict:
        total_collisions = sum(
            counts['total'] for counts in self.pedestrian_collision_counts.values()
        )
        mood_totals = {}
        for window in self.window_logs:
            for mood, count in window.mood_distribution.items():
                mood_totals[mood] = mood_totals.get(mood, 0) + count
        return {
            'total_simulation_time': self.total_simulation_time,
            'window_duration': self.window_duration,
            'n_windows': len(self.window_logs),
            'total_collisions': total_collisions,
            'collision_breakdown': {
                'user': sum(c['user'] for c in self.pedestrian_collision_counts.values()),
                'robot': sum(c['robot'] for c in self.pedestrian_collision_counts.values()),
                'static_obstacle': sum(c['static_obstacle'] for c in self.pedestrian_collision_counts.values()),
                'pedestrian': sum(c['pedestrian'] for c in self.pedestrian_collision_counts.values()),
            },
            'per_pedestrian_collisions': dict(self.pedestrian_collision_counts),
            'mood_distribution_total': mood_totals,
            'avg_pedestrians_in_vicinity': np.mean([w.n_pedestrians for w in self.window_logs]) if self.window_logs else 0
        }

    def get_window_logs(self) -> List[dict]:
        logs = []
        for w in self.window_logs:
            logs.append({
                'window_id': w.window_id,
                'start_time': w.start_time,
                'end_time': w.end_time,
                'n_pedestrians': w.n_pedestrians,
                'pedestrians_in_vicinity': w.pedestrians_in_vicinity,
                'pedestrian_moods': w.pedestrian_moods,
                'mood_distribution': w.mood_distribution,
                'user_position': w.user_position,
                'n_collisions': len(w.collisions_in_window),
                'collisions': [
                    {'time': c.time, 'ped_id': c.pedestrian_id,
                     'type': c.collision_type, 'other_id': c.other_id}
                    for c in w.collisions_in_window
                ]
            })
        return logs

    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "="*60)
        print("SIMULATION LOG SUMMARY")
        print("="*60)
        print(f"Total simulation time: {summary['total_simulation_time']:.1f}s")
        print(f"Window duration: {summary['window_duration']:.1f}s")
        print(f"Number of temporal windows: {summary['n_windows']}")
        print(f"Average pedestrians in vicinity: {summary['avg_pedestrians_in_vicinity']:.1f}")
        print(f"\nTotal collisions: {summary['total_collisions']}")
        print(f"  - With User: {summary['collision_breakdown']['user']}")
        print(f"  - With Robot: {summary['collision_breakdown']['robot']}")
        print(f"  - With Static Obstacles: {summary['collision_breakdown']['static_obstacle']}")
        print(f"  - With Other Pedestrians: {summary['collision_breakdown']['pedestrian']}")
        print(f"\nMood distribution (total across all windows):")
        for mood, count in sorted(summary['mood_distribution_total'].items()):
            print(f"  - {mood}: {count}")
        print(f"\nPer-pedestrian collision counts:")
        for pid, counts in sorted(summary['per_pedestrian_collisions'].items()):
            if counts['total'] > 0:
                print(f"  Ped {pid}: {counts['total']} total "
                      f"(U:{counts['user']}, R:{counts['robot']}, "
                      f"O:{counts['static_obstacle']}, P:{counts['pedestrian']})")
        print("="*60)