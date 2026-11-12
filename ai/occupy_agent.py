"""Capture seat: distribute owned pieces and request cover via messages."""
from .base_agent import BaseAgent


class OccupyAgent(BaseAgent):
    AGENT_TYPE = 'occupy'
    CONTROLLED_SUB_TYPES = [2, 4, 16]

    def assign_targets(self, observation):
        cities = [c['coord'] for c in self.objective_cities(observation) if c['flag'] != self.color]
        assigned = set()
        cap_counts = {}
        ordered = sorted(self.my_ops, key=lambda op: min((self.get_distance(op['cur_hex'], p) for p in cities), default=0))
        for op in ordered:
            available = [p for p in cities if p not in assigned] or cities
            if not available:
                continue
            def cost(pos):
                threats = sum(self.get_distance(info['hex'], pos) <= 3
                              for info in self.enemy_positions.values()
                              if info.get('hex') is not None and info.get('blood', 1) > 0)
                return self.get_distance(op['cur_hex'], pos) + 3 * threats
            target = min(available, key=cost)
            assigned.add(target)
            cap_counts[target] = cap_counts.get(target, 0) + 1
            destination = self.staging_target(op, target) if self.squad_count > 1 and cap_counts[target] > 4 else target
            self.target_assignments[op['obj_id']] = {'target': destination, 'type': 'occupy'}

    def step(self, observation):
        self.start_step(observation)
        if observation['time']['stage'] == 1:
            return [{'actor': self.seat, 'type': 333}]
        actions = []
        for obj_id in self.controllable_ops:
            action = (self.gen_occupy(obj_id) or self.gen_shoot(obj_id)
                      or self.gen_get_off(obj_id) or self.gen_prepare(obj_id) or self.gen_move(obj_id))
            if action:
                actions.append(action)
        return actions
