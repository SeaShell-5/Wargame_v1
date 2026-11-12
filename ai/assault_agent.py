"""Assault seat: cover capture units and engage legal targets."""
from .base_agent import BaseAgent


class AssaultAgent(BaseAgent):
    AGENT_TYPE = 'assault'
    CONTROLLED_SUB_TYPES = [0, 1, 9, 13, 14, 15, 17, 18]

    def assign_targets(self, observation):
        cities = [c['coord'] for c in self.objective_cities(observation) if c['flag'] != self.color]
        allowed = {city['coord'] for city in self.objective_cities(observation)}
        requests = [p for p in self.support_requests.values() if p in allowed]
        targets = requests or cities or [c['coord'] for c in self.objective_cities(observation)]
        for op in self.my_ops:
            if targets:
                target = min(targets, key=lambda p: self.get_distance(op['cur_hex'], p))
                self.target_assignments[op['obj_id']] = {
                    'target': self.staging_target(op, target), 'type': 'cover_capture' if requests else 'advance'}

    def step(self, observation):
        self.start_step(observation)
        if observation['time']['stage'] == 1:
            return [{'actor': self.seat, 'type': 333}]
        actions = []
        for obj_id in self.controllable_ops:
            action = (self.gen_shoot(obj_id) or self.gen_occupy(obj_id)
                      or self.gen_get_off(obj_id) or self.gen_prepare(obj_id) or self.gen_move(obj_id))
            if action:
                actions.append(action)
        return actions
