"""Support seat: respond to capture requests and shared enemy locations."""
from .base_agent import BaseAgent, ActionType


class SupportAgent(BaseAgent):
    AGENT_TYPE = 'support'
    CONTROLLED_SUB_TYPES = [3, 5, 6, 7, 8, 10, 19]

    def assign_targets(self, observation):
        enemy = self.get_enemy_center()
        requests = list(self.support_requests.values())
        for op in self.my_ops:
            if enemy is not None:
                target, mission = enemy, 'fire_support'
            elif requests:
                target, mission = min(requests, key=lambda p: self.get_distance(op['cur_hex'], p)), 'cover_capture'
            else:
                # Follow a teammate only through its status message, never global unit data.
                peers = [u['hex'] for status in self.team_status.values() for u in status['units']]
                if not peers:
                    continue
                target, mission = min(peers, key=lambda p: self.get_distance(op['cur_hex'], p)), 'follow'
            self.target_assignments[op['obj_id']] = {'target': target, 'type': mission}

    def gen_jm_plan(self, obj_id):
        candidates = self.candidates(obj_id).get(ActionType.JMPlan) or []
        target = self.get_enemy_center()
        if candidates and target is not None:
            friendly = [op['cur_hex'] for op in self.my_ops]
            friendly.extend(unit['hex'] for status in self.team_status.values() for unit in status['units'])
            if any(self.get_distance(pos, target) <= 1 for pos in friendly):
                return None
            return self.action(obj_id, ActionType.JMPlan, weapon_id=candidates[0]['weapon_id'], jm_pos=target)
        return None

    def step(self, observation):
        self.start_step(observation)
        if observation['time']['stage'] == 1:
            return [{'actor': self.seat, 'type': 333}]
        actions = []
        for obj_id in self.controllable_ops:
            action = (self.gen_shoot(obj_id) or self.gen_get_off(obj_id)
                      or self.gen_prepare(obj_id) or self.gen_jm_plan(obj_id) or self.gen_move(obj_id))
            if action:
                actions.append(action)
        return actions
