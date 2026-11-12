"""Common seat-owned knowledge, message dispatch and legal action generation."""
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy

from .map import Map


class ActionType:
    Move, Shoot, GetOn, GetOff, occupy, ChangeState, RemoveKeep = range(1, 8)
    JMPlan, GuideShoot, StopMove, WeaponLock, WeaponUnFold, CancelJMPlan = range(8, 14)
    Fork, Union, ChangeAltitude, ActivateRadar, EnterFort, ExitFort, LayMine = range(14, 21)


class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class MoveType:
    Maneuver, March, Walk, Fly = range(4)


class BaseAgent(ABC):
    AGENT_TYPE = 'base'
    CONTROLLED_SUB_TYPES = []
    INTEL_TTL = 30

    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.seat = 0
        self.role = 0
        self.color = 0
        self.squad_index, self.squad_count = 0, 1
        self.agent_roster = []
        self.map = None
        self.map_data = None
        self.reset()

    def setup(self, setup_info, seat, role):
        self.reset()
        self.seat, self.role, self.color = seat, role, setup_info['faction']
        self.agent_roster = setup_info.get('agent_roster', [])
        self.map = Map(setup_info['basic_data'], setup_info['cost_data'], setup_info['see_data'])
        self.map_data = self.map.get_map_data()

    @property
    def address(self):
        return f'{self.color}:{self.seat}'

    def get_controllable_ops(self, observation):
        groups = observation.get('role_and_grouping_info', {})
        group = groups.get(self.seat, groups.get(str(self.seat), {}))
        owned = set(group.get('operators', []))
        return [op['obj_id'] for op in observation.get('operators', [])
                if op['color'] == self.color and op['obj_id'] in owned and op.get('blood', 1) > 0]

    def observe(self, observation):
        self.observation = observation
        self.current_step = observation['time']['cur_step']
        self.controllable_ops = self.get_controllable_ops(observation)
        self.my_ops = [op for op in observation.get('operators', [])
                       if op['obj_id'] in self.controllable_ops and op['color'] == self.color]
        self.occupied_cities = {city['coord'] for city in observation.get('cities', [])
                                if city['flag'] == self.color}
        self.enemy_positions = {eid: info for eid, info in self.enemy_positions.items()
                                if self.current_step - info['step'] <= self.INTEL_TTL}
        self.update_enemy_tracking(observation)
        self.team_status = {key: value for key, value in self.team_status.items()
                            if self.current_step - value['step'] <= self.INTEL_TTL}
        self.support_requests = {}
        self.capture_claims = set()
        self.fire_claims = set()
        self.unit_claims = set()

    def update_enemy_tracking(self, observation):
        seen = self.get_visible_enemies()
        for op in observation.get('operators', []):
            if op['color'] != self.color and op['obj_id'] in seen:
                self.enemy_positions[op['obj_id']] = {
                    'hex': op['cur_hex'], 'blood': op['blood'], 'sub_type': op['sub_type'],
                    'step': self.current_step}
                self.last_seen[op['obj_id']] = self.current_step

    def get_visible_enemies(self):
        return sorted({eid for op in self.my_ops for eid in op.get('see_enemy_bop_ids', [])})

    def get_bop(self, obj_id):
        return next((op for op in self.my_ops if op['obj_id'] == obj_id), None)

    def get_distance(self, start, end):
        if self.map is None or start is None or end is None:
            return float('inf')
        distance = self.map.get_distance(start, end)
        return distance if distance >= 0 else float('inf')

    def get_enemy_center(self):
        positions = [info['hex'] for info in self.enemy_positions.values()
                     if info.get('hex') is not None and info.get('blood', 1) > 0]
        if not positions:
            return None
        row = sum(p // 100 for p in positions) / len(positions)
        col = sum(p % 100 for p in positions) / len(positions)
        return min(positions, key=lambda p: (p // 100 - row) ** 2 + (p % 100 - col) ** 2)

    def objective_cities(self, observation):
        cities = sorted(observation.get('cities', []), key=lambda c: c['coord'] % 100)
        local = [city for index, city in enumerate(cities)
                 if min(index * self.squad_count // max(len(cities), 1), self.squad_count - 1) == self.squad_index]
        return local or cities

    def staging_target(self, op, objective):
        """Spread reserves behind the objective instead of piling onto its hex."""
        if self.squad_count == 1:
            return objective
        mode = self.get_move_type(op)
        row, col = divmod(objective, 100)
        candidates = []
        direction = -1 if self.color == 0 else 1
        for dr in (2, 3, 4, 5):
            rr = row + direction * dr
            for dc in range(-4, 5):
                cc = col + dc
                pos = rr * 100 + cc
                if self.map.is_valid(pos) and self.map.cost[mode][rr][cc] and self.map.basic[rr][cc]['cond'] != 4:
                    candidates.append(pos)
        return candidates[op['obj_id'] % len(candidates)] if candidates else objective

    def send_message(self, receiver, content):
        if ':' in str(receiver):
            addresses = [receiver] if str(receiver).startswith(f'{self.color}:') else []
        else:
            addresses = [f"{self.color}:{peer['seat']}" for peer in self.agent_roster
                         if peer['kind'] == receiver and peer['seat'] != self.seat]
        for address in addresses:
            self.message_bus.send(self.address, address, {**content, 'step': self.current_step})

    def receive_messages(self):
        return self.message_bus.receive(self.address)

    def share_information(self):
        status = {'type': 'unit_status', 'seat': self.seat,
                  'units': [{'obj_id': op['obj_id'], 'hex': op['cur_hex'], 'blood': op['blood']}
                            for op in self.my_ops]}
        enemies = {eid: info for eid, info in self.enemy_positions.items()
                   if info['step'] == self.current_step and eid in self.get_visible_enemies()}
        targets = [city['coord'] for city in self.objective_cities(self.observation)
                   if city['flag'] != self.color]
        for peer in self.agent_roster:
            if peer['seat'] == self.seat:
                continue
            receiver = f"{self.color}:{peer['seat']}"
            if self.current_step % 10 == 0:
                self.send_message(receiver, status)
            if enemies and self.current_step % 5 == 0:
                self.send_message(receiver, {'type': 'enemy_spotted', 'enemies': enemies})
            if self.AGENT_TYPE == 'occupy' and self.my_ops and targets:
                target = min(targets, key=lambda p: min(self.get_distance(op['cur_hex'], p) for op in self.my_ops))
                self.send_message(receiver, {'type': 'support_request', 'target': target})

    def process_messages(self, messages):
        """Explicit branches update information used by the role strategies."""
        for msg in messages:
            if not msg['sender'].startswith(f'{self.color}:'):
                continue
            content = msg['content']
            if not 0 <= self.current_step - content.get('step', -1000) <= self.INTEL_TTL:
                continue
            kind = content.get('type')
            self.received_counts[kind] += 1
            if kind == 'enemy_spotted':
                for eid, info in content.get('enemies', {}).items():
                    eid = int(eid)
                    if info['step'] >= self.enemy_positions.get(eid, {}).get('step', -1):
                        self.enemy_positions[eid] = deepcopy(info)
                        self.last_seen[eid] = info['step']
            elif kind == 'unit_status':
                self.team_status[msg['sender']] = deepcopy(content)
            elif kind == 'support_request':
                self.support_requests[msg['sender']] = content['target']
            elif kind == 'point_captured':
                self.occupied_cities.add(content['target'])
            elif kind == 'action_intent' and content['step'] == self.current_step:
                self.unit_claims.update(content.get('units', []))
                if content['action'] == ActionType.occupy:
                    self.capture_claims.add(content['target'])
                elif content['action'] in (ActionType.Shoot, ActionType.GuideShoot):
                    self.fire_claims.add(content['target'])

    def announce_intent(self, kind, target, units=()):
        self.unit_claims.update(units)
        if kind == ActionType.occupy:
            self.capture_claims.add(target)
        elif kind in (ActionType.Shoot, ActionType.GuideShoot):
            self.fire_claims.add(target)
        for peer in self.agent_roster:
            if peer['seat'] != self.seat:
                self.send_message(f"{self.color}:{peer['seat']}",
                                  {'type': 'action_intent', 'action': kind, 'target': target,
                                   'units': list(units)})

    def candidates(self, obj_id):
        return self.observation.get('valid_actions', {}).get(obj_id, {})

    def action(self, obj_id, kind, **kwargs):
        if obj_id not in self.controllable_ops or kind not in self.candidates(obj_id):
            return None
        if kind != ActionType.occupy and obj_id in self.unit_claims:
            return None
        if kind not in (ActionType.occupy, ActionType.Shoot, ActionType.GuideShoot):
            self.announce_intent(kind, None, units=[obj_id])
        return dict(actor=self.seat, obj_id=obj_id, type=kind, **kwargs)

    def gen_shoot(self, obj_id):
        options = self.candidates(obj_id)
        for kind in (ActionType.Shoot, ActionType.GuideShoot):
            candidates = [c for c in options.get(kind) or []
                          if c['target_obj_id'] not in self.fire_claims
                          and c.get('guided_obj_id') not in self.unit_claims]
            if candidates:
                best = max(candidates, key=lambda c: c['attack_level'])
                fields = {k: best[k] for k in ('target_obj_id', 'weapon_id')}
                if kind == ActionType.GuideShoot:
                    fields['guided_obj_id'] = best['guided_obj_id']
                action = self.action(obj_id, kind, **fields)
                if action:
                    units = [obj_id] + ([best['guided_obj_id']] if kind == ActionType.GuideShoot else [])
                    self.announce_intent(kind, best['target_obj_id'], units=units)
                return action
        return None

    def get_move_type(self, bop):
        if bop['type'] == BopType.Vehicle:
            return MoveType.March if bop.get('move_state') == 1 else MoveType.Maneuver
        return MoveType.Walk if bop['type'] == BopType.Infantry else MoveType.Fly

    def gen_move_route(self, start, end, move_type):
        if not self.map:
            return []
        landmarks = self.observation.get('landmarks', {})
        blocked = {item['hex'] if isinstance(item, dict) else item
                   for item in landmarks.get('roadblocks', [])} if move_type != MoveType.Fly else set()
        return self.map.gen_move_route(start, end, move_type, blocked=blocked)

    def gen_move(self, obj_id):
        bop = self.get_bop(obj_id)
        target = self.target_assignments.get(obj_id, {}).get('target')
        if bop is None or target is None or target == bop['cur_hex'] or bop.get('sub_type') == 3:
            return None
        if bop.get('speed', 0) > 0 or bop.get('move_path') or bop.get('move_to_stop_remain_time', 0) > 0:
            return None
        options = self.candidates(obj_id)
        if ActionType.Move not in options:
            return self.action(obj_id, ActionType.WeaponLock)
        route = self.gen_move_route(bop['cur_hex'], target, self.get_move_type(bop))
        return self.action(obj_id, ActionType.Move, move_path=route) if route else None

    def gen_get_off(self, obj_id):
        candidates = self.candidates(obj_id).get(ActionType.GetOff) or []
        if candidates:
            return self.action(obj_id, ActionType.GetOff, target_obj_id=candidates[0]['target_obj_id'])
        return None

    def gen_occupy(self, obj_id):
        bop = self.get_bop(obj_id)
        if bop is None or bop['cur_hex'] in self.capture_claims:
            return None
        action = self.action(obj_id, ActionType.occupy)
        if action:
            self.announce_intent(ActionType.occupy, bop['cur_hex'])
        return action

    def gen_prepare(self, obj_id):
        # Suppression expires automatically under rule 8.8. Do not spend a
        # unit's blood on the SDK's optional immediate-removal action.
        if self.get_visible_enemies():
            return self.action(obj_id, ActionType.WeaponUnFold)
        return None

    def start_step(self, observation):
        # Coordinator has already observed/published all seats. Also supports standalone use.
        if observation is not self.observation:
            self.observe(observation)
        self.process_messages(self.receive_messages())
        self.target_assignments = {}
        self.assign_targets(observation)

    def assign_targets(self, observation):
        pass

    @abstractmethod
    def step(self, observation):
        pass

    def reset(self):
        self.observation = None
        self.current_step = 0
        self.controllable_ops = []
        self.my_ops = []
        self.occupied_cities = set()
        self.enemy_positions = {}
        self.last_seen = {}
        self.team_status = {}
        self.support_requests = {}
        self.capture_claims = set()
        self.fire_claims = set()
        self.unit_claims = set()
        self.target_assignments = {}
        self.received_counts = Counter()
