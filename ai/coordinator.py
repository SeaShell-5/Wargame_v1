"""Seat allocation and deterministic scheduling, without SDK object mutation."""
from copy import deepcopy

from .message_bus import MessageBus
from .assault_agent import AssaultAgent
from .occupy_agent import OccupyAgent
from .support_agent import SupportAgent


def int_keys(value):
    """SDK dictionaries use ints; JSON snapshots use strings."""
    return {int(k): v for k, v in value.items()}


class Coordinator:
    def __init__(self, squads_per_role=(1, 1, 1)):
        self.message_bus = MessageBus()
        self.squads_per_role = squads_per_role
        self.SEATS = {}
        self.red_agents, self.blue_agents = [], []
        self.unit_seats = {}
        for faction, agents in ((0, self.red_agents), (1, self.blue_agents)):
            base = 1 if faction == 0 else 101
            for cls, count in zip((AssaultAgent, OccupyAgent, SupportAgent), squads_per_role):
                for index in range(count):
                    agent = cls(self.message_bus)
                    agent.seat = base + len(agents)
                    agent.color = faction
                    agent.squad_index, agent.squad_count = index, count
                    agents.append(agent)
            self.SEATS[faction] = tuple(a.seat for a in agents)
        self.assault = self.red_agents[0]
        self.occupy = next(a for a in self.red_agents if a.AGENT_TYPE == 'occupy')
        self.support = next(a for a in self.red_agents if a.AGENT_TYPE == 'support')
        self.setup_complete = False

    def player_info(self):
        return [dict(seat=a.seat, faction=faction, role=int(index == 0),
                     user_name=f'{faction}_{a.AGENT_TYPE}_{a.squad_index + 1}', user_id=a.seat)
                for faction, agents in ((0, self.red_agents), (1, self.blue_agents))
                for index, a in enumerate(agents)]

    def setup_agents(self, setup_info, state=None):
        self.reset()
        for faction, agents in ((0, self.red_agents), (1, self.blue_agents)):
            roster = [{'seat': a.seat, 'kind': a.AGENT_TYPE} for a in agents]
            for index, agent in enumerate(agents):
                agent.setup({**setup_info, 'faction': faction, 'agent_roster': roster},
                            agent.seat, int(index == 0))
        self.setup_complete = True

    @staticmethod
    def unit_role(op):
        subtype = op['sub_type']
        return ('occupy' if subtype in (2, 4, 16) else
                'support' if subtype in (3, 5, 6, 7, 8, 10, 19) else 'assault')

    def grouping_actions(self, state):
        """Keep initial balanced assignments stable as units move or die."""
        actions = []
        for faction, agents in ((0, self.red_agents), (1, self.blue_agents)):
            seats = self.SEATS[faction]
            ob = state[faction]
            groups = int_keys(ob['role_and_grouping_info'])
            desired = {seat: set() for seat in seats}
            operators = {op['obj_id']: op for op in ob.get('operators', []) + ob.get('passengers', [])
                         if op['color'] == faction and op.get('blood', 1) > 0}
            for kind in ('assault', 'occupy', 'support'):
                peers = [a for a in agents if a.AGENT_TYPE == kind]
                units = sorted((op for op in operators.values() if self.unit_role(op) == kind),
                               key=lambda op: (op['cur_hex'] % 100, op['cur_hex'] // 100, op['obj_id']))
                new_units = [op for op in units if op['obj_id'] not in self.unit_seats]
                existing = [op for op in units if op['obj_id'] in self.unit_seats]
                if not existing:
                    for index, op in enumerate(new_units):
                        self.unit_seats[op['obj_id']] = peers[min(index * len(peers) // len(new_units), len(peers) - 1)].seat
                else:
                    loads = {a.seat: sum(self.unit_seats[op['obj_id']] == a.seat for op in existing) for a in peers}
                    for op in new_units:
                        seat = min(loads, key=loads.get)
                        self.unit_seats[op['obj_id']] = seat
                        loads[seat] += 1
            for obj_id in operators:
                desired[self.unit_seats[obj_id]].add(obj_id)
            alive = set(operators)
            if any(set(groups.get(seat, {}).get('operators', [])) & alive != desired[seat] for seat in seats):
                actions.append({'actor': seats[0], 'type': 100,
                                'info': {seat: {'operators': sorted(ids)} for seat, ids in desired.items()}})
        return actions

    def local_observation(self, observation, agent):
        """All owned pieces pool their knowledge; other seats need messages."""
        groups = int_keys(observation['role_and_grouping_info'])
        owned = set(groups.get(agent.seat, {}).get('operators', []))
        own = [op for op in observation.get('operators', [])
               if op['color'] == agent.color and op['obj_id'] in owned]
        passengers = [op for op in observation.get('passengers', [])
                      if op['color'] == agent.color and op['obj_id'] in owned]
        visible = {eid for op in own for eid in op.get('see_enemy_bop_ids', [])}
        enemies = [op for op in observation.get('operators', [])
                   if op['color'] != agent.color and op['obj_id'] in visible]
        # Explicit allowlist excludes global judge/action logs and other seats' units.
        local = {key: deepcopy(observation[key]) for key in
                 ('time', 'cities', 'scores', 'landmarks', 'terrain_id', 'scenario_id')
                 if key in observation}
        local.update(operators=deepcopy(own + enemies), passengers=deepcopy(passengers),
                     role_and_grouping_info={agent.seat: deepcopy(groups.get(agent.seat, {}))},
                     valid_actions={int(obj): int_keys(deepcopy(options))
                                    for obj, options in observation.get('valid_actions', {}).items()
                                    if int(obj) in owned})
        return local

    def step(self, observation, faction=0):
        if not self.setup_complete:
            raise RuntimeError('Coordinator.setup_agents must be called first')
        agents = self.red_agents if faction == 0 else self.blue_agents
        views = [self.local_observation(observation, agent) for agent in agents]
        # Observe -> publish -> consume/decide: same-tick delivery independent of order.
        for agent, view in zip(agents, views):
            agent.observe(view)
        for agent in agents:
            agent.share_information()
        actions = []
        for agent, view in zip(agents, views):
            for action in agent.step(view):
                if action['actor'] != agent.seat:
                    raise RuntimeError('Agent emitted another seat\'s action')
                if 'obj_id' in action and action['obj_id'] not in agent.controllable_ops:
                    raise RuntimeError('Agent attempted to control an unassigned piece')
                actions.append(action)
        return actions

    def reset(self):
        for agent in self.red_agents + self.blue_agents:
            agent.reset()
        self.message_bus.clear()
        self.unit_seats.clear()
        self.setup_complete = False
