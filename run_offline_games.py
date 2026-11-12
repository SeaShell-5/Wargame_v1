"""Run seat-owned multi-agent games in the wargame conda environment."""
import argparse
from collections import Counter, deque
from copy import deepcopy
import math
import json
from pathlib import Path
import pickle
import random
import time
import zipfile
import numpy
from train_env import TrainEnv
from ai import Coordinator

RED, BLUE, GREEN = 0, 1, -1
ROOT = Path(__file__).resolve().parent
LARGE_BATTLE_MIN_UNITS = 50


class SDKAdapter:
    """Read fresh SDK 4.1.0 observations without changing engine state."""

    def __init__(self, env):
        self.env = env
        self.refresh_count = 0

    def _fresh(self, returned):
        getter = getattr(getattr(self.env, 'engine', None), 'get_sd_data', None)
        if getter is None:
            return returned
        self.refresh_count += 1
        fresh = getter()
        # The runner streams its own replay. Bound redundant SDK replay copies
        # in long 500-piece games without changing simulation state.
        history = getattr(self.env, 'all_dic_state_data', None)
        if not getattr(self.env, 'save_replay_flag', True) and isinstance(history, list) and len(history) > 2:
            del history[:-1]
        return fresh

    def setup(self, info):
        return self._fresh(self.env.setup(info))

    def step(self, actions):
        returned, done = self.env.step(actions)
        return self._fresh(returned), done

    def reset(self):
        self.env.reset()


def to_json_string(value, indent=None):
    return json.dumps(value, ensure_ascii=False, indent=indent, separators=(',', ':'))


def normalize_map_data(basic):
    """Accept maps using abbreviated cell fields without changing their size."""
    aliases = {'elev': 'e', 'cond': 'cd', 'roads': 'rd',
               'rivers': 'rv', 'neighbors': 'nb'}
    basic = deepcopy(basic)
    for row, cells in enumerate(basic['map_data']):
        for col, cell in enumerate(cells):
            cell.setdefault('pos', row * 100 + col)
            for name, alias in aliases.items():
                if name not in cell and alias in cell:
                    cell[name] = cell[alias]
            # Compact maps omit fields whose values are zero.
            cell.setdefault('cond', 0)
            cell.setdefault('elev', 0)
            cell.setdefault('roads', [0] * 6)
            cell.setdefault('rivers', [0] * 6)
    return basic


def load_setup(scenario='1930331196', map_id='96', units=500, coordinator=None, seed=0):
    map_dir = ROOT / 'data/maps' / f'map_{map_id}'
    with (ROOT / 'data/scenarios' / f'{scenario}.json').open(encoding='utf8') as f:
        scenario_data = json.load(f)
    with (map_dir / 'basic.json').open(encoding='utf8') as f:
        basic_data = normalize_map_data(json.load(f))
    with (map_dir / 'cost.pickle').open('rb') as f:
        cost_data = pickle.load(f)
    with numpy.load(map_dir / f'{map_id}see.npz') as data:
        see_data = data['data']
    coordinator = coordinator or Coordinator((4, 4, 2) if units else (1, 1, 1))
    if units:
        scenario_data = build_battle(scenario_data, basic_data, cost_data, units, seed)
    return dict(scenario_data=scenario_data, basic_data=basic_data,
                cost_data=cost_data, see_data=see_data,
                player_info=coordinator.player_info())


def build_battle(source, basic, costs, units=500, seed=0):
    """Build mirrored armies, assigning an odd extra piece to red."""
    if units < 1:
        raise ValueError('Battle generation requires a positive total of pieces')
    grid = basic['map_data']
    height, width = len(grid), len(grid[0])
    # Small battles retain the source map dimensions; large battles keep the
    # existing minimum-size requirement.
    if units >= LARGE_BATTLE_MIN_UNITS and (height < 60 or width < 50):
        raise ValueError('Large battles require a map of at least 60 x 50 hexes')
    scenario = deepcopy(source)
    scenario.pop('blueprints', None)
    scenario['landmarks'] = {'roadblocks': [], 'minefields': []}
    templates = {op['sub_type']: op for op in reversed(source['operators']) if op['color'] == 0}
    weights = {0: 60, 1: 60, 2: 60, 4: 40, 3: 12, 5: 10, 6: 8}
    missing = set(weights) - templates.keys()
    if missing:
        raise ValueError(f'Source scenario lacks unit templates: {sorted(missing)}')
    half = units // 2
    counts = {kind: half * weight // 250 for kind, weight in weights.items()}
    for kind in sorted(weights, key=lambda k: -(half * weights[k] % 250))[:half - sum(counts.values())]:
        counts[kind] += 1
    # Eight separated neutral objectives, invariant as a set under 180-degree rotation.
    columns = [round((width - 1) * fraction) for fraction in (0.13, 0.34, 0.55, 0.76)]
    cities = [(height // 2 - 1) * 100 + col for col in columns]
    cities += [(height - 1 - pos // 100) * 100 + width - 1 - pos % 100 for pos in cities[:]]
    for pos in cities:
        if not costs[0][pos // 100][pos % 100] or not costs[2][pos // 100][pos % 100]:
            raise ValueError(f'Objective {pos} is not traversable; choose another map')
    scenario['cities'] = [dict(coord=pos, value=80, flag=-1, name=f'Neutral objective {index + 1}')
                          for index, pos in enumerate(sorted(cities))]
    reachable = {}
    for mode in (0, 2):
        seen, queue = set(cities), deque(cities)
        while queue:
            pos = queue.popleft()
            for neighbour, cost in costs[mode][pos // 100][pos % 100].items():
                if math.isfinite(cost) and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        reachable[mode] = seen
    rng = random.Random(seed)
    reserved = set()
    operators = []
    next_ids = {0: 0, 1: 10000}
    # Vehicles in front, infantry behind them, reconnaissance/artillery in the rear.
    bands = {0: (0.27, 0.32), 1: (0.27, 0.32), 2: (0.22, 0.28), 4: (0.22, 0.28),
             3: (0.15, 0.20), 5: (0.18, 0.23), 6: (0.18, 0.23)}
    for kind, count in counts.items():
        # Keep the paired army composition; an odd extra piece is a red type 0.
        paired_count = count
        count += int(kind == 0 and units % 2)
        if not count:
            continue
        mode = 2 if kind == 2 else 3 if kind in (5, 6) else 0
        low, high = bands[kind]
        candidates = []
        for row in range(int(height * low), int(height * high) + 1):
            for col in range(int(width * 0.10), int(width * 0.90)):
                pos = row * 100 + col
                opposite = (height - 1 - row) * 100 + width - 1 - col
                rr, cc = divmod(opposite, 100)
                if pos in reserved or grid[row][col]['cond'] != grid[rr][cc]['cond']:
                    continue
                if not costs[mode][row][col] or not costs[mode][rr][cc]:
                    continue
                if mode in reachable and (pos not in reachable[mode] or opposite not in reachable[mode]):
                    continue
                candidates.append(pos)
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f'Not enough mirrored passable deployment cells for type {kind}: {len(candidates)} < {count}')
        for index, pos in enumerate(candidates[:count]):
            reserved.add(pos)
            for faction, location in ((0, pos), (1, (height - 1 - pos // 100) * 100 + width - 1 - pos % 100)):
                if faction == 1 and index >= paired_count:
                    continue
                op = deepcopy(templates[kind])
                op.update(obj_id=next_ids[faction], color=faction,
                          cur_hex=location, cur_pos=0.0, speed=0, move_path=[],
                          on_board=0, car=None, launcher=None, passenger_ids=[], launch_ids=[],
                          get_on_partner_id=[], get_off_partner_id=[], see_enemy_bop_ids=[])
                operators.append(op)
                next_ids[faction] += 1
    # Dismounted infantry/UGVs retain their required carrier relationship.
    for faction in (0, 1):
        carriers = [op for op in operators if op['color'] == faction and op['sub_type'] == 1]
        for kind in (2, 4):
            children = [op for op in operators if op['color'] == faction and op['sub_type'] == kind]
            for index, child in enumerate(children):
                carrier = carriers[index % len(carriers)]
                child['launcher'] = carrier['obj_id']
                carrier['launch_ids'].append(child['obj_id'])
    scenario['operators'] = operators
    scenario['time'] = dict(cur_step=0, tick=1.0, max_time=1800)
    return scenario


def run_in_multi_agents_mode(scenario='1930331196', map_id='96', max_steps=None, seed=0, units=500):
    random.seed(seed)
    begin  = time.time()
    run_id = time.strftime('%m%d%H%M%S', time.localtime(begin))
    output = ROOT / 'logs' / 'runs' / run_id
    output.mkdir(parents=True)
    replay_path = ROOT / 'logs' / 'replays' / f'{scenario}_{run_id}.zip'
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    """    
        Coordinator(squads_per_role)表示配置双方的智能体数量，双方数量相同，
        (4, 4, 2)表示创建 10 个智能体，智能体角色分别为突击、占领、支援
    """
    coordinator = Coordinator((4, 4, 2) if units else (1, 1, 1))
    setup = load_setup(scenario, map_id, units, coordinator, seed)
    env = SDKAdapter(TrainEnv())
    errors, action_counts, sent_counts = Counter(), Counter(), Counter()
    recorded_results = set()
    summary = {'scenario': scenario, 'map': map_id, 'seed': seed, 'run_id': run_id, 'requested_pieces': units,
               'agents_per_faction': len(coordinator.red_agents)}
    steps, frame = 0, 0
    done = False
    try:
        state = env.setup(setup)
        coordinator.setup_agents(setup)
        summary['initial_piece_counts'] = dict(Counter(op['color'] for op in state[GREEN]['operators'] + state[GREEN].get('passengers', [])))
        if units and sum(summary['initial_piece_counts'].values()) != units:
            raise RuntimeError('SDK initial piece count does not match the requested deployment')
        (output / 'initial_deployment.json').write_text(to_json_string(setup['scenario_data'], 2), encoding='utf8')
        with zipfile.ZipFile(replay_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as replay, \
                (output / 'decisions.jsonl').open('w', encoding='utf8') as audit, \
                (output / 'messages.jsonl').open('w', encoding='utf8') as message_log:
            message_count = 0
            def record():
                nonlocal frame
                replay.writestr(f'{run_id}/{frame}', to_json_string(state[GREEN]))
                frame += 1
                for result in state[GREEN].get('actions', []):
                    result_key = to_json_string(result)
                    if result_key in recorded_results:
                        continue
                    recorded_results.add(result_key)
                    message = result['message']
                    action_counts[str(message['actor'])] += 1
                    if result.get('error'):
                        error = result['error']
                        errors[f"{error.get('code')}: {error.get('message')}"] += 1
            record()
            # Real SDK control allocation, through captain action 100.
            allocation = coordinator.grouping_actions(state)
            if allocation:
                state, done = env.step(allocation)
                record()
            summary['initial_groups'] = state[GREEN]['role_and_grouping_info']
            # Do not advance deployment by modifying the clock.
            deployment = [{'actor': seat, 'type': 333}
                          for seats in coordinator.SEATS.values() for seat in seats]
            state, done = env.step(deployment)
            record()
            if state[GREEN]['time']['stage'] != 2:
                raise RuntimeError('SDK did not finish deployment through EndDeploy actions')
            print('SDK allocation:', to_json_string(summary['initial_groups']), flush=True)
            while not done and (max_steps is None or steps < max_steps):
                allocation = coordinator.grouping_actions(state)
                if allocation:
                    state, done = env.step(allocation)
                    record()
                    if done:
                        break
                actions = coordinator.step(state[RED], RED) + coordinator.step(state[BLUE], BLUE)
                history = coordinator.message_bus.message_history
                message_count += len(history)
                for message in history:
                    message_log.write(to_json_string(message) + '\n')
                history.clear()
                seats = []
                for agent in coordinator.red_agents + coordinator.blue_agents:
                    seats.append({'seat': agent.seat, 'owned': agent.controllable_ops,
                                  'own_visible': agent.get_visible_enemies(),
                                  'known_enemies': agent.enemy_positions,
                                  'received': dict(agent.received_counts),
                                  'targets': agent.target_assignments})
                for action in actions:
                    sent_counts[str(action['actor'])] += 1
                audit.write(to_json_string({'step': state[GREEN]['time']['cur_step'],
                                            'seats': seats, 'actions': actions}) + '\n')
                state, done = env.step(actions)
                record()
                steps += 1
                if steps % 300 == 0:
                    print(f'Step {steps}: errors={sum(errors.values())}', flush=True)
            summary.update(steps=steps, done=done, truncated=not done,
                           final_time=state[GREEN]['time'], scores=state[GREEN].get('scores'),
                           actions_by_seat=dict(action_counts), sent_by_seat=dict(sent_counts),
                           action_errors=dict(errors), messages=message_count,
                           received_by_seat={str(a.seat): dict(a.received_counts)
                                             for a in coordinator.red_agents + coordinator.blue_agents},
                           final_groups=state[GREEN]['role_and_grouping_info'],
                           sdk_observation_refreshes=env.refresh_count,
                           elapsed_seconds=round(time.time() - begin, 3), replay=str(replay_path))
        (output / 'summary.json').write_text(to_json_string(summary, 2), encoding='utf8')
        # print(to_json_string(summary, 2), flush=True)
        print(f'Audit: {output}', flush=True)
        return summary
    finally:
        env.reset()
        coordinator.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='1930331196')
    parser.add_argument('--map', dest='map_id', default='96')
    parser.add_argument('--units', type=int, default=500,
                        help='Total pieces across both factions; 0 keeps the source scenario')
    parser.add_argument('--max-steps', type=int, default=3000,
                        help='Optional smoke-test cutoff; default runs a full game')
    parser.add_argument('--seed', type=int, default=0,
                        help='AI pathfinding seed; SDK combat randomness is separate')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    run_in_multi_agents_mode(args.scenario, args.map_id, args.max_steps, args.seed, args.units)


if __name__ == '__main__':
    main()
