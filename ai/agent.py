"""Centralized Commander Agent with intelligent group coordination.

Features:
1. Group operators by type (assault, occupy, support)
2. Assign different missions to each group
3. Coordinate via communication
4. Dynamic target assignment and adjustment
"""

import json
import os
import random
from collections import defaultdict

from .base_agent import BaseAgent
from .map import Map


class Agent(BaseAgent):
    def __init__(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.map_data = None
        self.seat = None
        self.faction = None
        self.role = None
        self.controllable_ops = None
        self.team_info = None
        self.user_name = None
        self.user_id = None

        self.enemy_positions = {}
        self.occupied_cities = set()
        self.group_assignments = {}
        self.last_seen_enemies = defaultdict(int)
        self.current_step = 0

        self.groups = {
            "assault": [],
            "occupy": [],
            "support": [],
        }

    def setup(self, setup_info):
        self.scenario = setup_info["scenario"]
        self.color = setup_info["faction"]
        self.faction = setup_info["faction"]
        self.seat = setup_info["seat"]
        self.role = setup_info["role"]
        self.user_name = setup_info["user_name"]
        self.user_id = setup_info["user_id"]
        self.observation = None
        self.map = Map(
            setup_info["basic_data"], setup_info["cost_data"], setup_info["see_data"]
        )
        self.map_data = self.map.get_map_data()

        self.priority = {
            ActionType.occupy: self.gen_occupy,
            ActionType.Shoot: self.gen_shoot,
            ActionType.GuideShoot: self.gen_guide_shoot,
            ActionType.Move: self.gen_move,
            ActionType.StopMove: self.gen_stop_move,
            ActionType.GetOn: self.gen_get_on,
            ActionType.GetOff: self.gen_get_off,
            ActionType.WeaponLock: self.gen_WeaponLock,
            ActionType.WeaponUnFold: self.gen_WeaponUnFold,
            ActionType.ChangeState: self.gen_change_state,
            ActionType.JMPlan: self.gen_jm_plan,
        }

    def reset(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.map_data = None
        self.enemy_positions = {}
        self.occupied_cities = set()
        self.group_assignments = {}
        self.last_seen_enemies = defaultdict(int)
        self.current_step = 0
        self.groups = {"assault": [], "occupy": [], "support": []}

    def step(self, observation: dict):
        self.observation = observation
        self.current_step = observation.get("time", {}).get("cur_step", 0)
        self.team_info = observation["role_and_grouping_info"]
        self.controllable_ops = observation["role_and_grouping_info"][self.seat][
            "operators"
        ]

        self.update_enemy_tracking(observation)
        self.process_incoming_communication(observation)

        total_actions = []

        if observation["time"]["stage"] == 1:
            return [{"actor": self.seat, "type": 333}]

        self.assign_groups(observation)
        self.assign_group_targets(observation)

        for obj_id in self.controllable_ops:
            action = self.gen_occupy(obj_id)
            if action:
                total_actions.append(action)
                continue

            action = self.gen_shoot(obj_id)
            if action:
                total_actions.append(action)
                continue

        for obj_id, valid_actions in observation["valid_actions"].items():
            if obj_id not in self.controllable_ops:
                continue
            skip_types = {ActionType.occupy, ActionType.Shoot}
            for action_type in self.priority:
                if action_type in skip_types:
                    continue
                if action_type not in valid_actions:
                    continue
                gen_action = self.priority[action_type]
                action = gen_action(obj_id, valid_actions[action_type])
                if action:
                    total_actions.append(action)
                    break

        comm_actions = self.gen_coordination_message()
        total_actions.extend(comm_actions)

        return total_actions

    def assign_groups(self, observation):
        self.groups = {"assault": [], "occupy": [], "support": []}

        for obj_id in self.controllable_ops:
            bop = self.get_bop(obj_id)
            if bop is None:
                continue

            sub_type = bop.get("sub_type", 0)
            bop_type = bop.get("type", 0)

            if sub_type in [2, 4]:
                self.groups["occupy"].append(obj_id)
            elif sub_type == 7:
                self.groups["support"].append(obj_id)
            elif sub_type in [0, 1, 9, 10, 17, 18]:
                self.groups["assault"].append(obj_id)
            else:
                self.groups["assault"].append(obj_id)

    def assign_group_targets(self, observation):
        cities = observation.get("cities", [])
        if not cities:
            return

        for city in cities:
            if city.get("flag") == self.color:
                self.occupied_cities.add(city["coord"])

        unoccupied = [c for c in cities if c["coord"] not in self.occupied_cities]
        if not unoccupied:
            unoccupied = cities

        primary_city = unoccupied[0]["coord"]
        secondary_city = unoccupied[1]["coord"] if len(unoccupied) > 1 else primary_city

        enemy_center = self.get_enemy_center()

        for obj_id in self.controllable_ops:
            if obj_id in self.groups["occupy"]:
                self.group_assignments[obj_id] = {
                    "target": primary_city,
                    "mission": "occupy",
                    "mode": "aggressive" if self.current_step < 300 else "defensive",
                }
            elif obj_id in self.groups["support"]:
                self.group_assignments[obj_id] = {
                    "target": enemy_center if enemy_center else primary_city,
                    "mission": "support",
                    "mode": "fire_support",
                }
            else:
                if unoccupied:
                    self.group_assignments[obj_id] = {
                        "target": primary_city,
                        "mission": "assault",
                        "mode": "aggressive",
                    }
                else:
                    self.group_assignments[obj_id] = {
                        "target": secondary_city,
                        "mission": "assault",
                        "mode": "aggressive",
                    }

    def get_enemy_center(self):
        if not self.enemy_positions:
            return None

        total_hex = 0
        count = 0
        for enemy_id, info in self.enemy_positions.items():
            if info.get("hex"):
                total_hex += info["hex"]
                count += 1

        if count > 0:
            return total_hex // count
        return None

    def update_enemy_tracking(self, observation):
        for op in observation.get("operators", []):
            if op.get("color") == self.color:
                seen_ids = op.get("see_enemy_bop_ids", [])
                for enemy_id in seen_ids:
                    if enemy_id not in self.enemy_positions:
                        self.enemy_positions[enemy_id] = {
                            "hex": None,
                            "blood": None,
                            "sub_type": None,
                            "first_seen": self.current_step,
                        }
                    self.last_seen_enemies[enemy_id] += 1

    def process_incoming_communication(self, observation):
        communications = observation.get("communication", [])
        for comm in communications:
            if comm.get("type") == 209:
                hex_pos = comm.get("hex")
                if hex_pos:
                    enemy_center = hex_pos
                    for eid in list(self.enemy_positions.keys())[:5]:
                        if self.enemy_positions[eid].get("hex") is None:
                            self.enemy_positions[eid]["hex"] = enemy_center

    def get_bop(self, obj_id):
        for bop in self.observation["operators"]:
            if obj_id == bop["obj_id"]:
                return bop
        return None

    def gen_occupy(self, obj_id):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None

        assignment = self.group_assignments.get(obj_id, {})
        if assignment.get("mission") != "occupy":
            return None

        target = assignment.get("target")
        if not target:
            return None

        dist = self.map.get_distance(bop["cur_hex"], target)
        if dist <= 5:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.occupy,
            }
        return None

    def gen_shoot(self, obj_id):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None

        valid_actions = self.observation.get("valid_actions", {}).get(obj_id, {})
        if ActionType.Shoot in valid_actions and valid_actions[ActionType.Shoot]:
            candidates = valid_actions[ActionType.Shoot]
            best = max(candidates, key=lambda x: x["attack_level"])
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Shoot,
                "target_obj_id": best["target_obj_id"],
                "weapon_id": best["weapon_id"],
            }

        visible_enemies = bop.get("see_enemy_bop_ids", [])
        if not visible_enemies:
            return None

        weapon_id = bop.get("carry_weapon_ids", [None])[0]
        if not weapon_id:
            return None

        target = self.prioritize_target(bop, visible_enemies)
        if target:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Shoot,
                "target_obj_id": target,
                "weapon_id": weapon_id,
            }
        return None

    def prioritize_target(self, bop, visible_enemies):
        targets_with_priority = []

        for enemy_id in visible_enemies:
            info = self.enemy_positions.get(enemy_id, {})
            blood = info.get("blood") or 5
            sub_type = info.get("sub_type") or 1

            priority = 0
            if blood <= 1:
                priority += 100
            elif blood <= 2:
                priority += 50

            if sub_type == 0:
                priority += 20
            elif sub_type == 3:
                priority += 30

            priority += self.last_seen_enemies.get(enemy_id, 0) * 0.1

            dist = 0
            if info.get("hex"):
                dist = self.map.get_distance(bop["cur_hex"], info["hex"])
            priority -= dist * 2

            targets_with_priority.append((enemy_id, priority))

        if targets_with_priority:
            targets_with_priority.sort(key=lambda x: x[1], reverse=True)
            return targets_with_priority[0][0]
        return visible_enemies[0] if visible_enemies else None

    def gen_move(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None

        assignment = self.group_assignments.get(obj_id, {})
        target = assignment.get("target")

        if not target:
            cities = self.observation.get("cities", [])
            if cities:
                target = cities[0]["coord"]
            else:
                return None

        dist = self.map.get_distance(bop["cur_hex"], target)

        visible_enemies = bop.get("see_enemy_bop_ids", [])
        assignment_mode = assignment.get("mode", "aggressive")

        if visible_enemies and assignment_mode == "aggressive":
            target = self.prioritize_enemy_position(bop)
            if target:
                dist = self.map.get_distance(bop["cur_hex"], target)

        if dist <= 3:
            return None

        move_type = self.get_move_type(bop)
        route = self.map.gen_move_route(bop["cur_hex"], target, move_type)
        if route:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Move,
                "move_path": route,
            }
        return None

    def prioritize_enemy_position(self, bop):
        visible = bop.get("see_enemy_bop_ids", [])
        if not visible:
            return None

        best_pos = None
        best_count = 0
        for enemy_id in visible:
            count = self.last_seen_enemies.get(enemy_id, 0)
            if count > best_count:
                best_count = count
                info = self.enemy_positions.get(enemy_id, {})
                if info.get("hex"):
                    best_pos = info["hex"]

        return best_pos

    def get_move_type(self, bop):
        bop_type = bop.get("type", 0)
        move_state = bop.get("move_state", 0)

        if bop_type == BopType.Vehicle:
            if move_state == 1:
                return MoveType.March
            return MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            return MoveType.Walk
        else:
            return MoveType.Fly

    def gen_stop_move(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None

        assignment = self.group_assignments.get(obj_id, {})
        target = assignment.get("target")

        if target:
            dist = self.map.get_distance(bop["cur_hex"], target)
            if dist <= 5:
                return {
                    "actor": self.seat,
                    "obj_id": obj_id,
                    "type": ActionType.StopMove,
                }
        return None

    def gen_get_on(self, obj_id, candidate):
        if not candidate:
            return None
        bop = self.get_bop(obj_id)
        if bop is None or bop.get("sub_type", 0) not in [2, 4]:
            return None
        target_obj_id = random.choice(candidate)["target_obj_id"]
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GetOn,
            "target_obj_id": target_obj_id,
        }

    def gen_get_off(self, obj_id, candidate):
        if not candidate:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GetOff,
            "target_obj_id": candidate[0]["target_obj_id"],
        }

    def gen_WeaponLock(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None
        visible = bop.get("see_enemy_bop_ids", [])
        if visible:
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.WeaponLock}
        return None

    def gen_WeaponUnFold(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        if bop is None:
            return None
        visible = bop.get("see_enemy_bop_ids", [])
        if visible:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.WeaponUnFold,
            }
        return None

    def gen_change_state(self, obj_id, candidate):
        if not candidate:
            return None
        bop = self.get_bop(obj_id)
        if bop is None:
            return None

        visible = bop.get("see_enemy_bop_ids", [])
        target_state = 1 if visible else 0

        for c in candidate:
            if c.get("target_state") == target_state:
                return {
                    "actor": self.seat,
                    "obj_id": obj_id,
                    "type": ActionType.ChangeState,
                    "target_state": target_state,
                }
        return None

    def gen_guide_shoot(self, obj_id, candidate):
        if not candidate:
            return None
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GuideShoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
            "guided_obj_id": best["guided_obj_id"],
        }

    def gen_jm_plan(self, obj_id, candidate):
        if not candidate:
            return None
        weapon_id = random.choice(candidate)["weapon_id"]
        cities = self.observation.get("cities", [])
        if not cities:
            return None

        enemy_pos = self.get_enemy_center()
        target = enemy_pos if enemy_pos else cities[0]["coord"]

        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.JMPlan,
            "jm_pos": target,
            "weapon_id": weapon_id,
        }

    def gen_coordination_message(self):
        if self.role != 1:
            return []

        if not hasattr(self, "last_comm_step"):
            self.last_comm_step = -100

        if self.current_step - self.last_comm_step < 20:
            return []

        self.last_comm_step = self.current_step

        if self.current_step < 100:
            return []

        msg = {
            "actor": self.seat,
            "type": 209,
            "seat": self.seat,
            "hex": 0,
            "radius": 10,
            "start_time": self.current_step,
            "end_time": self.current_step + 50,
        }
        return [msg]


class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class ActionType:
    Move = 1
    Shoot = 2
    GetOn = 3
    GetOff = 4
    occupy = 5
    ChangeState = 6
    RemoveKeep = 7
    JMPlan = 8
    GuideShoot = 9
    StopMove = 10
    WeaponLock = 11
    WeaponUnFold = 12
    CancelJMPlan = 13
    Fork = 14
    Union = 15
    ChangeAltitude = 16
    ActivateRadar = 17
    EnterFort = 18
    ExitFort = 19
    LayMine = 20


class MoveType:
    Maneuver, March, Walk, Fly = range(4)
