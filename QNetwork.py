import time
import logging
import random
import json
import numpy as np
import socketio
from flask import Flask  # เผื่อใช้งาน (อาจไม่จำเป็น)
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


# =========================================
# (A) EXTRACT / PARSE FEATURE STATE
# =========================================
def extract_features(features):
    """
    ดึงข้อมูลคุณลักษณะที่สำคัญจาก state (features) ในรูปแบบของ nested dict
    ให้ได้เวกเตอร์ตัวเลขที่มีขนาดคงที่ (ในที่นี้ 36 ค่า) โดยใช้ค่า default หากไม่มีข้อมูล
    """
    # ตรวจสอบหาก features เป็น string ให้แปลงเป็น dict
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing features JSON: {e}")
            features = {}

    # Basic Status: health, hunger, armor, stamina, is_alive, experience, level, morale, fatigue, cooldown_time
    bs = features.get("basic_status", {})
    basic = [
        float(bs.get("health", 100)),
        float(bs.get("hunger", 100)),
        float(bs.get("armor", 0)),
        float(bs.get("stamina", 100)),
        1.0 if bs.get("is_alive", True) else 0.0,
        float(bs.get("experience", 0)),
        float(bs.get("level", 1)),
        float(bs.get("morale", 50)),
        float(bs.get("fatigue", 0)),
        float(bs.get("cooldown_time", 0)),
    ]

    # Location: x, y, z, facing_direction (map to number), velocity: vx, vy, vz
    loc = features.get("location", {})
    x = float(loc.get("x", 0))
    y = float(loc.get("y", 0))
    z = float(loc.get("z", 0))
    facing = loc.get("facing_direction", "north")
    facing_map = {"north": 0, "east": 1, "south": 2, "west": 3}
    fd = float(facing_map.get(facing.lower(), 0))
    vel = loc.get("velocity", {})
    vx = float(vel.get("vx", 0))
    vy = float(vel.get("vy", 0))
    vz = float(vel.get("vz", 0))
    location_features = [x, y, z, fd, vx, vy, vz]

    # Combat: can_attack, is_under_attack, enemy_count, attack_range, critical_hit_chance
    combat = features.get("combat", {})
    can_attack = 1.0 if combat.get("can_attack", False) else 0.0
    is_under_attack = 1.0 if combat.get("is_under_attack", False) else 0.0
    enemy_count = float(combat.get("enemy_count", 0))
    attack_range = float(combat.get("attack_range", 3.0))
    critical_hit_chance = float(combat.get("critical_hit_chance", 0))
    combat_features = [
        can_attack,
        is_under_attack,
        enemy_count,
        attack_range,
        critical_hit_chance,
    ]

    # Environment: is_water_near, is_daytime, obstacle_density, escape_route_available, visibility_level
    env = features.get("environment", {})
    is_water_near = 1.0 if env.get("is_water_near", False) else 0.0
    is_daytime = 1.0 if env.get("is_daytime", True) else 0.0
    obstacle_density = float(env.get("obstacle_density", 0))
    escape_route_available = 1.0 if env.get("escape_route_available", False) else 0.0
    visibility_level = float(env.get("visibility_level", 1))
    environment_features = [
        is_water_near,
        is_daytime,
        obstacle_density,
        escape_route_available,
        visibility_level,
    ]

    # Inventory: main_hand durability ratio, ammo_count
    inv = features.get("inventory", {})
    main_hand = inv.get("main_hand", {})
    mh_durability = float(main_hand.get("durability", 0))
    mh_max = float(main_hand.get("max_durability", 1))  # กำหนด default เป็น 1 หากไม่มีข้อมูล
    if mh_max == 0:
        mh_ratio = 0.0
    else:
        mh_ratio = mh_durability / mh_max
    ammo_count = float(main_hand.get("ammo_count", 0))
    inventory_features = [mh_ratio, ammo_count]

    # Quest: quest_progress, is_on_mission, time_limit, quest_reward (experience, gold)
    quest = features.get("quest", {})
    quest_progress = float(quest.get("quest_progress", 0))
    is_on_mission = 1.0 if quest.get("is_on_mission", False) else 0.0
    time_limit = float(quest.get("time_limit", 0))
    quest_reward = quest.get("quest_reward", {})
    reward_exp = float(quest_reward.get("experience", 0))
    reward_gold = float(quest_reward.get("gold", 0))
    quest_features = [
        quest_progress,
        is_on_mission,
        time_limit,
        reward_exp,
        reward_gold,
    ]

    # Nearby Blocks: จำนวน block ที่อยู่ใกล้
    nearby_blocks = features.get("nearby_blocks", [])
    num_nearby_blocks = float(len(nearby_blocks))

    # Ally Support Nearby: boolean
    ally_support = 1.0 if features.get("ally_support_nearby", False) else 0.0

    # รวมคุณลักษณะทั้งหมดเข้าด้วยกัน (เวกเตอร์ความยาว 36)
    feature_vector = (
        basic
        + location_features
        + combat_features
        + environment_features
        + inventory_features
        + quest_features
        + [num_nearby_blocks, ally_support]
    )

    return np.array(feature_vector, dtype=np.float32)


# =========================================
# (B) REPLAY BUFFER
# =========================================
import collections


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones),
        )

    def size(self):
        return len(self.buffer)


# =========================================
# (C) DQN AGENT
# =========================================
ACTION_MAP = {
    0: "move_forward",
    1: "move_backward",
    2: "turn_left",
    3: "turn_right",
    4: "jump",
    5: "sprint",
    6: "sneak",
    7: "attack_melee",
    8: "attack_ranged",
    9: "block",
    10: "use_item",
    11: "open_inventory",
    12: "craft_item",
    13: "place_block",
    14: "destroy_block",
    15: "trade",
    16: "communicate",
}


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,  # ควรตรงกับจำนวน action ใน ACTION_MAP (เช่น 17)
        buffer_size=10000,
        batch_size=32,
        gamma=0.9,
        lr=0.001,
        update_target_every=500,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target_model()

        self.memory = ReplayBuffer(max_size=buffer_size)
        self.train_step = 0

    def _build_model(self, lr):
        model = Sequential(
            [
                Dense(256, input_dim=self.state_size, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="softmax"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        q_values = self.model.predict(np.array([state]), verbose=0)
        action_idx = np.argmax(q_values[0])
        return action_idx

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        q_current = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = (
                rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(q_next[i])
            )
            q_current[i][actions[i]] = target

        self.model.fit(states, q_current, epochs=1, verbose=0)
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_model()


# =========================================
# (D) NPCEngine (AI MAIN)
# =========================================
class NPCEngine:
    def __init__(self, server_url="http://localhost:5000"):
        self.sio = socketio.Client()
        self.agent = DQNAgent(state_size=36, action_size=len(ACTION_MAP))
        self.sio.connect(server_url)
        self.sio.on("process_request", self.on_process_request)

    def on_process_request(self, data, callback=None):
        try:
            if callback:
                callback({"status": "received"})
            raw_features = data.get("features", {})
            features_vector = extract_features(raw_features)
            action_idx = self.agent.act(features_vector)
            self.agent.remember(
                features_vector,
                action_idx,
                data["reward"],
                features_vector,
                data["done"],
            )
            self.agent.replay()
            action_name = self._action_to_command(action_idx)
            details = self._generate_details_for(action_name, raw_features)
            response = {
                "uuid": data["uuid"],
                "name": data["name"],
                "decision": {"action": action_name, "details": details},
            }
            self.sio.emit("process_response", response)
            logging.info(f"Sent decision => {response}")
        except Exception as e:
            logging.exception(f"Error in process_request: {e}")

    def _action_to_command(self, action_idx):
        """แมป action index จาก DQN ให้เป็นชื่อ action โดยตรง"""
        return ACTION_MAP.get(action_idx, "idle")

    def _generate_details_for(self, action, raw_features):
        """
        สร้างรายละเอียดสำหรับแต่ละ action
        """
        if action in [
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "jump",
            "sprint",
            "sneak",
        ]:
            return {"speed": 1.0, "direction": "default"}
        elif action in ["attack_melee", "attack_ranged", "block"]:
            combat = raw_features.get("combat", {})
            enemy_details = combat.get("enemy_details", [])
            target_id = enemy_details[0]["type"] if enemy_details else "unknown"
            return {"target_id": target_id}
        elif action in [
            "use_item",
            "open_inventory",
            "craft_item",
            "place_block",
            "destroy_block",
        ]:
            return {"info": "action_details_optional"}
        elif action in ["trade", "communicate"]:
            return {"message": "optional_message"}
        else:
            return {}

    def run(self):
        while True:
            time.sleep(1)


# =========================================
# (E) MAIN
# =========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = NPCEngine(server_url="http://localhost:5000")
    engine.run()
