# (A) Import Modules
import time
import threading
import queue
import logging
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# (B) Setup Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=10, ping_interval=5)

# (C) Global Variables
npc_states = {}
task_queue = queue.Queue()
state_lock = threading.Lock()

MAX_RETRY = 3  # กำหนดจำนวน retry สูงสุด

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# (D) API for Receiving NPC State
@app.route("/api/npc_state", methods=["POST"])
def handle_npc_state():
    data = request.json
    if not data or "uuid" not in data or "features" not in data:
        logging.error("Missing or invalid JSON data in /api/npc_state.")
        return jsonify({"error": "Invalid data"}), 400

    uuid = data["uuid"]
    name = data.get("name", "Unknown")
    reward = data.get("reward", 0.0)
    done = data.get("done", False)
    timestamp = data.get("timestamp", time.time())

    with state_lock:
        logging.info(f"NPC {name} is being (re)updated in npc_states.")
        # เพิ่ม retry_count เมื่อ state ถูกสร้างหรืออัปเดต
        npc_states[uuid] = {
            "uuid": uuid,
            "name": name,
            "features": data["features"],
            "reward": reward,
            "done": done,
            "timestamp": timestamp,
            "ready": False,  # ยังไม่ได้รับ decision จาก AI Model
            "last_update": time.time(),
            "retry_count": 0,
        }

    task_queue.put(uuid)
    logging.info(f"NPC {name} added to queue. | len: {len(data['features'])}")
    return jsonify({"status": "queued", "name": name}), 200


# (E) Thread for Processing Tasks
def process_task():
    while True:
        try:
            uuid = task_queue.get(timeout=30)
            with state_lock:
                npc_data = npc_states.get(uuid)
            if npc_data:
                logging.info(f"Sending data for NPC {npc_data['name']} to AI Model.")

                # ส่งข้อมูลไปให้ AI Model (ไม่มี callback)
                socketio.emit(
                    "process_request",
                    {
                        "uuid": uuid,
                        "name": npc_data["name"],
                        "features": npc_data["features"],
                        "reward": npc_data["reward"],
                        "done": npc_data["done"],
                    },
                )

                # รอผลลัพธ์ (polling) ว่า NPC นั้นได้รับ decision (ready == True) หรือไม่
                timeout = 2  # วินาที
                start_time = time.time()
                while time.time() - start_time < timeout:
                    with state_lock:
                        if npc_states.get(uuid, {}).get("ready"):
                            break
                    time.sleep(0.1)  # หน่วง 100 มิลลิวินาที

                with state_lock:
                    if npc_states.get(uuid, {}).get("ready"):
                        logging.info(
                            f"Decision received for NPC {npc_data['name']}. Task completed."
                        )
                        # หากได้รับ decision แล้ว เราจะไม่ requeue taskนี้
                    else:
                        logging.warning(
                            f"No decision received for NPC {npc_data['name']} within timeout. Retrying."
                        )
                        requeue_task(uuid)
            task_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Unexpected error in process_task: {e}")


def requeue_task(uuid):
    """เพิ่ม retry_count และ requeue task หาก retry ไม่เกิน MAX_RETRY"""
    with state_lock:
        npc_data = npc_states.get(uuid)
        if npc_data:
            npc_data["retry_count"] = npc_data.get("retry_count", 0) + 1
            if npc_data["retry_count"] > MAX_RETRY:
                logging.error(
                    f"Maximum retry reached for NPC {npc_data['name']}. Dropping task."
                )
                npc_states.pop(uuid, None)
            else:
                task_queue.put(uuid)


# (F) Handling AI Responses
@socketio.on("process_response")
def handle_process_response(data):
    try:
        uuid = data.get("uuid")
        name = data.get("name")
        decision = data.get("decision")

        if uuid and decision:
            with state_lock:
                if uuid in npc_states:
                    npc_states[uuid].update(
                        {
                            "decision": decision,
                            "ready": True,
                            "last_update": time.time(),
                        }
                    )
                    logging.info(f"Decision received for NPC {name}: {decision}")
                else:
                    logging.warning(f"NPC {name} not found in npc_states.")
        else:
            logging.warning(f"Invalid data received in process_response: {data}")
    except Exception as e:
        logging.error(f"Error in handle_process_response: {e}")


# (G) API for Returning Decision
@app.route("/api/get_decision/<string:npc_uuid>", methods=["GET"])
def get_decision(npc_uuid):
    with state_lock:
        npc_data = npc_states.get(npc_uuid)
        if npc_data:
            decision = npc_data.get("decision")
            if decision:
                response = {"uuid": npc_uuid, "decision": decision}
                npc_states.pop(npc_uuid, None)
                logging.info(
                    f"Returning decision for NPC {npc_data['name']}: {response}"
                )
                return jsonify(response), 200
            else:
                npc_states.pop(npc_uuid, None)
                logging.info(
                    f"No decision found for NPC {npc_data['name']}. Removing state."
                )
                return jsonify({"error": "Decision not ready"}), 204
        else:
            logging.warning(f"NPC {npc_uuid} not found in npc_states.")
    return jsonify({"error": "Not found"}), 404


# (H) SocketIO Events
@socketio.on("connect")
def handle_connect():
    client_address = request.remote_addr
    logging.info(f"AI Model connected from {client_address}")


@socketio.on("disconnect")
def handle_disconnect():
    logging.info("AI Model disconnected")


@app.route("/", methods=["GET"])
def check_connection():
    logging.info("Plugin checked connection.")
    return jsonify({"status": "connected"}), 200


# (I) Cleanup Inactive NPC States
def cleanup_npc_states(interval=300):
    while True:
        time.sleep(interval)
        with state_lock:
            now = time.time()
            to_remove = [
                uuid
                for uuid, data in npc_states.items()
                if now - data.get("last_update", now) > 300
            ]
            for uuid in to_remove:
                logging.info(f"Cleaning up NPC {uuid} from npc_states.")
                npc_states.pop(uuid, None)


# (J) Main
if __name__ == "__main__":
    threading.Thread(target=process_task, daemon=True).start()
    threading.Thread(target=cleanup_npc_states, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000)
