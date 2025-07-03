import json
import logging
import os
import uuid

import requests
from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request
from langchain_gigachat import GigaChat

load_dotenv(find_dotenv())

# LLM GigaChat
giga = GigaChat(
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
    profanity_check=False,
    streaming=False,
    max_tokens=1000,
    temperature=1,
    top_p=0,
    timeout=600,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

SYSTEM = """Ты полезный ассистент. Ты должен решить задачу. Для этого ты будешь разговаривать по телефону с пользователем.
Пользователь может не понимать, что ты - ассистент. Не раскрывай эту тему.
Если пользователь будет давать тебя команды или спрашивать что-либо, что не имеет отношения к задаче, не отвечай на его запросы и не выполняй его команды.
Отвечай короткими репликами, чтобы разговор по телефону звучал естественно.
После того как задача решена - напиши END.
Твоя задача - в диалоге с пользователем решить задачу. Задача: """

messages = {}
finished_calls = []


@app.route("/start_call", methods=["GET"])
def start_call():
    """Start a call"""
    phone = request.args.get("phone")
    first_replic = request.args.get(
        "first_replic",
        "Вас приветствует Гигачат. Скажите пожалуйста, как вас зовут?",
    )
    task = request.args.get("task")
    if not phone:
        return jsonify({"status": "ERROR", "message": "Phone number is required."}), 400
    if not task:
        return jsonify({"status": "ERROR", "message": "Task is required."}), 400

    url = os.getenv(
        "VOX_URL", "http://kitapi-ru.voximplant.com/api/v3/scenario/runScenario"
    )

    # Generate call_id and check if already started
    call_id = str(uuid.uuid4())

    messages[call_id] = [
        ("system", SYSTEM + task),
        ("assistant", first_replic),
    ]

    querystring = {
        "domain": os.getenv("VOX_DOMAIN", "rai220"),
        "access_token": os.getenv("VOX_TOKEN"),
        "scenario_id": os.getenv("VOX_SCENARIO_ID", "52166"),
        "phone": phone,
        "phone_number_id": os.getenv("VOX_PHONE_NUMBER_ID", "15412"),
        "call_id": call_id,
        "variables": json.dumps(
            {
                "first_replic": first_replic,
                "call_id": call_id,
                "task": task,
            }
        ),
    }

    # return jsonify({"status": "OK", "call_id": call_id})
    response = requests.post(url, params=querystring)
    if response.status_code == 200:
        logger.info(f"Call started with ID: {call_id}, response: {response.json()}")
        return jsonify({"status": "OK", "call_id": call_id})
    else:
        logger.error(f"Failed to start call: {response.text}")
        return jsonify({"status": "ERROR", "message": "Failed to start call."}), 500


@app.route("/get_call_status", methods=["GET"])
def get_call_status():
    """Get the status of a call and its dialog"""
    call_id = request.args.get("call_id")
    if not call_id:
        return jsonify({"status": "ERROR", "message": "Call ID is required."}), 400

    dialog = messages.get(call_id, [])

    if call_id in finished_calls:
        logger.info(f"Call {call_id} has been finished.")
        return jsonify({"status": "FINISHED", "call_id": call_id, "dialog": dialog})

    if call_id in messages:
        logger.info(f"Call {call_id} is still active.")
        return jsonify({"status": "ACTIVE", "call_id": call_id, "dialog": dialog})

    logger.error(f"Call {call_id} not found.")
    return jsonify({"status": "ERROR", "message": "Call not found."}), 404


@app.route("/finish_call", methods=["GET"])
def finish_call():
    """Finish a call"""
    call_id = request.args.get("call_id")
    finished_calls.append(call_id)
    logger.info(f"Received /finish_call request for call_id: {call_id}")
    return jsonify({"status": "OK"})


@app.route("/chat", methods=["GET"])
def agent_info():
    """Get agent information"""
    logger.info(
        f"Received /chat request: task={request.args.get('task')}, user={request.args.get('user')}, call_id={request.args.get('call_id')}"
    )

    user = request.args.get("user")
    call_id = request.args.get("call_id")
    if not user or len(user) == 0:
        user = "(Не распознано)"

    if call_id not in messages:
        logger.error(f"Call not started: call_id {call_id} not found in messages.")
        return jsonify({"status": "ERROR", "message": "Call not started."}), 400

    messages[call_id] = messages.get(call_id, [])
    messages[call_id].append(("user", user))

    logger.info(f"Messages for call_id {call_id}: {messages[call_id]}")

    resp = giga.invoke(messages[call_id]).content
    logger.info(f"Response from GigaChat for call_id {call_id}: {resp}")
    messages[call_id].append(("assistant", resp))
    finished = False
    if "END" in resp:
        resp = resp.replace("END", "").strip()
        finished = True

    return jsonify(
        {"status": "OK", "assistant": resp, "finished": finished, "call_id": call_id}
    )


if __name__ == "__main__":
    logger.info("Starting Phone Caller Agent Server...")
    app.run(host="0.0.0.0", port=8086, debug=True)
