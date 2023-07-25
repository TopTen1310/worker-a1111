import time

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(params):
    config = {
        "sd": "http://127.0.0.1:3000",
        "lora": "http://127.0.0.1:7861",
        "api": {
            "txt2img":  ("POST", "/sdapi/v1/txt2img"),
            "img2img":  ("POST", "/sdapi/v1/img2img"),
            "getModels": ("GET", "/sdapi/v1/sd-models"),
            "getOptions": ("GET", "/sdapi/v1/options"),
            "setOptions": ("POST", "/sdapi/v1/options"),
            "createTraining": ("POST", "/loraapi/v1/training"),
            "getTrainingLog": ("GET", "/loraapi/v1/training"),
            "checkTraining": ("POST", "/loraapi/v1/training"),
            "terminateTraining": ("PUT", "/loraapi/v1/training"),
            "deleteTraining": ("DELETE", "/loraapi/v1/training"),
        },
        "timeout": 600
    }

    api_name = params["api_name"]
    path = None

    if api_name in config["api"]:
        api_config = config["api"][api_name]
    else:
        raise Exception("Method '%s' not yet implemented")

    api_verb = api_config[0]
    api_path = api_config[1]

    response = {}
    baseUrl = config["sd"]
    if "Training" in api_name and api_name != "createTraining":
        train_id = params["train_id"]
        api_path += f"/{train_id}"
        baseUrl = config["lora"]

    if api_name == "createTraining":
        baseUrl = config["lora"]

    if api_verb == "GET":
        response = automatic_session.get(
            url='%s%s' % (baseUrl, api_path),
            timeout=config["timeout"])

    if api_verb == "POST":
        response = automatic_session.post(
            url='%s%s' % (baseUrl, api_path),
                json=params,
            timeout=config["timeout"])

    if api_verb == "PUT":
        response = automatic_session.put(
            url='%s%s' % (baseUrl, api_path),
            timeout=config["timeout"])

    if api_verb == "DELETE":
        response = automatic_session.delete(
            url='%s%s' % (baseUrl, api_path),
            timeout=config["timeout"])

    return response.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    json = run_inference(event["input"])

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return json


if __name__ == "__main__":
    wait_for_service(url='http://127.0.0.1:3000/sdapi/v1/txt2img')

    print("WebUI API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})
