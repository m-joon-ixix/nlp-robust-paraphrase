import requests

from common.secret_utils import load_secret


def slack_notify(message: str, **kwargs):
    params = [{"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in kwargs.items()]

    # reference: https://api.slack.com/messaging/webhooks#advanced_message_formatting
    data = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": message}},
            {
                "type": "section",
                "block_id": "section789",
                "fields": params,
            },
        ],
    }

    slack_webhook_url = load_secret("slack_webhook_url")
    response = requests.post(slack_webhook_url, json=data)
    return response.status_code
