"""In-process, addressed communication; payloads never share mutable state."""
from copy import deepcopy


class MessageBus:
    def __init__(self):
        self.messages = {}
        self.message_history = []

    def send(self, sender, receiver, content):
        msg = {"sender": sender, "receiver": receiver, "content": deepcopy(content)}
        self.messages.setdefault(receiver, []).append(msg)
        self.message_history.append(deepcopy(msg))

    def broadcast(self, sender, content, receivers):
        for receiver in receivers:
            self.send(sender, receiver, content)

    def receive(self, receiver):
        return self.messages.pop(receiver, [])

    def peek(self, receiver):
        return deepcopy(self.messages.get(receiver, []))

    def clear(self):
        self.messages.clear()
        self.message_history.clear()

    def get_history(self, limit=None):
        if limit is None:
            return deepcopy(self.message_history)
        return deepcopy(self.message_history[-limit:]) if limit > 0 else []
