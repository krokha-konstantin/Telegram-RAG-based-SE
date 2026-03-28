from telethon.tl import patched

class Channel():
    def __init__(self, link: str, username: str, desc: str, lang: str, cat: str, subs: int):
        self.link = link
        self.username = username
        self.description = desc
        self.language = lang
        self.subs = subs
        
    def __str__(self):
        return self.username

class Message():
    def __init__(self, message: patched.Message, channel: Channel):
        self.text = message.text
        self.date = message.date
        self.id = message.id
        self.channel_username = channel.username
        self.link = f"https://{channel.username[1::]}/{message.id}"
        self.embedding = None