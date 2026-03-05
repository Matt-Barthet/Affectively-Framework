import uuid
from abc import ABC

from mlagents_envs.side_channel import SideChannel, IncomingMessage


class AffectivelySideChannel(SideChannel, ABC):
	
	def __init__(self, socket_id: uuid.UUID):
		super().__init__(socket_id)
		self.levelEnd = False
		self.interactiveReset = False
		self.arousal_vector = []
	
	def on_message_received(self,
	                        msg: IncomingMessage) -> None:
		test = msg.read_string()
		self.levelEnd = False
		
		if test == '[Level Ended]':
			self.levelEnd = True
			self.interactiveReset = True
		elif '[Vector]' in test:
			test = test.removeprefix("[Vector]:")
			self.arousal_vector = [float(value) for value in test.split(",")[:-1]]
