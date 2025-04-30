import uuid
from abc import ABC

from mlagents_envs.side_channel import SideChannel, IncomingMessage


class AffectivelySideChannel(SideChannel, ABC):
	
	def __init__(self,
	             socket_id: uuid.UUID):
		"""
		Creates an AffectivelySideChannel object.
		Args:
			socket_id: The socket ID of the side channel.
		"""
		super().__init__(socket_id)
		self.levelEnd = False
		self.arousal_vector = []
	
	def on_message_received(self,
	                        msg: IncomingMessage) -> None:
		"""
		Process the incoming message from the side channel.
		
		Args:
			msg: The incoming message from the side channel.
		"""
		test = msg.read_string()
		self.levelEnd = False
		
		if test == '[Level Ended]':
			self.levelEnd = True
		elif '[Vector]' in test:
			test = test.removeprefix("[Vector]:")
			self.arousal_vector = [float(value) for value in test.split(",")[:-1]]
