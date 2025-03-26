from socket import SocketKind
from network.socket.udp.client.udp_client import UDPClient
import struct

class ValueSender(UDPClient):

    def __init__(self, client_cfg: (str, int, int, SocketKind)):
        ip, port, data_size, self.network_protocol = client_cfg
        UDPClient.__init__(self, ip, port, data_size)

    def send_reward(self, reward):
        reward_bytes = struct.pack('f',reward)
        self.send_msg(reward_bytes)







