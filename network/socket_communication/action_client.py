from socket import SocketKind
from network.socket.udp.client.udp_client import UDPClient


class ActionSender(UDPClient):

    def __init__(self, client_cfg: (str, int, int, SocketKind)):
        ip, port, data_size, self.network_protocol = client_cfg
        UDPClient.__init__(self, ip, port, data_size)

    def send_action(self, action_bytes):
        self.send_msg(action_bytes)



