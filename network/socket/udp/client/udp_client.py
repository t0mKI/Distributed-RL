import socket

class UDPClient():

    def __init__(self, ip:str, port:int, data_size):
        self.ip = ip
        self.port=port
        self.data_size=data_size

    def send_msg(self, msg):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock.sendto(msg, (self.ip, self.port))
        self.sock.close()

