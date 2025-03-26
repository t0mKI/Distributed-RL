from threading import Thread
from functools import partial
from abc import ABC
from socketserver import ThreadingUDPServer

from socket import SocketKind
from network.socket.udp.server.udp_request_handler import UdpRequestHandler


class MultiplexServer(ABC):
    SERVERS = {}

    def __init__(self, *listening_servers: [(str, str, int, SocketKind, {})]):
        '''
        :param listening_servers: [(str,str,int,Callable[[Arg1Type, Arg2Type], ReturnType])]
                each clients contains a 'name', 'ip' , port, protocol_family and callback funktion for a function handler
        '''

        for listening_server in listening_servers:
            name, ip, port, protocol_family, server_cfg = listening_server
            self.SERVERS[name] = {}
            callback_fkt = server_cfg['callback_fkt']

            # create UDP server
            if protocol_family is SocketKind.SOCK_DGRAM:
                self.SERVERS[name]['server'] = ThreadingUDPServer((ip, port), partial(UdpRequestHandler, callback_fkt))
                protocol_str = "UDP"
            # create TCP server
            elif protocol_family is SocketKind.SOCK_STREAM:
                pass

            self.SERVERS[name]['thread'] = Thread(target=self.SERVERS[name]['server'].serve_forever)
            self.SERVERS[name]['thread'].start()
            print( str(protocol_str) + '-Server "' + name + '" listening on address ' + str(ip) + ':' + str(port))
        print('')