
import socketserver

class UdpRequestHandler(socketserver.BaseRequestHandler):

    def __init__(self, callback_handler, request, client_address, server):
        self.callback_handler = callback_handler
        super().__init__(request=request, client_address=client_address, server=server)

    def handle(self):
        client_ip, client_port = self.client_address
        data = self.request[0]
        sock = self.request[1]


        if data:
            self.callback_handler(data)
        else:
            print("[{}, {}] connection closed".format(client_ip, client_port))




