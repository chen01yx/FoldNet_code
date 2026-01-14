import socket
import struct


def sendall(sock: socket.socket, msg: str):
    assert isinstance(msg, str), type(msg)
    assert isinstance(sock, socket.socket), type(s)

    data = msg.encode()
    sock.sendall(struct.pack('!I', len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket) -> str:
    l_data = sock.recv(4)
    if not l_data:
        return None
    data_len = struct.unpack('!I', l_data)[0]
    data = sock.recv(data_len)
    return data.decode()