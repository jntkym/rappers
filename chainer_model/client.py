# Echo client program
import socket
import sys

HOST = '0.0.0.0'    # The remote host
PORT = 50100              # The same port as used by the server
s = None
for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
    af, socktype, proto, canonname, sa = res
    try:
        s = socket.socket(af, socktype, proto)
    except socket.error, msg:
        s = None
        continue
    try:
        s.connect(sa)
    except socket.error, msg:
        s.close()
        s = None
        continue
    break

if s is None:
    print 'could not open socket'
    sys.exit(1)

while 1:
    send_data = str(raw_input("input seed: "))
    if send_data == "close":
        break
    s.send(send_data)
    while(1):
        received_data = s.recv(1024)
        if received_data == "finish":
            break
        else:
            print received_data,

s.close()
