import paramiko
import os
import time
import socket

key_path = os.path.expanduser("~/.ssh/id_ed25519")
host = "ssh.runpod.io"
user = "hn05v8n20u7btj-6441183f"
b64_path = r"C:\Users\Aakif\Desktop\FitFusion\patches.b64"

with open(b64_path, "r", encoding="utf-8") as f:
    b64_data = f.read().replace('\n', '').replace('\r', '')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting...")
ssh.connect(host, username=user, key_filename=key_path)

print("Invoking shell...")
chan = ssh.invoke_shell()

time.sleep(2)
print("Sending commands...")
chan.send("cd /workspace\n")
time.sleep(1)
chan.send("mkdir -p FitFusion\ncd FitFusion\n")
time.sleep(1)
chan.send("echo > patches.b64\n")
time.sleep(1)

print(f"Sending base64 payload ({len(b64_data)} bytes)...")
chunk_size = 1024
for i in range(0, len(b64_data), chunk_size):
    chunk = b64_data[i:i+chunk_size]
    chan.send(f"echo -n '{chunk}' >> patches.b64\n")
    time.sleep(0.05)
    
print("Decoding and extracting...")
chan.send("base64 -d patches.b64 > patches.tar\n")
time.sleep(1)
chan.send("tar -xf patches.tar\n")
time.sleep(1)
chan.send("echo 'DONE!'\n")
time.sleep(1)

resp = b""
while chan.recv_ready():
    resp += chan.recv(8192)

print(resp.decode('utf-8', 'ignore')[-500:])
ssh.close()
