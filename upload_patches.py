import paramiko
from scp import SCPClient
import os

key_path = os.path.expanduser("~/.ssh/id_ed25519")
host = "ssh.runpod.io"
user = "hn05v8n20u7btj-6441183f"
# In typical runpod ssh, the subdomain handles the routing, port 22.

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(host, username=user, key_filename=key_path)
    with SCPClient(ssh.get_transport()) as scp:
        print("Uploading patches.tar...")
        scp.put("patches.tar", remote_path="/workspace/")
    
    # Extract
    stdin, stdout, stderr = ssh.exec_command("cd /workspace/FitFusion && tar -xf ../patches.tar")
    print(stdout.read().decode())
    print(stderr.read().decode())
    print("DONE!!!")
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()
