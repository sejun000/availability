import subprocess

# Get the list of processes
result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE)
processes = result.stdout.decode('utf-8').splitlines()

# Iterate through the processes and find the target process
for process in processes:
    if 'python3 monte_carlo_simulation.py' in process:
        # Extract the PID (second column)
        pid = int(process.split()[1])
        # Kill the process
        subprocess.run(['kill', '-9', str(pid)])
        print(f"Killed process with PID: {pid}")

