import subprocess
import gc
import shutil
import os

# Define the command to run along with its arguments
command = ["/global/cfs/projectdirs/dune/www/data/2x2/simulation/silentc_work/miniforge-pypy3/envs/point_mae2/bin/python", "main.py", "--config", "cfgs/edepsim_pretrain.yaml", "--exp_name", "train_on_one_npy_file"]


'''
# Open a subprocess with stdout and stderr pipes
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Continuously read and print output as it comes
for line in iter(process.stdout.readline, ''):
    print(line, end='')  # Print each line of stdout
    # You can process the output line by line here if needed

# Wait for the subprocess to finish and get the return code
process.communicate()

# Check the return code
if process.returncode == 0:
    print("Script executed successfully!")
else:
    print("Script execution failed!")
'''    
for i in range(400):    
    # Open a subprocess with stdout and stderr pipes
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously read and print output from stdout and stderr as it comes
    while True:
        output_stdout = process.stdout.readline()
        output_stderr = process.stderr.readline()
        if output_stdout:
            print(output_stdout.strip())  # Print stdout
        if output_stderr:
            print(output_stderr.strip())  # Print stderr
        if not output_stdout and not output_stderr:
            break  # Break the loop if both streams are empty

    # Wait for the subprocess to finish and get the return code
    process.communicate()

    # Check the return code
    if process.returncode == 0:
        print(f"train script executed successfully for epoch {i}")
    else:
        print(f"train script execution failed at epoch {i}") 
    
    # rename ckpt file from train to use later (and so it wont be overriten)
    source_file = "./experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last.pth"
    destination_file = "./experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last_train.pth"
    
    shutil.copyfile(source_file, destination_file)
    
    
    command_val  = ["/global/cfs/projectdirs/dune/www/data/2x2/simulation/silentc_work/miniforge-pypy3/envs/point_mae2/bin/python", "main.py", "--config", "cfgs/edepsim_pretrain_val.yaml", "--exp_name", "train_on_one_npy_file_val", "--ckpts", "/global/cfs/projectdirs/dune/www/data/2x2/simulation/silentc_work/point_mae/Point-MAE/experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last.pth", "--resume"]     
    
    # Open a subprocess with stdout and stderr pipes
    process_val = subprocess.Popen(command_val, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously read and print output from stdout and stderr as it comes
    while True:
        output_stdout = process_val.stdout.readline()
        output_stderr = process_val.stderr.readline()
        if output_stdout:
            print(output_stdout.strip())  # Print stdout
        if output_stderr:
            print(output_stderr.strip())  # Print stderr
        if not output_stdout and not output_stderr:
            break  # Break the loop if both streams are empty

    # Wait for the subprocess to finish and get the return code
    process_val.communicate()

    # Check the return code
    if process_val.returncode == 0:
        print(f"val script executed successfully for epoch {i}")
    else:
        print(f"val script execution failed at epoch {i}") 
        
        
    # deleting ckpt file from val and keeping ckpt file from loop train     
    source_file = "./experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last.pth"
    destination_file = "./experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last_train.pth"

    # Deleting the original file
    if os.path.exists(source_file):
        os.remove(source_file)
        print(f"Original file '{source_file}' deleted successfully.")
    else:
        print(f"File '{source_file}' does not exist.")

    # Renaming ckpt-last_train.pth file to 'ckpt-last.pth' for use in next loop
    if os.path.exists(destination_file):
        os.rename(destination_file, source_file)
        print(f"Duplicate file '{destination_file}' renamed to 'ckpt-last.pth'.")
    else:
        print(f"File '{destination_file}' does not exist.")

        
        
    command = ["/global/cfs/projectdirs/dune/www/data/2x2/simulation/silentc_work/miniforge-pypy3/envs/point_mae2/bin/python", "main.py", "--config", "cfgs/edepsim_pretrain.yaml", "--exp_name", "train_on_one_npy_file", "--ckpts", "./experiments/edepsim_pretrain/cfgs/train_on_one_npy_file/ckpt-last.pth", "--resume"] 
    
    if (i + 1) % 10 == 0:  # Adding 1 to i because range() starts from 0
        gc.collect()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
# Run the command
process = subprocess.run(command, capture_output=True, text=True)

# Check the output and return code
if process.returncode == 0:
    print("Script executed successfully!")
    print("Output:")
    print(process.stdout)
else:
    print("Script execution failed!")
    print("Error:")
    print(process.stderr)
'''