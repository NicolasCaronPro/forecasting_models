import subprocess
def reset_gpu(bash_script = "src/tools/reset_gpu.sh"):
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        # Allouer 1 MB de mémoire sur le GPU
        d_a = cuda.mem_alloc(1024 * 1024)
        print("Allocation de mémoire réussie.")
    except Exception as e:
        print(e)
        try:
            # Try to restart ollama
            # Try to reset the GPU
            result = subprocess.run(['sudo', bash_script], check=True, text=True, capture_output=True)

            # Print the script's output if needed
            print("Script output:", result.stdout)

            import pycuda.driver as cuda
            import pycuda.autoinit
            # Allouer 1 MB de mémoire sur le GPU
            d_a = cuda.mem_alloc(1024 * 1024)
            print("Allocation de mémoire réussie.")
        except subprocess.CalledProcessError as err:
            # Handle errors
            print(f"Error: {err}")
            print(f"Script stderr: {err.stderr}")

if __name__ == "__main__":
    reset_gpu()