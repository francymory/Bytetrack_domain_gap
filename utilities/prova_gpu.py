import torch

def check_pytorch_gpu():
    if torch.cuda.is_available():
        print(f"PyTorch: GPU disponibile - {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch: GPU non disponibile, uso della CPU")

if __name__ == "__main__":
    check_pytorch_gpu()
    
