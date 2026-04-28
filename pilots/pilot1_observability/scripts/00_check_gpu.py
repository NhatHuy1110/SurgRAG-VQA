import torch


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("VRAM GB:", round(props.total_memory / 1024**3, 2))
        print("CUDA version:", torch.version.cuda)


if __name__ == "__main__":
    main()

