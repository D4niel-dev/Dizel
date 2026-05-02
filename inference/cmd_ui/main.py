import argparse
from inference.cmd_ui.app import DizelCMDApp

def main():
    parser = argparse.ArgumentParser(description="Dizel CMD UI")
    parser.add_argument("--checkpoint", "-c", type=str, default="", help="Path to .pt model checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    app = DizelCMDApp(checkpoint=args.checkpoint, device=args.device)
    app.run()

if __name__ == "__main__":
    main()
