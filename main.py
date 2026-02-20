import warnings
warnings.filterwarnings("ignore")
import argparse
import src.modes.offline_mode as offline_mode
import src.modes.online_mode as online_mode
from src.utils import config



def main(parsed_args):
    if parsed_args.mode == "offline":
        #offline_mode.run_offline_mode("breakfast")
        history = offline_mode.run_offline_mode(parsed_args.dataset, log=True)
        config.saveCSV(f"history_{parsed_args.dataset}.csv", history)
    
    if parsed_args.mode == "online":
        online_mode.run_online_mode(parsed_args.dataset, log=True)
    
        



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyABD: Action Boundary Detection Framework")
    args.add_argument("--mode", type=str, default="offline", help="Mode to run: offline or online")
    args.add_argument("--dataset", type=str, default="breakfast", help="Dataset to use: breakfast, 50salads")
    parsed_args = args.parse_args() 

    main(parsed_args)