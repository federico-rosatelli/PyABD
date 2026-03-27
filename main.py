import warnings
warnings.filterwarnings("ignore")
import argparse
from tests import test_accuracy



def main(parsed_args):
    if parsed_args.test:
        if parsed_args.mode == "offline":
            test_accuracy.testOfflineMode(parsed_args.dataset)
        if parsed_args.mode == "online":
            test_accuracy.testOnlineMode(parsed_args.dataset)
        return

    if parsed_args.mode == "offline":
        test_accuracy.testOfflineMode(parsed_args.dataset, 0.6, 7)
        return
    
    if parsed_args.mode == "online":
        test_accuracy.testOnlineMode(parsed_args.dataset)
        return



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyABD: Action Boundary Detection Framework")
    args.add_argument("--mode", type=str, default="offline", help="Mode to run: offline or online")
    args.add_argument("--dataset", type=str, default="breakfast", help="Dataset to use: breakfast, 50salads")
    args.add_argument("--boundaries", type=str, default="eval", help="Boundaries to use: eval, mid")
    args.add_argument("--test", action="store_true", help="Run test")
    parsed_args = args.parse_args() 

    main(parsed_args)