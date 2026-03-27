from src.modes.offline_mode import run_grid_search
from src.modes.online_mode import run_online_mode


def testOfflineMode(dataset, alphas=None, k=None):
    if alphas is None:
        alphas = [0.1, 0.2, 0.4, 0.6, 0.8]
    if k is None:
        if dataset == "breakfast":
            k = [5, 7, 6, 10, 12]
        else:
            k = [12, 19]
    
    if isinstance(alphas, (int, float)):
        alphas = [alphas]
    if isinstance(k, (int, float)):
        k = [k]

    run_grid_search(
        dataset_name=dataset,
        boundaries_type="eval",
        alphas=alphas,
        Ks=k,
        output_csv=f"grid_search_{dataset}.csv"
    )

def testOnlineMode(dataset):
    run_online_mode(dataset_name=dataset, boundaries_type="mid", log=False)