from src.modes.offline_mode import run_grid_search
from src.modes.online_mode import run_online_mode


def testBrakfast():
    alphas_to_test = [0.1, 0.2, 0.4, 0.6, 0.8]

    ks_to_test = [5,7,6,10,12] 

    run_grid_search(
        dataset_name="breakfast", 
        boundaries_type="eval", 
        alphas=alphas_to_test, 
        Ks=ks_to_test,
        output_csv="grid_search_breakfast.csv"
    )


def test50Salads():
    alphas_to_test = [0.1, 0.2, 0.4, 0.6, 0.8]

    ks_to_test = [12, 19]

    run_grid_search(
        dataset_name="50salads", 
        boundaries_type="eval", 
        alphas=alphas_to_test, 
        Ks=ks_to_test,
        output_csv="grid_search_50salads.csv"
    )

def testOnlineMode(dataset):
    run_online_mode(dataset_name=dataset, boundaries_type="eval", log=False)