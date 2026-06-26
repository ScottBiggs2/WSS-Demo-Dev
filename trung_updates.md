# Jun 26, 2026

1. Added WSS variants `wss_trung`, `wss_trung_1`, `wss_trung_2`, and `wss_trung_3` allow to use standard optimization.
    - `wss_trung`: Init U, Sigma, V as u do. Then let L = U sqrt Sigma and R = sqrt Sigma V.
    - `wss_trung_1`: SVD the original weight of the model (if any, otherwise init a random kaiming one), then construct L and R based on the output components.
    - `wss_trung_2`: initializes `L/R` directly so the product `LR` matches the detected layer init variance (kaiming/xavier).
    - `wss_trung_3`: Since `wss_trung_1` truncates the original weight, so `LR` doesn't follow the original variance. This approach init `L` and `R` as `wss_trung_1`, then rescale them to make sure `LR` matches the original init variance.
 
2. Add larger scale experiments in the file `src/complex/experiments/headline_torchvision_vit.py`.
    - Use `vit_b_16` raw model from torch vision.
    - Add cifar10, flowers102, pets.
    - Add shared seed handling to make sure all experiments run on the same seed.
    - Add tqdm epoch progress bars (i love tqdm bar :) ).

3. New results in `src/complex/experiments/outputs/trung`, all run on the same set of hyperparams with lr=1e-3 (default)
    - `scott_experiment_files`: run by using scott's experiment files (`headline_mnist.py`, `headline_vit.py`): dense > wss_trung* > wss
    - `trung_experiment_files`: rung by using trung's experiment file (`headline_torchvision_vit.py`) - larger model: wss > wss_trung* > dense.