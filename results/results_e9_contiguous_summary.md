# E9 contiguous rerun — full matrix

Protocol: frozen E8 splits (test_start + val_ratio, stamped in the
standardized packages); L/H per dataset = 12 h / 1 h physical
(folsom/skippd 720/60, sunlake 24/2, gefcom 12/1 steps); windowing
stride per E8 FULL map {1 min: 30, 30 min: 1, 1 h: 1}; strict
per-window contiguity (timestamp span == (L+H-1)*dt); daytime mask
cos_zenith > 0.05; Smart Persistence reference on the governed test.
Arms: raw = target-only (anchors dropped, ffill/0 NaN policy);
governed = E8 features [cos_zenith, ghi_clearsky,
clearness_index_aligned]. CrossViViT: image datasets only, 8 frames
per window (image_stride = L//8), raw arm = RGB 3-channel.

## folsom

| arm | model | MAE | RMSE | Skill_MAE_% | Skill_RMSE_% | n_day |
|---|---|---|---|---|---|---|
| cleaned | CrossViViT | 73.52072 | 111.09612 | -78.9 | -12.6 | 5993 |
| cleaned | LSTM | 0.02473 | 0.05398 | +11.3 | +13.8 | 8310 |
| cleaned | LightGBM | 0.02553 | 0.05281 | +8.4 | +15.7 | 8310 |
| cleaned | MLP | 0.02971 | 0.05444 | -6.6 | +13.1 | 8310 |
| cleaned | PatchTST-CD | 0.02786 | 0.05556 | +0.0 | +11.3 | 8310 |
| cleaned | SmartPersistence | 0.02786 | 0.06262 | +0.0 | +0.0 | 8310 |
| cleaned | iTransformer | 0.02805 | 0.05542 | -0.7 | +11.5 | 8310 |
| raw | CrossViViT | 83.97502 | 122.84901 | -104.3 | -24.5 | 5993 |
| raw | LSTM | 0.04250 | 0.07180 | -52.5 | -14.7 | 8310 |
| raw | LightGBM | 0.02821 | 0.05619 | -1.2 | +10.3 | 8310 |
| raw | MLP | 0.03036 | 0.05725 | -9.0 | +8.6 | 8310 |
| raw | PatchTST-CD | 0.02825 | 0.05805 | -1.4 | +7.3 | 8310 |
| raw | iTransformer | 0.02571 | 0.05707 | +7.7 | +8.9 | 8310 |

## gefcom2014

| arm | model | MAE | RMSE | Skill_MAE_% | Skill_RMSE_% | n_day |
|---|---|---|---|---|---|---|
| cleaned | LSTM | 0.07433 | 0.11407 | +29.6 | +22.1 | 792 |
| cleaned | LightGBM | 0.07305 | 0.10640 | +30.8 | +27.3 | 792 |
| cleaned | MLP | 0.07341 | 0.10857 | +30.5 | +25.8 | 792 |
| cleaned | PatchTST-CD | 0.07464 | 0.11318 | +29.3 | +22.7 | 792 |
| cleaned | SmartPersistence | 0.10556 | 0.14641 | +0.0 | +0.0 | 792 |
| cleaned | iTransformer | 0.07333 | 0.11221 | +30.5 | +23.4 | 792 |
| raw | LSTM | 0.07961 | 0.11812 | +24.6 | +19.3 | 792 |
| raw | LightGBM | 0.09152 | 0.12556 | +13.3 | +14.2 | 792 |
| raw | MLP | 0.08648 | 0.12295 | +18.1 | +16.0 | 792 |
| raw | PatchTST-CD | 0.08417 | 0.12105 | +20.3 | +17.3 | 792 |
| raw | iTransformer | 0.08348 | 0.12261 | +20.9 | +16.3 | 792 |

## skippd

| arm | model | MAE | RMSE | Skill_MAE_% | Skill_RMSE_% | n_day |
|---|---|---|---|---|---|---|
| cleaned | CrossViViT | 0.11073 | 0.25520 | -113.6 | -51.1 | 2684 |
| cleaned | LSTM | 0.02677 | 0.11686 | -2.7 | -1.9 | 6754 |
| cleaned | LightGBM | 0.03850 | 0.11550 | -47.7 | -0.7 | 6754 |
| cleaned | MLP | 0.04536 | 0.11941 | -74.0 | -4.1 | 6754 |
| cleaned | PatchTST-CD | 0.03119 | 0.11663 | -19.6 | -1.7 | 6754 |
| cleaned | SmartPersistence | 0.02607 | 0.11468 | +0.0 | +0.0 | 6754 |
| cleaned | iTransformer | 0.03060 | 0.11905 | -17.4 | -3.8 | 6754 |
| raw | CrossViViT | 0.06216 | 0.18536 | -19.9 | -9.8 | 2684 |
| raw | LSTM | 0.02759 | 0.11485 | -5.8 | -0.2 | 6754 |
| raw | LightGBM | 0.03604 | 0.11366 | -38.3 | +0.9 | 6754 |
| raw | MLP | 0.04010 | 0.11728 | -53.8 | -2.3 | 6754 |
| raw | PatchTST-CD | 0.02907 | 0.11375 | -11.5 | +0.8 | 6754 |
| raw | iTransformer | 0.02947 | 0.12153 | -13.0 | -6.0 | 6754 |

## sunlake

| arm | model | MAE | RMSE | Skill_MAE_% | Skill_RMSE_% | n_day |
|---|---|---|---|---|---|---|
| cleaned | LSTM | 0.03584 | 0.05926 | +26.9 | +16.2 | 23461 |
| cleaned | LightGBM | 0.03622 | 0.05775 | +26.2 | +18.3 | 23461 |
| cleaned | MLP | 0.03686 | 0.05782 | +24.8 | +18.2 | 23461 |
| cleaned | PatchTST-CD | 0.03669 | 0.05897 | +25.2 | +16.6 | 23461 |
| cleaned | SmartPersistence | 0.04905 | 0.07069 | +0.0 | +0.0 | 23461 |
| cleaned | iTransformer | 0.03853 | 0.06147 | +21.4 | +13.0 | 23461 |
| raw | LSTM | 0.03940 | 0.06341 | +19.7 | +10.3 | 23461 |
| raw | LightGBM | 0.04158 | 0.06361 | +15.2 | +10.0 | 23461 |
| raw | MLP | 0.04003 | 0.06264 | +18.4 | +11.4 | 23461 |
| raw | PatchTST-CD | 0.03915 | 0.06334 | +20.2 | +10.4 | 23461 |
| raw | iTransformer | 0.04017 | 0.06428 | +18.1 | +9.1 | 23461 |

## Crossover margins (skill %)

| dataset | micro | chi_mu | chi_best | chi_mu_3 | chi_best_3 | pool |
|---|---|---|---|---|---|---|
| folsom | +8.4 | +46.0 | +0.6 | +23.8 | +0.6 | LSTM/iTr/PatchTST-CD+CrossViViT |
| gefcom2014 | +30.8 | +8.9 | +6.2 | +8.9 | +6.2 | LSTM/iTr/PatchTST-CD |
| skippd | -47.7 | -35.1 | -41.9 | -37.6 | -41.9 | LSTM/iTr/PatchTST-CD+CrossViViT |
| sunlake | +26.2 | +6.8 | +6.0 | +6.8 | +6.0 | LSTM/iTr/PatchTST-CD |
