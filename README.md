# MASENet：Multi-level Adaptive Selection and Enhancement Network for Infrared Small Target Detection
## Abstract
Infrared small target detection plays a critical role in complex environments with varying illumination and low contrast. However, existing methods still suffer from weak target features, low signal-to-noise ratios, and severe confusion between small targets and complex backgrounds.

To address these challenges, we propose **MASENet**, an efficient enhancement network specifically designed for infrared small target detection. The core idea is to explicitly strengthen fragile small-target features through **multi-level adaptive selection and enhancement mechanisms**.

Specifically, we introduce the **MASEM (Multi-level Adaptive Selection and Enhancement Module)**, which incorporates a series of adaptive selection operators across feature hierarchies with different intensities, effectively improving feature preservation and representation for small targets. Building upon MASEM, we further develop the **MCAFEM (Multi-layer Cascaded Adaptive Feature Enhancement Module)** for feature extraction and the **MASEM-Head**, and seamlessly integrate them into the **YOLO (v5–v13) family**.

Extensive experiments demonstrate that the proposed modules consistently bring significant performance gains across multiple benchmarks:

-   On the **FLIR** dataset, MASENet achieves **state-of-the-art (SOTA)** performance, improving mAP@50 and mAP@50–95 by **8.9%** and **8.5%**, respectively, over the YOLOv13n baseline;
    
-   On the **VisDrone** dataset, the proposed method improves mAP@50 and mAP@50–95 by **10.5%** and **6.7%**, respectively, and outperforms all YOLO (v5–v11) variants across all model scales;
    
-   On the merged **IRSTD-1k + SIRST-V2** infrared small target dataset, MASENet achieves **77.8% mAP@50** and **75.4% recall**, maintaining a favorable balance between precision and recall under low-contrast and small-scale target scenarios.

Moreover, with fewer parameters and lower computational cost, the **S-scale model** of MASENet delivers performance comparable to Transformer-based detectors, highlighting its efficiency, robustness, and strong adaptability in challenging infrared detection scenarios.
## Network
![MASENet Framework](https://github.com/dongziyu89-nuist/NUIST-dongziyu-MASENet/blob/main/assets/Network.png)
## Recommended Environment
- python 3.10.18
- pytorch 2.7.1
- torchvision 0.22.1
## Datasets
- FLIR
-  VisDrone
- IRSTD-1k + SIRST-V2 

## Results
#### Visualization results
![result1](https://github.com/dongziyu89-nuist/NUIST-dongziyu-MASENet/blob/main/assets/result1.png)
![result2](https://github.com/dongziyu89-nuist/NUIST-dongziyu-MASENet/blob/main/assets/result2.png)
#### Grad-CAM results
![result3](https://github.com/dongziyu89-nuist/NUIST-dongziyu-MASENet/blob/main/assets/result3.png)
#### Quantitative Results on FLIR
|Method|Data|Year|Source|P(%)|R(%)|mAP@50|mAP@50-95|Pa(M)|GFLOPS|FPS|
|-|-|-|-|-|-|-|-|-|-|-|
|YOLOv3-Tiny|IR|2018|arXiv|82.6|57.0|66.2|40.4|12.2|19|-|
|YOLOv5n|IR|2020|7.0u|84.8|68.6|78.6|47.2|2.2|6.1|-|
|YOLOv6n|IR|2022|arXiv|83.2|65.3|75.3|46.2|5.1|11.8|-|
|YOLOv8n|IR|2024|8.3u|83.8|66.8|76.7|46.4|3.2|7.1|-|
|YOLOv9t|IR|2024|ECCV|83.5|70.0|77.9|47.4|2.1|8.2|-|
|YOLOv10n|IR|2024|NeurIPS|85.3|68.3|78.7|48.6|3.4|9.5|-|
|YOLOv11n|IR|2024|arXiv|85.8|69.4|79.3|47.9|2.6|6.3|-|
|YOLOv12n|IR|2025|arXiv|84.8|68.4|78.1|47.0|2.5|6.3|603|
|YOLOv13n|IR|2025|arXiv|85.2|67.8|78.3|47.1|2.4|6.2|561|
|GoldYolo|IR|2023|NeurIPS|83.8|62.0|74.1|43.8|9.2|12.1|-|
|LeYolo|IR|2025|arXiv|83.2|65.1|74.5|44.8|1.1|2.7|-|
|CM-YOLO|IR|2025|TGRS|-|-|80.4|-|16.97|24.92|-|
|YOLO-CIR|IR|2023|arXiv|-|-|84.9|-|35.9|-|-|
|CDC-YOLO|Multi|2025|TIV|-|-|83.1|44.7|153.6|-|-|
|YOLOAdaptor|Multi|2024|TIV|-|-|80.1|-|-|-|-|
|Dual-YOLO|Multi|2023|Sensors|-|-|84.5|-|175.1|-|-|
|YOLOFusion|Multi|2022|PR|-|-|76.6|39.8|1|||


 




<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM3MDQ3NzE2OSwxMzM3MjgwNDcwLC0xMT
k2MzUxODM1LDEyODc0MzU5ODddfQ==
-->