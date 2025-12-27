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
|YOLOFusion|Multi|2022|PR|-|-|76.6|39.8|12.52|-|-|
|MDSFYOLO|IR|2025|TGRS|88.5|78.3|87.6|56.7|5.9|28.4|238|
|ours|IR|-|-|87.4|79.7|87.2(+8.9)|55.6(+8.5)|6.2|24.2|253|
|ours-s|IR|-|-|89.3|79.8|88.6(+10.3)|58.5(+11.4)|24|82.2|137|

|Method|Data|Year|Source|P(%)|R(%)|mAP@50|mAP@50-95|Pa(M)|GFLOPS|FPS|
|-|-|-|-|-|-|-|-|-|-|-|
|SSD|IR|2016|ECCV|86.0|69.0|73.0|41.0|26.3|-|-|
|Faster R-CNN|IR|2015|NeurIPS|85.0|69.1|79.3|49.0|41.2|156.3|-|
|RTDETR|IR|2024|CVPR|76.8|64.1|73.7|41.2|33|103.8|27|
|MMTOD-UNIT|Multi|2020|CVPR|-|-|61.5|-|-|-|-|
|RGBT|Multi|2022|IV|-|-|82.9|-|82.7|-|-|
|LRAF-Net|Multi|2024|TNNLS|81.6|75.3|80.5|42.8|18.8|40.5|-|
|ICAFusion|Multi|2024|PR|-|-|79.2|41.4|120.2|-|38.46|
|CSAA|Multi|2023|CVPR|-|-|79.2|41.3|-|-|-|
|GM-DETR|Multi|2024|CVPR|-|-|83.9|45.8|70|176|-|
|PFGF|IR|2025|CVPR|-|-|84.8|47.1|66.9|409.0|-|
|DPSNET|Multi|2025|TIM|-|-|84.9|-|-|-|132|
|IRDFusion|Multi|2025|arXiv|-|-|88.3|50.7|510.5|1213.5|-|
|RSDet|Multi|2024|arXiv|-|-|83.9|43.8|-|-|-|
|C2DFFNet|Multi|2025|TGRS|-|-|76.9|40.8|6.58|14.6|-|
|CIRDet|Multi|2025|TCSVT|-|-|81.2|46.5|-|-|-|
|COFNet|Multi|2025|TMM|-|-|83.6|44.6|90.2|196.1|-|
|DHANet|Multi|2025|TGRS|-|-|74.3|-|-|-|-|
|DPAL|Multi|2025|TGRS|-|-|75.95|-|-|-|-|
|IGT|Multi|2023|KBS|-|-|85.0|43.6|-|-|-|
|MMFN|Multi|2025|TCSVT|-|-|80.8|41.7|176.4|-|-|
|Fusion-Mamba|Multi|2025|arXiv|-|-|84.3|44.4|244.6|-|-|
|ours|IR|-|-|87.4|79.7|87.2(+8.9)|55.6(+8.5)|6.2|24.2|253|
|ours-s|IR|-|-|89.3|79.8|88.6(+10.3)|58.5(+11.4)|24|82.2|137|

#### Quantitative Results on VisDrone
|Method|mAP@50|mAP@50-95|Pa(M)|
|-|-|-|-|
|YOLOv8-n|32.9|19.1|2.6|
|YOLOv8-s|38.0|22.9|9.8|
|YOLOv8-m|41.7|25.4|23|
|YOLOv8-l|43.2|26.5|39|
|YOLOv8-x|44.9|27.6|61|
|YOLOv9-t|33.0|19.3|1.7|
|YOLOv9-s|38.9|23.4|6.3|
|YOLOv9-m|42.9|26.2|16|
|YOLOv9-c|43.6|26.4|21|
|YOLOv9-e|46.7|29.0|53|
|YOLOv10-n|33.0|19.2|2.7|
|YOLOv10-s|38.3|22.9|8.0|
|YOLOv10-m|41.7|25.5|16|
|YOLOv10-b|43.1|26.3|20|
|YOLOv10-l|43.7|26.7|25|
|YOLOv10-x|45.0|27.7|31|
|YOLOv11-n|31.8|18.2|2.5|
|YOLOv11-s|38.4|22.9|9.4|
|YOLOv11-m|43.0|26.3|19|
|YOLOv11-l|43.8|26.9|25|
|YOLOv11-x|45.8|28.5|56|
|baseline|33.5|19.1|2.4|
|ours|44.0|25.8|6.2|
|ours-s|49.0|29.6|24|

|Method|Year||||
 




<!--stackedit_data:
eyJoaXN0b3J5IjpbODcyNTk1NDgyLC00MjM4MzQ4OTBdfQ==
-->