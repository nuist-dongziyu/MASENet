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
![MASENet Framework](https://github.com/dongziyu89-nuist/NUIST-dongziyu-MASENet/blob/main/Network.png)
## Recommended Environment
1. python 3.10.18
2. pytorch 2.7.1
3. torchvision 0.22.1
## Datasets
1. FLIR
2. VisDrone
3. 
## Results
#### Visualization results

#### Grad-CAM results

#### Quantitative Results on FLIR、VisDrone、IRSTD-1k and SIRST-V2 dataset merging
 




<!--stackedit_data:
eyJoaXN0b3J5IjpbNzA3MTU4MDksMTU2MzIxNjA1N119
-->