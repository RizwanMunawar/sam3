# SAM 3: Segment Anything 3

![SAM 3 architecture](assets/model_diagram.png?raw=true) SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor [SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new [SA-CO benchmark](https://github.com/facebookresearch/sam3?tab=readme-ov-file#sa-co-dataset) which contains 270K unique concepts, over 50 times more than existing benchmarks.

This breakthrough is driven by an innovative data engine that has automatically annotated over 4 million unique concepts, creating the largest high-quality open-vocabulary segmentation dataset to date. In addition, SAM 3 introduces a new model architecture featuring a presence token that improves discrimination between closely related text prompts (e.g., “a player in white” vs. “a player in red”), as well as a decoupled detector–tracker design that minimizes task interference and scales efficiently with data.

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a new Conda environment:**

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **Clone the repository and install the package:**

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

## Getting Started

⚠️ Note: Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face [repo](https://huggingface.co/facebook/sam3). 
Once accepted, you can use the `sam3.pt` model file for inference. 


### Basic Usage

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

SAM 3 consists of a detector and a tracker that share a vision encoder. It has 848M parameters. 
The detector is a DETR-based model conditioned on text, geometry, and image  exemplars. The tracker 
inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement.

## Contributing

See [contributing](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov,
Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour,
Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya,
Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney,
Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann,
Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz,
Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha
Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan,
Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake.

Meta Superintelligence Labs

[Nicolas Carion](https://www.nicolascarion.com/)\*,
[Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en)\*,
[Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)\*,
[Shoubhik Debnath](https://scholar.google.com/citations?user=fb6FOfsAAAAJ&hl=en)\*,
[Ronghang Hu](https://ronghanghu.com/)\*,
[Didac Suris](https://www.didacsuris.com/)\*,
[Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en)\*,
[Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en)\*,
[Haitham Khedr](https://hkhedr.com/)\*, Andrew Huang,
[Jie Lei](https://jayleicn.github.io/),
[Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en),
[Baishan Guo](https://scholar.google.com/citations?user=BC5wDu8AAAAJ&hl=en),
Arpit Kalla, [Markus Marks](https://damaggu.github.io/),
[Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en),
Meng Wang, [Peize Sun](https://peizesun.github.io/),
[Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en),
[Triantafyllos Afouras](https://www.robots.ox.ac.uk/~afourast/),
[Effrosyni Mavroudi](https://scholar.google.com/citations?user=vYRzGGEAAAAJ&hl=en),
[Katherine Xu](https://k8xu.github.io/)°,
[Tsung-Han Wu](https://patrickthwu.com/)°,
[Yu Zhou](https://yu-bryan-zhou.github.io/)°,
[Liliane Momeni](https://scholar.google.com/citations?user=Lb-KgVYAAAAJ&hl=en)°,
[Rishi Hazra](https://rishihazra.github.io/)°,
[Shuangrui Ding](https://mark12ding.github.io/)°,
[Sagar Vaze](https://sgvaze.github.io/)°,
[Francois Porcher](https://scholar.google.com/citations?user=LgHZ8hUAAAAJ&hl=en)°,
[Feng Li](https://fengli-ust.github.io/)°,
[Siyuan Li](https://siyuanliii.github.io/)°,
[Aishwarya Kamath](https://ashkamath.github.io/)°,
[Ho Kei Cheng](https://hkchengrex.com/)°,
[Piotr Dollar](https://pdollar.github.io/)†,
[Nikhila Ravi](https://nikhilaravi.com/)†,
[Kate Saenko](https://ai.bu.edu/ksaenko.html)†,
[Pengchuan Zhang](https://pzzhang.github.io/pzzhang/)†,
[Christoph Feichtenhofer](https://feichtenhofer.github.io/)†

\* core contributor, ° intern, † project lead, order is random within groups.
