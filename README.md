# Privacy-First Edge AI Coach (PoC)

This repository contains the Proof of Concept (PoC) for the **2026 ASUS UGen AI League** (Stage I Submission). It is an Edge AI-powered pose evaluation system that provides real-time, privacy-preserving feedback for home workouts or rehabilitation exercises.

## 🚀 Getting Started

Follow these steps to set up and run the program on your local machine.

### 1. Prerequisites
Ensure you have Python 3.10 or 3.11 installed on your system.

### 2. Download and Setup
Open your terminal and create a new directory for the project, then set up a virtual environment:

```bash
# Clone the repository or download the files into a folder
git clone <your-github-repo-url>
cd <your-folder-name>

# Create a virtual environment
uv venv -p 3.10

# Activate the virtual environment
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate

# Install the required dependencies
uv pip install -r requirements.txt
```

### 3. Prepare the Assets
Before running the program, ensure you have the following files in your root directory:
*   `expert.jpg`: A clear image of an expert/coach performing the standard posture.
*   `test_video.mp4`: A video of the user attempting to imitate the posture.

### 4. Run the Application
Execute the main script:
```bash
python app.py
```
*Note: The script will automatically download the lightweight `yolov8n-pose.pt` model on its first run.* 

Once completed, the system will generate a `demo_action_score_fullbody.mp4` file in the same directory, demonstrating the real-time scoring and visual feedback.

## 🧠 Program Logic

The current PoC operates on the following core logic:
1.  **Keypoint Extraction:** Utilizes the lightweight YOLOv8-pose model to detect 17 human skeletal keypoints.
2.  **Expert Baseline Calculation:** Reads `expert.jpg` to calculate standard joint angles (shoulders, elbows, hips, knees) to serve as the baseline "Loss" target.
3.  **Real-Time Comparison:** Processes `test_video.mp4` frame by frame, extracting the user's joint angles.
4.  **Scoring & Visual Feedback:** Calculates the absolute difference (Loss) between the expert's and the user's angles. The loss is converted into a 0-100 score. 
5.  **Independent Limb Tracking:** Each of the 4 limbs is scored independently and color-coded (Green for perfect, Yellow for acceptable, Red for needs correction) alongside an overall average score.

## 🔮 Future Outlook (Stage II Roadmap)

Moving forward into Stage II and integrating with the **ASUS UGen300 (Hailo-10H NPU)**, we plan to enhance the system with the following advanced features:

*   **Holistic Posture Analysis:** Transition from calculating isolated limb angles to evaluating the complex spatial relationships between the limbs and the torso for a more robust pose assessment.
*   **Video-to-Video Imitation Learning:** Replace the static expert image with an expert reference video. By leveraging concepts from Imitation Learning and temporal alignment algorithms (e.g., Dynamic Time Warping), the system will score continuous movements rather than static frames.
*   **Temporal Dynamics & Optical Flow:** Correlate consecutive frames and apply optical flow analysis to observe nuanced micro-movements, such as limb tremors, stability, and movement velocity, providing professional-grade rehabilitation metrics.
*   **Absolute Privacy at the Edge:** Fully optimize and quantize the model for the Hailo NPU, ensuring zero network latency and guaranteeing that sensitive home video data never leaves the local device.