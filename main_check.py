import cv2
import math
import numpy as np
from ultralytics import YOLO

# 1. Calculate the angle formed by three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Helper function to get color based on score
def get_color_by_score(score):
    if score > 80:
        return (0, 255, 0)   # Green
    elif score > 50:
        return (0, 255, 255) # Yellow
    else:
        return (0, 0, 255)   # Red

# 2. Load the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# --- Step A: Analyze the expert's standard posture ---
expert_img = cv2.imread('expert.jpg') 
expert_results = model(expert_img, verbose=False)
expert_kp = expert_results[0].keypoints.xy[0].cpu().numpy()

# Calculate standard angles for all 4 limbs
exp_angles = {
    "left_arm": calculate_angle(expert_kp[5], expert_kp[7], expert_kp[9]),
    "right_arm": calculate_angle(expert_kp[6], expert_kp[8], expert_kp[10]),
    "left_leg": calculate_angle(expert_kp[11], expert_kp[13], expert_kp[15]),
    "right_leg": calculate_angle(expert_kp[12], expert_kp[14], expert_kp[16])
}

print("Expert's Standard Angles:", exp_angles)

# --- Step B: Analyze the user's video ---
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('demo_action_score_fullbody.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Define limb connections for drawing
limbs = {
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16)
}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            user_kp = result.keypoints.xy[0].cpu().numpy()
            
            scores = {}
            total_score = 0
            valid_limbs = 0

            # Calculate score for each limb
            for limb_name, (pt1, pt2, pt3) in limbs.items():
                # Check if all 3 points of the limb are detected
                if user_kp[pt1][0] != 0 and user_kp[pt2][0] != 0 and user_kp[pt3][0] != 0:
                    user_angle = calculate_angle(user_kp[pt1], user_kp[pt2], user_kp[pt3])
                    angle_diff = abs(exp_angles[limb_name] - user_angle)
                    
                    # 45 degrees difference = 0 points
                    score = max(0, int(100 - (angle_diff / 45.0 * 100)))
                    scores[limb_name] = score
                    total_score += score
                    valid_limbs += 1
                    
                    # Get color based on this specific limb's score
                    limb_color = get_color_by_score(score)
                    
                    # Draw the lines for this limb
                    p1, p2, p3 = tuple(map(int, user_kp[pt1])), tuple(map(int, user_kp[pt2])), tuple(map(int, user_kp[pt3]))
                    cv2.line(frame, p1, p2, limb_color, 4)
                    cv2.line(frame, p2, p3, limb_color, 4)
                    # Draw points
                    cv2.circle(frame, p1, 5, limb_color, -1)
                    cv2.circle(frame, p2, 5, limb_color, -1)
                    cv2.circle(frame, p3, 5, limb_color, -1)

            # Display overall score if at least one limb is detected
            if valid_limbs > 0:
                avg_score = int(total_score / valid_limbs)
                overall_color = get_color_by_score(avg_score)
                
                # Show Total Score at the top
                cv2.putText(frame, f"TOTAL SCORE: {avg_score}/100", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, overall_color, 3)
                
                # Show individual limb details (Optional, helps visually debug)
                y_offset = 110
                for limb_name, score in scores.items():
                    text_color = get_color_by_score(score)
                    cv2.putText(frame, f"{limb_name}: {score}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    y_offset += 30

    cv2.imshow("AI Coach PoC - Full Body", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("demo_action_score_fullbody.mp4 generated successfully.")