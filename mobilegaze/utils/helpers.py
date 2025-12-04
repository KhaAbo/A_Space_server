import cv2
import numpy as np
from models.resnet import resnet50


def get_model(arch, bins, inference_mode=False):
    """Return the model based on the specified architecture."""
    # Always return resnet50 as it is the only supported model for now
    model = resnet50(num_classes=bins)
    return model


def gaze_to_3d(gaze):
    yaw = gaze[0]   # Horizontal angle
    pitch = gaze[1]  # Vertical angle

    gaze_vector = np.zeros(3)
    gaze_vector[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_vector[1] = -np.sin(pitch)
    gaze_vector[2] = -np.cos(pitch) * np.cos(yaw)

    return gaze_vector


def draw_gaze(frame, bbox, pitch, yaw, thickness=3, color=(0, 0, 255)):
    """
    Draws solid 3D-style gaze direction arrows on a frame (Gaze360-inspired).
    
    Args:
        frame: Video frame to draw on
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max, ...]
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        thickness: Base arrow thickness (not used, kept for compatibility)
        color: Base arrow color (not used, kept for compatibility)
    """
    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Calculate center of the bounding box (face center)
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Handle grayscale frames by converting them to BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Solid 3D-style arrow rendering (Gaze360-inspired)
    # Scale factor based on face size - make arrows very prominent
    face_size = x_max - x_min
    length = face_size * 2.5  # Extra long arrows
    
    # Project gaze angles to 2D screen coordinates (ORIGINAL ACCURATE FORMULA)
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))
    
    # Calculate depth for 3D effects
    gaze_vector = gaze_to_3d([yaw, pitch])
    depth = -gaze_vector[2]  # positive = towards viewer, negative = away
    depth_factor = np.clip(depth, -1, 1)
    
    # Solid red/orange color - Gaze360 style
    arrow_color = (0, 60, 255)  # Bright red-orange
    
    # Much thicker arrows for solid look
    arrow_thickness = max(6, int(face_size * 0.08))
    
    # Calculate arrow endpoints
    start_point = (x_center, y_center)
    end_point = (x_center + dx, y_center + dy)
    
    # Draw SOLID arrow that looks like a 3D object
    # Thick black outline for definition
    cv2.arrowedLine(
        frame,
        (start_point[0] + 2, start_point[1] + 2),
        (end_point[0] + 2, end_point[1] + 2),
        (0, 0, 0),
        thickness=arrow_thickness + 4,
        line_type=cv2.LINE_AA,
        tipLength=0.3
    )
    
    # Main solid arrow
    cv2.arrowedLine(
        frame,
        start_point,
        end_point,
        arrow_color,
        thickness=arrow_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.3
    )
    
    # Add 3D lighting effect - bright edge on top/left
    highlight_offset = max(2, arrow_thickness // 4)
    quarter_point = (
        start_point[0] + dx // 4,
        start_point[1] + dy // 4
    )
    half_point = (
        start_point[0] + dx // 2,
        start_point[1] + dy // 2
    )
    
    # Perpendicular offset for highlight - calculate perpendicular vector
    if dx != 0 or dy != 0:
        # Get perpendicular vector: rotate 90 degrees
        perp_x = -dy
        perp_y = dx
        # Normalize
        norm = np.sqrt(perp_x**2 + perp_y**2) + 0.001
        perp_x = int(perp_x / norm * highlight_offset)
        perp_y = int(perp_y / norm * highlight_offset)
    else:
        perp_x = highlight_offset
        perp_y = 0
    
    # Bright highlight
    cv2.line(
        frame,
        (start_point[0] + perp_x, start_point[1] + perp_y),
        (half_point[0] + perp_x, half_point[1] + perp_y),
        (100, 150, 255),  # Lighter color for highlight
        thickness=max(2, arrow_thickness // 2),
        lineType=cv2.LINE_AA
    )
    
    # Small white shine
    cv2.line(
        frame,
        (start_point[0] + perp_x, start_point[1] + perp_y),
        (quarter_point[0] + perp_x, quarter_point[1] + perp_y),
        (200, 220, 255),
        thickness=max(1, arrow_thickness // 3),
        lineType=cv2.LINE_AA
    )
    
    # Large origin point - solid and prominent
    point_size = int(face_size * 0.1)
    # Black outline
    cv2.circle(frame, start_point, radius=point_size + 2, color=(0, 0, 0), thickness=-1)
    # Main color
    cv2.circle(frame, start_point, radius=point_size, color=arrow_color, thickness=-1)
    # White highlight
    cv2.circle(frame, start_point, radius=max(2, point_size // 2), color=(200, 220, 255), thickness=-1)



def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    width = x_max - x_min
    height = y_max - y_min

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_bbox_gaze(frame: np.ndarray, bbox, pitch, yaw):
    """
    Draw bounding box and gaze arrow on frame.
    
    Args:
        frame: Video frame
        bbox: Face bounding box
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
    """
    draw_bbox(frame, bbox)
    draw_gaze(frame, bbox, pitch, yaw)
