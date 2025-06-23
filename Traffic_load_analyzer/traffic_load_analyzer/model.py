import sys
import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sqlite3
from collections import deque
import os
import copy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFrame, QMessageBox, QDialog, QLineEdit,
                            QGridLayout, QGroupBox, QTextEdit, QScrollArea, QCheckBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QRect, QPoint, QMetaObject, Q_ARG
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QBrush
import torch
import mysql.connector

# YOLOv8 Integration
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 (ultralytics) imported successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available. Install with: pip install ultralytics")

class CongestionAnalysisThread(QThread):
    congestion_updated = pyqtSignal(int, str, float, str)  # signal_idx, level, score, color
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.signal_data = {}  # Store signal data for analysis
        self.analysis_interval = 2.0  # Analyze every 2 seconds
        self.last_analysis_time = time.time()
        
    def update_signal_data(self, signal_idx, vehicle_count, traffic_weight, area_size):
        """Update signal data for congestion analysis"""
        self.signal_data[signal_idx] = {
            'vehicle_count': vehicle_count,
            'traffic_weight': traffic_weight,
            'area_size': area_size,
            'timestamp': time.time()
        }
    
    def calculate_congestion_level(self, vehicle_count, traffic_weight, area_size=1000):
        """Calculate congestion level based on traffic density"""
        # Calculate traffic density (vehicles per unit area)
        if area_size > 0:
            density = vehicle_count / area_size
        else:
            density = vehicle_count / 1000  # Default area size
        
        # Calculate weighted density (considering vehicle types)
        weighted_density = traffic_weight / area_size if area_size > 0 else traffic_weight / 1000
        
        # For large areas, use a different approach - focus more on absolute vehicle count
        # and traffic weight rather than just density
        if area_size > 50000:  # Large detection areas
            # Use a combination of density and absolute counts
            congestion_score = (density * 1000000) + (vehicle_count * 0.5) + (traffic_weight * 0.3)
        else:
            # Original formula for smaller areas
            congestion_score = (density * 0.3 + weighted_density * 0.7) * 1000
        
        # More realistic thresholds for traffic congestion
        if congestion_score < 2:
            congestion_level = 'LOW'
            color = 'green'
        elif congestion_score < 8:
            congestion_level = 'MODERATE'
            color = 'orange'
        elif congestion_score < 20:
            congestion_level = 'HIGH'
            color = 'red'
        else:
            congestion_level = 'SEVERE'
            color = 'darkred'
        
        return congestion_level, congestion_score, color
    
    def run(self):
        """Main analysis loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Only analyze if enough time has passed
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    self.last_analysis_time = current_time
                    
                    # Analyze each signal's congestion
                    for signal_idx, data in self.signal_data.items():
                        # Only analyze if data is recent (within last 5 seconds)
                        if current_time - data['timestamp'] <= 5.0:
                            congestion_level, congestion_score, color = self.calculate_congestion_level(
                                data['vehicle_count'], 
                                data['traffic_weight'], 
                                data['area_size']
                            )
                            
                            # Emit the result to update UI
                            self.congestion_updated.emit(signal_idx, congestion_level, congestion_score, color)
                
                # Sleep to prevent excessive CPU usage
                self.msleep(500)  # Check every 500ms
                
            except Exception as e:
                print(f"Error in congestion analysis thread: {e}")
                self.msleep(1000)  # Wait longer on error
    
    def stop(self):
        """Stop the analysis thread"""
        self.running = False
        self.wait()

class VideoThread(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)  # Send raw frame for processing
    
    def __init__(self, video_path, signal_idx, detector, areas=None, get_signal_state_func=None, get_current_signal_func=None, frame_skip=2):
        super().__init__()
        self.video_path = video_path
        self.signal_idx = signal_idx
        self.running = True
        self.cap = None
        self.detector = detector
        self.area = areas[signal_idx].copy() if areas and signal_idx < len(areas) else None
        self.frame_counter = 0
        self.current_frame = None
        self.target_size = (320, 240)  # Fixed target size for display
        self.get_signal_state_func = get_signal_state_func
        self.get_current_signal_func = get_current_signal_func
        self.original_width = 1280
        self.original_height = 720
        self.frame_skip = frame_skip
    
    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source for Signal {self.signal_idx}")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1 or fps > 120:
                fps = 25
            frame_time = 1.0 / fps
            if self.area is not None:
                area_points = np.array(self.area, dtype=np.int32)
                print(f"Signal {self.signal_idx} using area points: {area_points}")
            else:
                print(f"No area points found for Signal {self.signal_idx}")
            while self.running:
                for _ in range(self.frame_skip - 1):
                    self.cap.grab()
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.current_frame = frame.copy()
                self.frame_ready.emit(self.signal_idx, frame.copy())  # Emit original frame
                self.msleep(int(frame_time * 1000))
        except Exception as e:
            print(f"Error in video thread {self.signal_idx}: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_video_frame(self, signal_idx, frame):
        """Process video frame and update display with bboxes and vehicle count"""
        try:
            if not self.areas or signal_idx >= len(self.areas):
                display_frame = cv2.resize(frame, (320, 240))
                vehicle_count = 0
                traffic_weight = 0
                vehicle_counts = {'auto': 0, 'bike': 0, 'bus': 0, 'car': 0, 'emergency_vehicles': 0, 'truck': 0}
            else:
                # Run detection on the original frame (not resized)
                vehicle_count, traffic_weight, processed_frame, vehicle_counts = \
                    self.detector.detect_vehicles_in_area(frame, self.areas[signal_idx])
                # Update signal data
                if signal_idx < len(self.signals):
                    signal = self.signals[signal_idx]
                    signal.vehicle_count = vehicle_count
                    signal.traffic_weight = traffic_weight
                    signal.vehicle_type_counts = vehicle_counts.copy()  # CRITICAL: Update vehicle type counts
                    signal.calculate_adaptive_green_time(vehicle_count, traffic_weight)
                    
                    # Send data to congestion analysis thread (non-blocking)
                    if signal_idx < len(self.areas) and self.areas[signal_idx]:
                        area_size = calculate_polygon_area(self.areas[signal_idx])
                    else:
                        area_size = 1000  # Default area size if areas not available
                    
                    # Update congestion thread with new data
                    if hasattr(self, 'congestion_thread'):
                        self.congestion_thread.update_signal_data(signal_idx, vehicle_count, traffic_weight, area_size)
                display_frame = cv2.resize(processed_frame, (320, 240))
            
            # Convert to QImage and update QLabel
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            if signal_idx < len(self.video_labels) and self.video_labels[signal_idx]:
                self.video_labels[signal_idx].setPixmap(pixmap)
            
            # Update vehicle class counts in the UI
            vehicle_info_frame = self.findChild(QFrame, f'vehicle_info_{signal_idx}')
            if vehicle_info_frame:
                for class_name, count in vehicle_counts.items():
                    label = vehicle_info_frame.findChild(QLabel, f'{class_name}_label')
                    if label:
                        emoji = label.text().split()[0]
                        label.setText(f"{emoji} {count}")
            
            # Update vehicle count and weight label if present
            count_label = getattr(self, f'count_label_{signal_idx}', None)
            if count_label:
                count_label.setText(f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}")
                
            # Log emergency vehicle detection for debugging
            if vehicle_counts.get('emergency_vehicles', 0) > 0:
                self.log_message(f"🚑 Emergency vehicle detected at Signal {chr(65+signal_idx)}: Count = {vehicle_counts['emergency_vehicles']}")
                
        except Exception as e:
            print(f"Error processing frame for signal {signal_idx}: {e}")

class EnhancedVehicleDetector:
    def __init__(self):
        self.model = None
        self.vehicle_classes = ['auto', 'bike', 'bus', 'car', 'emergency_vehicles', 'truck']
        self.vehicle_weights = {
            'auto': 0.8,
            'bike': 0.5,
            'bus': 2.0,
            'car': 1.0,
            'emergency_vehicles': 1.5,
            'truck': 2.5
        }
        self.new_vehicle_classes = {
            0: 'auto',
            1: 'bike',
            2: 'bus',
            3: 'car',
            4: 'emergency_vehicles',
            5: 'truck'
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Performance optimization settings
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0
        self.input_size = (640, 640)  # Standard YOLO input size
        self.last_detection_time = 0
        self.avg_inference_time = 0
        
        self.load_yolo_model()
    
    def load_yolo_model(self):
        try:
            if YOLO_AVAILABLE:
                # Try loading with specific version and device
                try:
                    self.model = YOLO(r"my_model.pt")
                    
                    self.model.to(self.device)
                    print("✅ YOLOv8 model loaded successfully")
                    return True
                except Exception as e:
                    print(f"Error loading YOLOv8 model: {e}")
                    return False
            else:
                print("YOLOv8 not available. Using simulated detection.")
                return False
        except Exception as e:
            print(f"Error in YOLO initialization: {e}")
            print("Falling back to simulated detection")
            self.model = None
            return False

    def detect_vehicles_in_area(self, frame, area_points, draw_area=True):
        if frame is None:
            return 0, 0, None, {'auto': 0, 'bike': 0, 'bus': 0, 'car': 0, 'emergency_vehicles': 0, 'truck': 0}, 0.0
        
        try:
            # Create mask for the defined area
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            area_points_np = np.array(area_points, dtype=np.int32)
            cv2.fillPoly(mask, [area_points_np], 255)
            
            # Create a copy of the frame for visualization
            processed_frame = frame.copy()
            
            # Draw detection area only if requested
            if draw_area:
                cv2.polylines(processed_frame, [area_points_np], True, (0, 255, 255), 3)
                cv2.putText(processed_frame, "Detection Area", 
                           (area_points_np[0][0], area_points_np[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Initialize vehicle counts
            vehicle_counts = {'auto': 0, 'bike': 0, 'bus': 0, 'car': 0, 'emergency_vehicles': 0, 'truck': 0}
            
            # Detect vehicles using YOLOv8
            if self.model is not None:
                try:
                    # Preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run inference with optimized parameters
                    results = self.model(
                        frame_rgb,
                        conf=0.25,
                        iou=0.45,
                        max_det=50,
                        classes=[0, 1, 2, 3, 4, 5],
                        verbose=False
                    )
                    
                    vehicle_count = 0
                    traffic_weight = 0
                    detected_centers = []
                    confidences = []
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)  # center point
                                
                                # Check if center point is in the detection area
                                if self.point_in_polygon((x, y), area_points):
                                    # Check if we already detected a vehicle at this location
                                    too_close = False
                                    for cx, cy in detected_centers:
                                        if abs(x - cx) < 30 and abs(y - cy) < 30:  # 30 pixel threshold
                                            too_close = True
                                            break
                                    
                                    if not too_close:
                                        detected_centers.append((x, y))
                                        class_id = int(box.cls[0].cpu().numpy())
                                        confidence = float(box.conf[0].cpu().numpy())
                                        confidences.append(confidence)
                                        
                                        if class_id in self.new_vehicle_classes:
                                            vehicle_count += 1
                                            class_name = self.new_vehicle_classes[class_id]
                                            traffic_weight += self.vehicle_weights.get(class_name, 1.0)
                                            vehicle_counts[class_name] += 1
                                            
                                            # Draw bounding box with different colors based on vehicle type
                                            color = {
                                                'auto': (128, 128, 128),
                                                'bike': (0, 255, 255),
                                                'bus': (255, 165, 0),
                                                'car': (0, 255, 0),
                                                'emergency_vehicles': (255, 0, 0),
                                                'truck': (255, 0, 0)
                                            }.get(class_name, (0, 255, 0))
                                            
                                            # Draw bounding box and label
                                            cv2.rectangle(processed_frame, 
                                                        (int(x1), int(y1)), 
                                                        (int(x2), int(y2)), 
                                                        color, 2)
                                            
                                            # Add background to text for better visibility
                                            label = f"{class_name}: {confidence:.2f}"
                                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                            cv2.rectangle(processed_frame,
                                                        (int(x1), int(y1 - 20)),
                                                        (int(x1 + label_w), int(y1)),
                                                        color, -1)
                                            cv2.putText(processed_frame, label,
                                                      (int(x1), int(y1 - 5)),
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.5, (255, 255, 255), 2)
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # Add detection info with background
                    info_bg_color = (0, 0, 0)
                    info_text_color = (255, 255, 255)
                    
                    # Vehicle count background
                    cv2.rectangle(processed_frame, (10, 10), (150, 35), info_bg_color, -1)
                    cv2.putText(processed_frame, f"Vehicles: {vehicle_count}", 
                               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
                    
                    # Traffic weight background
                    cv2.rectangle(processed_frame, (10, 40), (200, 65), info_bg_color, -1)
                    cv2.putText(processed_frame, f"Traffic Weight: {traffic_weight:.1f}", 
                               (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
                    
                    return vehicle_count, traffic_weight, processed_frame, vehicle_counts, avg_confidence
                    
                except Exception as e:
                    print(f"YOLO detection error: {str(e)}")
                    return self.simulate_detection(frame, mask)
            else:
                return self.simulate_detection(frame, mask)
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return 0, 0, frame, {'auto': 0, 'bike': 0, 'bus': 0, 'car': 0, 'emergency_vehicles': 0, 'truck': 0}, 0.0

    def simulate_detection(self, frame, mask):
        """Simulate vehicle detection when YOLO is not available"""
        vehicle_type_counts = {
            'auto': 0,
            'bike': 0,
            'bus': 0,
            'car': 0,
            'emergency_vehicles': 0,
            'truck': 0
        }
        
        # Create a copy of frame for visualization
        processed_frame = frame.copy()
        
        # Simulate random detections
        height, width = frame.shape[:2]
        num_vehicles = np.random.randint(1, 12)
        total_weight = 0
        confidences = []
        
        for _ in range(num_vehicles):
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 60)
            w = np.random.randint(80, 120)
            h = np.random.randint(40, 80)
            class_name = np.random.choice(self.vehicle_classes)
            confidence = np.random.uniform(0.5, 0.95)
            confidences.append(confidence)
            
            # Update vehicle type count
            vehicle_type_counts[class_name] += 1
            
            # Add to total weight
            total_weight += self.vehicle_weights.get(class_name, 1.0)
            
            # Draw bounding box with different colors based on vehicle type
            color = {
                'auto': (128, 128, 128),      # Gray
                'bike': (0, 255, 255),        # Yellow
                'bus': (255, 165, 0),         # Orange
                'car': (0, 255, 0),           # Green
                'emergency_vehicles': (255, 0, 0),  # Red
                'truck': (255, 0, 0)          # Red
            }.get(class_name, (0, 255, 0))
            
            # Draw rectangle and label
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(processed_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return num_vehicles, total_weight, processed_frame, vehicle_type_counts, avg_confidence

    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

class EnhancedTrafficSignal:
    def __init__(self, signal_id):
        self.signal_id = signal_id
        self.min_green_time = 10
        self.max_green_time = 45
        self.default_green_time = 15
        self.yellow_time = 3
        self.all_red_time = 2
        self.current_state = 'RED'
        self.remaining_time = 0
        self.vehicle_history = deque(maxlen=10)
        self.priority_mode = False
        self.last_detection_time = 0
        self.vehicle_count = 0
        self.traffic_weight = 0
        self.calculated_green_time = self.default_green_time
        self.is_active = False
        self.pending_green_time = 0
        self.last_update_time = time.time()
        self.has_emergency_vehicle = False
        self.emergency_vehicle_detected_time = 0
        self.emergency_vehicle_wait_time = 0
        # Initialize vehicle type counts
        self.vehicle_type_counts = {
            'auto': 0,
            'bike': 0,
            'bus': 0,
            'car': 0,
            'emergency_vehicles': 0,
            'truck': 0
        }
        # Initialize detection scores
        self.detection_scores = []
        # Initialize queue metrics
        self.queue_length = 0
        self.avg_wait_time = 0
        # Congestion tracking
        self.congestion_level = 'LOW'
        self.congestion_score = 0.0
        self.congestion_history = deque(maxlen=20)
        self.avg_confidence = 0.0

    def calculate_adaptive_green_time(self, vehicle_count, traffic_weight, time_of_day=None):
        """Calculate optimal green time based on traffic conditions"""
        self.vehicle_count = vehicle_count
        self.traffic_weight = traffic_weight
        
        # Base calculation
        if traffic_weight == 0:
            calculated_time = self.min_green_time
        else:
            # Scale green time based on traffic weight
            density_factor = min(traffic_weight / 5.0, 3.0)  # Max 3x multiplier
            calculated_time = self.min_green_time + (density_factor * 10)
        
        # Apply time-of-day factor
        if time_of_day:
            current_hour = time_of_day.hour
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hours
                calculated_time *= 1.2
            elif 22 <= current_hour or current_hour <= 6:  # Night time
                calculated_time *= 0.8
        
        # Consider historical data
        if len(self.vehicle_history) > 3:
            avg_traffic = sum(self.vehicle_history) / len(self.vehicle_history)
            if traffic_weight > avg_traffic * 1.5:  # Heavy traffic spike
                calculated_time *= 1.3
        
        # Store history and apply constraints
        self.vehicle_history.append(traffic_weight)
        self.calculated_green_time = max(self.min_green_time, 
                                       min(int(calculated_time), self.max_green_time))
        
        return self.calculated_green_time

    def calculate_congestion_level(self, vehicle_count, traffic_weight, area_size=1000):
        """Calculate congestion level based on traffic density"""
        # Calculate traffic density (vehicles per unit area)
        if area_size > 0:
            density = vehicle_count / area_size
        else:
            density = vehicle_count / 1000  # Default area size
        
        # Calculate weighted density (considering vehicle types)
        weighted_density = traffic_weight / area_size if area_size > 0 else traffic_weight / 1000
        
        # For large areas, use a different approach - focus more on absolute vehicle count
        # and traffic weight rather than just density
        if area_size > 50000:  # Large detection areas
            # Use a combination of density and absolute counts
            congestion_score = (density * 1000000) + (vehicle_count * 0.5) + (traffic_weight * 0.3)
        else:
            # Original formula for smaller areas
            congestion_score = (density * 0.3 + weighted_density * 0.7) * 1000
        
        # More realistic thresholds for traffic congestion
        if congestion_score < 2:
            congestion_level = 'LOW'
            color = 'green'
        elif congestion_score < 8:
            congestion_level = 'MODERATE'
            color = 'orange'
        elif congestion_score < 20:
            congestion_level = 'HIGH'
            color = 'red'
        else:
            congestion_level = 'SEVERE'
            color = 'darkred'
        
        # Update congestion history
        self.congestion_history.append(congestion_score)
        self.congestion_score = congestion_score
        self.congestion_level = congestion_level
        
        return congestion_level, congestion_score, color

    def get_congestion_trend(self):
        """Get congestion trend over time"""
        if len(self.congestion_history) < 2:
            return 'STABLE'
        
        recent_avg = sum(list(self.congestion_history)[-5:]) / min(5, len(self.congestion_history))
        older_avg = sum(list(self.congestion_history)[:-5]) / max(1, len(self.congestion_history) - 5)
        
        if recent_avg > older_avg * 1.2:
            return 'INCREASING'
        elif recent_avg < older_avg * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'

class TrafficDatabase:
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root',
            'password': 'Pokeash@1234',
            'database': 'traffic_load_analyzer'
        }
        self.last_update_time = time.time()
        self.update_interval = 120  # Update every 120 seconds
        self.init_database()
        self.create_powerbi_view()
    
    def create_powerbi_view(self):
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create a view for Power BI that combines all relevant data
            cursor.execute('''
                CREATE OR REPLACE VIEW traffic_analytics_view AS
                SELECT 
                    t.signal_id,
                    t.timestamp,
                    t.vehicle_count,
                    t.traffic_weight,
                    t.green_time,
                    t.vehicle_type_counts,
                    s.min_green_time,
                    s.max_green_time,
                    s.current_state,
                    s.vehicle_history
                FROM traffic_data t
                JOIN traffic_signals s ON t.signal_id = s.signal_id
                ORDER BY t.timestamp DESC
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error creating Power BI view: {e}")
    
    def init_database(self):
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create signal_timing_logs table for logging timing changes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_timing_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    signal_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    green_time INT,
                    yellow_time INT,
                    red_time INT,
                    reason VARCHAR(255) DEFAULT NULL
                )
            ''')
            
            # Create traffic_data table with specific vehicle type columns (remove FK to traffic_signals)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    signal_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vehicle_count INT,
                    traffic_weight FLOAT,
                    green_time INT,
                    auto_count INT DEFAULT 0,
                    bike_count INT DEFAULT 0,
                    bus_count INT DEFAULT 0,
                    car_count INT DEFAULT 0,
                    emergency_vehicles_count INT DEFAULT 0,
                    truck_count INT DEFAULT 0,
                    vehicle_type_counts JSON
                )
            ''')
            
            # Add missing columns if they don't exist (for existing tables)
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN auto_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding auto_count: {e}")
            
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN bike_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding bike_count: {e}")
            
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN bus_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding bus_count: {e}")
            
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN car_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding car_count: {e}")
            
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN emergency_vehicles_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding emergency_vehicles_count: {e}")
            
            try:
                cursor.execute('ALTER TABLE traffic_data ADD COLUMN truck_count INT DEFAULT 0')
            except mysql.connector.Error as e:
                if e.errno != 1060:  # Not duplicate column name
                    print(f"Error adding truck_count: {e}")
            
            # Create detection_areas table (remove FK to traffic_signals)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_areas (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    signal_id INT,
                    area_points JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create congestion_events table (no FK to traffic_signals)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS congestion_events (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    signal_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    severity VARCHAR(10),  -- 'LOW', 'MEDIUM', 'HIGH'
                    cause VARCHAR(255),
                    resolution_time INT
                )
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            print("Database initialization completed successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def log_traffic_data(self, signal_id, vehicle_count, traffic_weight, green_time, vehicle_type_counts):
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Ensure vehicle_type_counts has all required vehicle types
            default_counts = {
                'auto': 0,
                'bike': 0,
                'bus': 0,
                'car': 0,
                'emergency_vehicles': 0,
                'truck': 0
            }
            # Update default counts with actual counts
            default_counts.update(vehicle_type_counts)

                        
            # Log the data being sent for verification
            print(f"\n=== Database Log Entry for Signal {signal_id} ===")
            print(f"Timestamp: {datetime.now()}")
            print(f"Signal ID: {signal_id}")
            print(f"Vehicle Count: {vehicle_count}")
            print(f"Traffic Weight: {traffic_weight}")
            print(f"Green Time: {green_time}")
            print(f"Vehicle Type Counts:")
            for vehicle_type, count in default_counts.items():
                print(f"  {vehicle_type}: {count}")
            print(f"JSON Data: {json.dumps(default_counts)}")

            
            # Insert traffic data with individual vehicle type columns
            cursor.execute('''
                INSERT INTO traffic_data 
                (signal_id, vehicle_count, traffic_weight, green_time, 
                auto_count, bike_count, bus_count, car_count, emergency_vehicles_count, truck_count,
                vehicle_type_counts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                signal_id, vehicle_count, traffic_weight, green_time,
                default_counts['auto'], default_counts['bike'], default_counts['bus'],
                default_counts['car'], default_counts['emergency_vehicles'], default_counts['truck'],
                json.dumps(default_counts)
            ))
            
            # Update signal state if needed
            cursor.execute('''
                UPDATE traffic_signals 
                SET vehicle_history = JSON_ARRAY_APPEND(
                    COALESCE(vehicle_history, JSON_ARRAY()),
                    '$',
                    %s
                )
                WHERE signal_id = %s
            ''', (traffic_weight, signal_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error logging traffic data: {e}")
            return False
    
    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def verify_database_data(self):
        """Verify that all data is being properly transferred to the database"""
        pass

    def test_data_insertion(self):
        """Test data insertion to verify all columns work"""
        pass

    def log_signal_timing_change(self, signal_id, green_time, yellow_time, red_time, reason=None):
        """Log a signal timing change to the signal_timing_logs table."""
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_timing_logs (signal_id, green_time, yellow_time, red_time, reason)
                VALUES (%s, %s, %s, %s, %s)
            ''', (signal_id, green_time, yellow_time, red_time, reason))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error logging signal timing change: {e}")

    def log_congestion_event(self, signal_id, severity, cause, resolution_time=None):
        """Log a congestion spike or anomaly event to the congestion_events table."""
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO congestion_events (signal_id, severity, cause, resolution_time)
                VALUES (%s, %s, %s, %s)
            ''', (signal_id, severity, cause, resolution_time))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error logging congestion event: {e}")
    
    def verify_data_before_logging(self, signal_id, vehicle_count, traffic_weight, green_time, vehicle_type_counts):
        """Verify data integrity before sending to database"""
        print(f"\n=== Data Verification for Signal {signal_id} ===")
        
        # Check data types
        issues = []
        
        if not isinstance(signal_id, int) or signal_id < 0 or signal_id > 3:
            issues.append(f"Invalid signal_id: {signal_id} (should be 0-3)")
        
        if not isinstance(vehicle_count, (int, float)) or vehicle_count < 0:
            issues.append(f"Invalid vehicle_count: {vehicle_count} (should be >= 0)")
        
        if not isinstance(traffic_weight, (int, float)) or traffic_weight < 0:
            issues.append(f"Invalid traffic_weight: {traffic_weight} (should be >= 0)")
        
        if not isinstance(green_time, (int, float)) or green_time < 0:
            issues.append(f"Invalid green_time: {green_time} (should be >= 0)")
        
        if not isinstance(vehicle_type_counts, dict):
            issues.append(f"Invalid vehicle_type_counts: {vehicle_type_counts} (should be dict)")
        else:
            expected_keys = {'auto', 'bike', 'bus', 'car', 'emergency_vehicles', 'truck'}
            actual_keys = set(vehicle_type_counts.keys())
            missing_keys = expected_keys - actual_keys
            extra_keys = actual_keys - expected_keys
            
            if missing_keys:
                issues.append(f"Missing vehicle types: {missing_keys}")
            if extra_keys:
                issues.append(f"Unexpected vehicle types: {extra_keys}")
            
            # Check individual counts
            for vehicle_type, count in vehicle_type_counts.items():
                if not isinstance(count, (int, float)) or count < 0:
                    issues.append(f"Invalid count for {vehicle_type}: {count}")
        
        # Check if vehicle_count matches sum of vehicle_type_counts
        if isinstance(vehicle_type_counts, dict):
            calculated_total = sum(vehicle_type_counts.values())
            if abs(calculated_total - vehicle_count) > 0.1:  # Allow small floating point differences
                issues.append(f"Vehicle count mismatch: total={vehicle_count}, sum of types={calculated_total}")
        
        # Check if traffic_weight is reasonable
        if isinstance(vehicle_type_counts, dict):
            max_possible_weight = sum(vehicle_type_counts.get(vehicle_type, 0) * {
                'auto': 0.8, 'bike': 0.5, 'bus': 2.0, 'car': 1.0, 'emergency_vehicles': 1.5, 'truck': 2.5
            }.get(vehicle_type, 1.0) for vehicle_type in vehicle_type_counts)
            
            if traffic_weight > max_possible_weight * 1.5:  # Allow some tolerance
                issues.append(f"Traffic weight seems too high: {traffic_weight} vs max possible {max_possible_weight}")
        
        if issues:
            print("❌ Data verification failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Data verification passed")
            return True

    def comprehensive_database_test(self):
        """Comprehensive test of all database functionality"""
        pass

def calculate_polygon_area(points):
    """Calculate the area of a polygon using the shoelace formula"""
    if len(points) < 3:
        return 0
    
    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2

class EnhancedTrafficManagementSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Smart Traffic Management System with YOLOv8")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.detector = EnhancedVehicleDetector()
        self.signals = [EnhancedTrafficSignal(i) for i in range(4)]
        self.database = TrafficDatabase()
        
        # Initialize congestion analysis thread
        self.congestion_thread = CongestionAnalysisThread()
        self.congestion_thread.congestion_updated.connect(self.update_congestion_display)
        
        # Initialize analytics thread (will be created when needed)
        self.analytics_thread = None
        
        # Thread safety
        self.signal_lock = threading.Lock()
        self.last_update_time = time.time()
        
        # Video threads
        self.video_threads = [None] * 4
        self.video_labels = [None] * 4
        
        # Configuration
        self.areas_file = "areas.json"
        self.config_file = "config.json"
        
        # Video sources - one for each signal
        self.video_sources = [
            os.path.abspath(r"video.mp4"),    # Signal A
            os.path.abspath(r"video2.mp4"),   # Signal B
            os.path.abspath(r"video3.mp4"),   # Signal C
            os.path.abspath(r"video4.mp4")    # Signal D
        ]
        
        # Verify video files exist
        for i, video_path in enumerate(self.video_sources):
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found for Signal {chr(65+i)}: {video_path}")
        
        self.areas = []
        self.is_running = False
        self.emergency_mode = False
        self.current_signal = 0
        self.cycle_start_time = time.time()
        # Track interrupted signal for emergency mode
        self.interrupted_signal_idx = None
        
        # Initialize analytics data
        self.analytics_data = {
            'timestamps': deque(maxlen=100),
            'vehicle_counts': [deque(maxlen=100) for _ in range(4)],
            'green_times': [deque(maxlen=100) for _ in range(4)],
            'total_vehicles_processed': 0,
            'signal_usage': [0] * 4
        }
        
        # Initialize UI labels
        self.cycle_time_label = QLabel("Cycle Time: 0s")
        self.current_signal_label = QLabel("Active Signal: A")
        
        # Thread-safe logging
        self.log_mutex = threading.Lock()
        
        # Initialize UI
        self.setup_ui()
        
        # Load configuration
        self.load_config()
        
        # Initialize timers but don't start them yet
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system)
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        
        # Start UI timer only (not system timer)
        self.ui_timer.start(150)  # Update UI every 150ms

        # Load areas if they exist
        self.load_areas(show_message=False)

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Header
        header = QLabel("🚦 Enhanced Smart Traffic Management System")
        header.setStyleSheet("""
            QLabel {
                background-color: #34495e;
                color: white;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(header)
        
        # Control Panel
        control_panel = QGroupBox("System Controls")
        control_layout = QHBoxLayout()
        
        buttons = [
            ("🎯 New Areas", self.define_new_areas, "#3498db"),
            ("📁 Load Areas", self.load_areas, "#2ecc71"),
            ("▶️ Start System", self.start_system, "#27ae60"),
            ("⏹️ Stop System", self.stop_system, "#e74c3c"),
            ("🚨 Emergency Mode", self.toggle_emergency_mode, "#f39c12"),
            ("📊 Analytics", self.show_analytics, "#9b59b6"),
            ("⚙️ Settings", self.show_settings, "#34495e")
        ]
        
        for text, callback, color in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
            """)
            btn.clicked.connect(callback)
            control_layout.addWidget(btn)
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Video Display Grid
        video_grid = QGridLayout()
        for i in range(2):
            for j in range(2):
                signal_idx = i * 2 + j
                signal_frame = QGroupBox(f"Signal {chr(65 + signal_idx)}")
                signal_layout = QVBoxLayout()
                
                # Video display
                video_label = QLabel()
                video_label.setMinimumSize(320, 240)
                video_label.setStyleSheet("background-color: black;")
                signal_layout.addWidget(video_label)
                
                # Add vehicle class information display
                vehicle_info_frame = QFrame()
                vehicle_info_frame.setObjectName(f"vehicle_info_{signal_idx}")
                vehicle_info_frame.setStyleSheet("""
                    QFrame {
                        background-color: #2c3e50;
                        border-radius: 4px;
                        padding: 5px;
                    }
                """)
                vehicle_info_layout = QHBoxLayout()
                vehicle_info_layout.setSpacing(5)
                vehicle_info_layout.setContentsMargins(5, 5, 5, 5)
                vehicle_info_layout.addStretch()
                vehicle_classes = {
                    'auto': ('🛺', '#95a5a6'),
                    'bike': ('🏍️', '#3498db'),
                    'bus': ('🚌', '#f1c40f'),
                    'car': ('🚗', '#2ecc71'),
                    'emergency_vehicles': ('🚑', '#e74c3c'),
                    'truck': ('🚚', '#e74c3c')
                }
                for class_name, (emoji, color) in vehicle_classes.items():
                    label = QLabel(f"{emoji} 0")
                    label.setObjectName(f"{class_name}_label")
                    label.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: white;
                            border-radius: 3px;
                            padding: 2px 18px;
                            font-weight: bold;
                            min-width: 60px;
                            text-align: center;
                        }}
                    """)
                    vehicle_info_layout.addWidget(label)
                vehicle_info_layout.addStretch()
                vehicle_info_frame.setLayout(vehicle_info_layout)
                vehicle_info_container = QWidget()
                vehicle_info_container.setFixedWidth(320)  # Match video width
                vehicle_info_container.setStyleSheet("background: transparent;")
                vehicle_info_layout = QHBoxLayout()
                vehicle_info_layout.setSpacing(5)
                vehicle_info_layout.setContentsMargins(0, 0, 0, 0)
                vehicle_info_layout.setAlignment(Qt.AlignCenter)
                vehicle_info_frame = QFrame()
                vehicle_info_frame.setObjectName(f"vehicle_info_{signal_idx}")
                vehicle_info_frame.setStyleSheet("""
                    QFrame {
                        background-color: #2c3e50;
                        border-radius: 4px;
                        padding: 5px;
                    }
                """)
                vehicle_bar_layout = QHBoxLayout()
                vehicle_bar_layout.setSpacing(5)
                vehicle_bar_layout.setContentsMargins(5, 5, 5, 5)
                vehicle_bar_layout.setAlignment(Qt.AlignCenter)
                vehicle_classes = {
                    'auto': ('🛺', '#95a5a6'),
                    'bike': ('🏍️', '#3498db'),
                    'bus': ('🚌', '#f1c40f'),
                    'car': ('🚗', '#2ecc71'),
                    'emergency_vehicles': ('🚑', '#e74c3c'),
                    'truck': ('🚚', '#e74c3c')
                }
                for class_name, (emoji, color) in vehicle_classes.items():
                    label = QLabel(f"{emoji} 0")
                    label.setObjectName(f"{class_name}_label")
                    label.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: white;
                            border-radius: 3px;
                            padding: 2px 18px;
                            font-weight: bold;
                            min-width: 60px;
                            text-align: center;
                        }}
                    """)
                    vehicle_bar_layout.addWidget(label)
                vehicle_info_frame.setLayout(vehicle_bar_layout)
                vehicle_info_layout.addWidget(vehicle_info_frame)
                vehicle_info_container.setLayout(vehicle_info_layout)
                # Center the bar under the video
                signal_layout.addWidget(vehicle_info_container, alignment=Qt.AlignHCenter)
                # --- End horizontal bar ---
                
                # Status indicators
                status_label = QLabel("🔴 RED")
                status_label.setStyleSheet("color: red; font-weight: bold;")
                signal_layout.addWidget(status_label)
                
                time_label = QLabel("Time: 0s")
                signal_layout.addWidget(time_label)
                
                count_label = QLabel("Vehicles: 0 | Weight: 0.0")
                signal_layout.addWidget(count_label)
                
                # Add congestion level display
                congestion_label = QLabel("Congestion: LOW")
                congestion_label.setStyleSheet("color: green; font-weight: bold; font-size: 10px;")
                signal_layout.addWidget(congestion_label)

                #ADD Efficiency label
                efficiency_label = QLabel("Efficiency: N/A")
                efficiency_label.setStyleSheet("color: #2980b9; font-size: 10px;")
                signal_layout.addWidget(efficiency_label)
                
                signal_frame.setLayout(signal_layout)
                video_grid.addWidget(signal_frame, i, j)
                
                # Store references
                self.video_labels[signal_idx] = video_label
                setattr(self, f'status_label_{signal_idx}', status_label)
                setattr(self, f'time_label_{signal_idx}', time_label)
                setattr(self, f'count_label_{signal_idx}', count_label)
                setattr(self, f'congestion_label_{signal_idx}', congestion_label)
                setattr(self, f'efficiency_label_{signal_idx}', efficiency_label)
        
        main_layout.addLayout(video_grid)
        
        # System Status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        
        self.total_vehicles_label = QLabel("Total Vehicles: 0")
        self.cycle_time_label = QLabel("Cycle Time: 0s")
        self.current_signal_label = QLabel("Active Signal: A")
        
        status_layout.addWidget(self.total_vehicles_label)
        status_layout.addWidget(self.cycle_time_label)
        status_layout.addWidget(self.current_signal_label)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Status Log
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
    def log_message(self, message):
        """Thread-safe logging"""
        try:
            with self.log_mutex:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}\n"
                # Use invokeMethod to update UI from any thread
                QMetaObject.invokeMethod(self.log_text, 
                                       "append",
                                       Qt.QueuedConnection,
                                       Q_ARG(str, log_entry))
                
                # Keep only last 100 messages
                text = self.log_text.toPlainText()
                lines = text.split('\n')
                if len(lines) > 100:
                    QMetaObject.invokeMethod(self.log_text,
                                           "setPlainText",
                                           Qt.QueuedConnection,
                                           Q_ARG(str, '\n'.join(lines[-100:])))
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def start_system(self):
        # Reload areas before starting
        self.log_message("Starting system - loading areas...")
        if not self.load_areas():
            QMessageBox.warning(self, "Warning", "Could not load detection areas. Please check areas.json!")
            return
            
        if not self.areas:
            QMessageBox.warning(self, "Warning", "Please define detection areas first!")
            return

        if not self.is_running:
            self.is_running = True
            self.cycle_start_time = time.time()
            self.last_update_time = time.time()  # Reset last update time
            self.log_message("🚀 Traffic management system started")
            self.log_message(f"Using areas: {self.areas}")
            
            # Start video threads with fresh areas
            for i, video_path in enumerate(self.video_sources):
                if os.path.exists(video_path):
                    # Log the area being used for this signal
                    self.log_message(f"Setting up Signal {chr(65 + i)} with area: {self.areas[i]}")
                    
                    # Create a new thread with just this signal's area
                    self.video_threads[i] = VideoThread(
                        video_path, 
                        i, 
                        self.detector,
                        self.areas,  # Pass all areas
                        get_signal_state_func=lambda idx: self.signals[idx].current_state,
                        get_current_signal_func=lambda: self.current_signal
                    )
                    self.video_threads[i].frame_ready.connect(self.process_video_frame)
                    self.video_threads[i].start()
                    self.log_message(f"✅ Video thread started for Signal {chr(65 + i)}")
            
            # Start congestion analysis thread
            self.congestion_thread.start()
            self.log_message("✅ Congestion analysis thread started")
            
            # Set all signals to RED initially
            with self.signal_lock:
                for signal in self.signals:
                    signal.current_state = 'RED'
                    signal.remaining_time = 0
            
            self.current_signal = 0
            # Immediately run YOLO for signal A and set to GREEN
            self.run_initial_detection_for_signal(0)
            
            # Start system update timer (10 times per second)
            self.update_timer.start(100)
            
            QMessageBox.information(self, "System Started", 
                                  "Traffic management system is now running!")

    def run_initial_detection_for_signal(self, signal_idx):
        # Run YOLO for signal_idx, set to GREEN for calculated time
        def detection():
            time.sleep(0.2)
            frame = None
            if self.video_threads[signal_idx] and self.video_threads[signal_idx].isRunning():
                frame = self.video_threads[signal_idx].current_frame
            if frame is not None and signal_idx < len(self.areas):
                vehicle_count, traffic_weight, _, vehicle_type_counts, avg_confidence = self.detector.detect_vehicles_in_area(frame, self.areas[signal_idx], draw_area=False)
                with self.signal_lock:
                    signal = self.signals[signal_idx]
                    green_time = signal.calculate_adaptive_green_time(vehicle_count, traffic_weight, datetime.now())
                    signal.vehicle_count = vehicle_count
                    signal.traffic_weight = traffic_weight
                    signal.vehicle_type_counts = vehicle_type_counts.copy()  # Store vehicle type counts
                    signal.current_state = 'GREEN'
                    signal.remaining_time = green_time
                    signal.avg_confidence = avg_confidence
                    self.log_message(f"🟢 Signal {chr(65 + signal_idx)} → GREEN for {green_time}s (Vehicles: {vehicle_count}, Weight: {traffic_weight:.1f})")
                    # Log timing change
                    self.database.log_signal_timing_change(
                        signal_id=signal_idx,
                        green_time=green_time,
                        yellow_time=signal.yellow_time,
                        red_time=signal.all_red_time,
                        reason='automatic'
                    )
            else:
                # Fallback
                with self.signal_lock:
                    signal = self.signals[signal_idx]
                    signal.current_state = 'GREEN'
                    signal.remaining_time = signal.default_green_time
                    self.log_message(f"🟢 Signal {chr(65 + signal_idx)} → GREEN for {signal.default_green_time}s (default)")
                    # Log timing change
                    self.database.log_signal_timing_change(
                        signal_id=signal_idx,
                        green_time=signal.default_green_time,
                        yellow_time=signal.yellow_time,
                        red_time=signal.all_red_time,
                        reason='default'
                    )
        threading.Thread(target=detection, daemon=True).start()

    def handle_signal_transitions(self, elapsed):
        """Handle signal state transitions"""
        with self.signal_lock:
            active_signal = self.signals[self.current_signal]
            # Update remaining time
            if active_signal.remaining_time > 0:
                active_signal.remaining_time = max(0, active_signal.remaining_time - elapsed)
                self.log_message(f"Timer update: Signal {chr(65 + self.current_signal)} - Remaining: {active_signal.remaining_time:.1f}s, Elapsed: {elapsed:.3f}s")

            # Emergency mode logic
            if self.emergency_mode:
                emergency_detected = False
                emergency_signal_idx = None
                # Check all signals for emergency vehicles
                for i, signal in enumerate(self.signals):
                    if signal.vehicle_type_counts.get('emergency_vehicles', 0) > 0:
                        emergency_detected = True
                        emergency_signal_idx = i
                        break
                if emergency_detected:
                    # Emergency in a different signal
                    if active_signal.current_state == 'GREEN' and self.current_signal != emergency_signal_idx:
                        # Store interrupted info
                        if self.interrupted_signal_idx is None and active_signal.remaining_time > 8.0:
                            self.interrupted_signal_idx = self.current_signal
                            self.interrupted_signal_remaining = active_signal.remaining_time
                        else:
                            self.interrupted_signal_idx = None
                            self.interrupted_signal_remaining = None
                        # Force current signal to yellow for 3 seconds
                        active_signal.current_state = 'YELLOW'
                        active_signal.remaining_time = 3.0
                        # Set a flag to indicate this yellow is due to emergency
                        self._emergency_force_red = True
                        self.log_message(f"🚑 Emergency vehicle detected at Signal {chr(65+emergency_signal_idx)} - Forcing Signal {chr(65+self.current_signal)} to YELLOW for 3s")
                        return
                    # If current signal is YELLOW due to emergency, after yellow expires, set to RED
                    elif active_signal.current_state == 'YELLOW' and self.current_signal != emergency_signal_idx:
                        if hasattr(self, '_emergency_force_red') and self._emergency_force_red and active_signal.remaining_time <= 0:
                            active_signal.current_state = 'RED'
                            active_signal.remaining_time = 0
                            del self._emergency_force_red
                            self.log_message(f"Signal {chr(65+self.current_signal)}: YELLOW → RED due to emergency")
                            return
                    # Emergency in the same signal
                    elif active_signal.current_state == 'GREEN' and self.current_signal == emergency_signal_idx:
                        vehicle_count = active_signal.vehicle_count
                        extended_time = min(vehicle_count * 2 + 10, active_signal.max_green_time)
                        active_signal.remaining_time = max(active_signal.remaining_time, extended_time)
                        self.log_message(f"🚑 Emergency vehicle at current Signal {chr(65+self.current_signal)} - Extended green time to {active_signal.remaining_time:.1f}s")
                        return
                # If we are currently serving the emergency signal, after its green/yellow/red, resume from interrupted
                if self.interrupted_signal_idx is not None:
                    # If emergency signal just finished its green and is now yellow
                    if active_signal.current_state == 'YELLOW' and self.current_signal == emergency_signal_idx:
                        pass
                    elif active_signal.current_state == 'RED' and self.current_signal == emergency_signal_idx:
                        if elapsed >= active_signal.remaining_time:
                            resume_idx = self.interrupted_signal_idx
                            resume_time = self.interrupted_signal_remaining
                            self.interrupted_signal_idx = None
                            self.interrupted_signal_remaining = None
                            if resume_time is not None and resume_time > 8.0:
                                next_signal = self.signals[resume_idx]
                                next_signal.current_state = 'GREEN'
                                next_signal.remaining_time = resume_time
                                self.current_signal = resume_idx
                                self.log_message(f"Resuming from interrupted Signal {chr(65+resume_idx)} after emergency (with {resume_time:.1f}s left).")
                                self.database.log_signal_timing_change(
                                    signal_id=resume_idx,
                                    green_time=resume_time,
                                    yellow_time=next_signal.yellow_time,
                                    red_time=next_signal.all_red_time,
                                    reason='resume_after_emergency'
                                )
                                return
            # Normal state transitions
            if active_signal.remaining_time <= 0:
                if active_signal.current_state == 'GREEN':
                    active_signal.current_state = 'YELLOW'
                    active_signal.remaining_time = active_signal.yellow_time
                    self.log_message(f"🟡 Signal {chr(65 + self.current_signal)} → YELLOW")
                    next_signal_idx = (self.current_signal + 1) % 4
                    self.run_detection_for_next_signal(next_signal_idx)
                elif active_signal.current_state == 'YELLOW':
                    active_signal.current_state = 'RED'
                    active_signal.remaining_time = 0
                    next_signal_idx = (self.current_signal + 1) % 4
                    next_signal = self.signals[next_signal_idx]
                    if hasattr(next_signal, 'pending_green_time') and next_signal.pending_green_time > 0:
                        next_signal.current_state = 'GREEN'
                        next_signal.remaining_time = next_signal.pending_green_time
                        self.log_message(f"🟢 Signal {chr(65 + next_signal_idx)} → GREEN for {next_signal.pending_green_time}s")
                        self.database.log_signal_timing_change(
                            signal_id=next_signal_idx,
                            green_time=next_signal.pending_green_time,
                            yellow_time=next_signal.yellow_time,
                            red_time=next_signal.all_red_time,
                            reason='automatic'
                        )
                        next_signal.pending_green_time = 0
                    else:
                        next_signal.current_state = 'GREEN'
                        next_signal.remaining_time = next_signal.default_green_time
                        self.log_message(f"🟢 Signal {chr(65 + next_signal_idx)} → GREEN for {next_signal.default_green_time}s (default)")
                        self.database.log_signal_timing_change(
                            signal_id=next_signal_idx,
                            green_time=next_signal.default_green_time,
                            yellow_time=next_signal.yellow_time,
                            red_time=next_signal.all_red_time,
                            reason='default'
                        )
                    self.current_signal = next_signal_idx
                    self.cycle_start_time = time.time()
                elif active_signal.current_state == 'RED':
                    active_signal.remaining_time = 0

    def run_detection_for_next_signal(self, signal_idx):
        # Run YOLO for signal_idx during yellow phase, store green time as pending
        def detection():
            time.sleep(0.2)
            frame = None
            if self.video_threads[signal_idx] and self.video_threads[signal_idx].isRunning():
                frame = self.video_threads[signal_idx].current_frame
            if frame is not None and signal_idx < len(self.areas):
                vehicle_count, traffic_weight, processed_frame, vehicle_type_counts, avg_confidence = self.detector.detect_vehicles_in_area(
                    frame, self.areas[signal_idx], draw_area=True
                )
                signal = self.signals[signal_idx]
                green_time = signal.calculate_adaptive_green_time(vehicle_count, traffic_weight, datetime.now())
                signal.vehicle_count = vehicle_count  # <-- Store detected count
                signal.traffic_weight = traffic_weight  # <-- Store detected weight
                signal.vehicle_type_counts = vehicle_type_counts.copy()  # Store vehicle type counts
                signal.pending_green_time = green_time
                signal.avg_confidence = avg_confidence
                self.log_message(f"[Detection during Yellow] Signal {chr(65 + signal_idx)}: Vehicles={vehicle_count}, Weight={traffic_weight:.1f}, Green={green_time}s")
                # Update the QLabel with processed_frame (with bboxes and count)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_labels[signal_idx].size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.video_labels[signal_idx].setPixmap(pixmap)
                # Update vehicle count and weight in the UI
                count_label = getattr(self, f'count_label_{signal_idx}', None)
                if count_label:
                    count_label.setText(f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}")
                
                # Update vehicle type count labels
                auto_label = getattr(self, f'auto_count_label_{signal_idx}', None)
                bike_label = getattr(self, f'bike_count_label_{signal_idx}', None)
                bus_label = getattr(self, f'bus_count_label_{signal_idx}', None)
                car_label = getattr(self, f'car_count_label_{signal_idx}', None)
                emergency_vehicles_label = getattr(self, f'emergency_vehicles_count_label_{signal_idx}', None)
                truck_label = getattr(self, f'truck_count_label_{signal_idx}', None)
                
                if auto_label:
                    auto_label.setText(f"🛺 Autos: {vehicle_type_counts['auto']}")
                if bike_label:
                    bike_label.setText(f"🏍️ Bikes: {vehicle_type_counts['bike']}")
                if bus_label:
                    bus_label.setText(f"🚌 Buses: {vehicle_type_counts['bus']}")
                if car_label:
                    car_label.setText(f"🚗 Cars: {vehicle_type_counts['car']}")
                if emergency_vehicles_label:
                    emergency_vehicles_label.setText(f"🚑 Emergency Vehicles: {vehicle_type_counts['emergency_vehicles']}")
                if truck_label:
                    truck_label.setText(f"🚚 Trucks: {vehicle_type_counts['truck']}")
                
                # Update signal data
                signal = self.signals[signal_idx]
                signal.vehicle_count = vehicle_count
                signal.traffic_weight = traffic_weight
                signal.vehicle_type_counts = vehicle_type_counts.copy()  # Store vehicle type counts
                # Calculate and set green time
                green_time = signal.calculate_adaptive_green_time(vehicle_count, traffic_weight, datetime.now())
                signal.pending_green_time = green_time
            else:
                signal = self.signals[signal_idx]
                signal.pending_green_time = signal.default_green_time
                self.log_message(f"[Detection during Yellow] Signal {chr(65 + signal_idx)}: Green={signal.default_green_time}s (default)")
        threading.Thread(target=detection, daemon=True).start()

    def process_video_frame(self, signal_idx, frame):
        """Process video frame and update display with bboxes and vehicle count"""
        try:
            if not self.areas or signal_idx >= len(self.areas):
                display_frame = cv2.resize(frame, (320, 240))
                vehicle_count = 0
                traffic_weight = 0
                vehicle_counts = {'auto': 0, 'bike': 0, 'bus': 0, 'car': 0, 'emergency_vehicles': 0, 'truck': 0}
            else:
                # Run detection on the original frame (not resized)
                vehicle_count, traffic_weight, processed_frame, vehicle_counts, avg_confidence = \
                    self.detector.detect_vehicles_in_area(frame, self.areas[signal_idx])
                # Update signal data
                if signal_idx < len(self.signals):
                    signal = self.signals[signal_idx]
                    signal.vehicle_count = vehicle_count
                    signal.traffic_weight = traffic_weight
                    signal.vehicle_type_counts = vehicle_counts.copy()  # CRITICAL: Update vehicle type counts
                    signal.calculate_adaptive_green_time(vehicle_count, traffic_weight)
                    
                    # Send data to congestion analysis thread (non-blocking)
                    if signal_idx < len(self.areas) and self.areas[signal_idx]:
                        area_size = calculate_polygon_area(self.areas[signal_idx])
                    else:
                        area_size = 1000  # Default area size if areas not available
                    
                    # Update congestion thread with new data
                    if hasattr(self, 'congestion_thread'):
                        self.congestion_thread.update_signal_data(signal_idx, vehicle_count, traffic_weight, area_size)
                display_frame = cv2.resize(processed_frame, (320, 240))
            
            # Convert to QImage and update QLabel
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            if signal_idx < len(self.video_labels) and self.video_labels[signal_idx]:
                self.video_labels[signal_idx].setPixmap(pixmap)
            
            # Update vehicle class counts in the UI
            vehicle_info_frame = self.findChild(QFrame, f'vehicle_info_{signal_idx}')
            if vehicle_info_frame:
                for class_name, count in vehicle_counts.items():
                    label = vehicle_info_frame.findChild(QLabel, f'{class_name}_label')
                    if label:
                        emoji = label.text().split()[0]
                        label.setText(f"{emoji} {count}")
            
            # Update vehicle count and weight label if present
            count_label = getattr(self, f'count_label_{signal_idx}', None)
            if count_label:
                count_label.setText(f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}")
                
            # Log emergency vehicle detection for debugging
            if vehicle_counts.get('emergency_vehicles', 0) > 0:
                self.log_message(f"🚑 Emergency vehicle detected at Signal {chr(65+signal_idx)}: Count = {vehicle_counts['emergency_vehicles']}")
                
        except Exception as e:
            print(f"Error processing frame for signal {signal_idx}: {e}")

    def update_system(self):
        if not self.is_running:
            return
            
        try:
            current_time = time.time()
            elapsed = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Ensure elapsed time is reasonable (not too large)
            if elapsed > 1.0:  # If more than 1 second has passed, cap it
                elapsed = 1.0
                self.log_message(f"Warning: Large time gap detected ({elapsed:.1f}s)")
            
            # Performance monitoring
            update_start = time.time()
            
            self.handle_signal_transitions(elapsed)
            self.update_analytics()
            
            # Log traffic data to MySQL database every 120 seconds (reduced frequency)
            if self.database.should_update():
                for i, signal in enumerate(self.signals):
                    # Verify data before logging
                    if self.database.verify_data_before_logging(
                        signal_id=i,
                        vehicle_count=signal.vehicle_count,
                        traffic_weight=signal.traffic_weight,
                        green_time=signal.calculated_green_time if hasattr(signal, 'calculated_green_time') else signal.default_green_time,
                        vehicle_type_counts=signal.vehicle_type_counts
                    ):
                        # Data is valid, proceed with logging
                        success = self.database.log_traffic_data(
                            signal_id=i,
                            vehicle_count=signal.vehicle_count,
                            traffic_weight=signal.traffic_weight,
                            green_time=signal.calculated_green_time if hasattr(signal, 'calculated_green_time') else signal.default_green_time,
                            vehicle_type_counts=signal.vehicle_type_counts
                        )
                        
                        if success:
                            self.log_message(f"✅ Signal {chr(65+i)} data logged successfully")
                        else:
                            self.log_message(f"❌ Signal {chr(65+i)} data logging failed")
                    else:
                        self.log_message(f"⚠️ Signal {chr(65+i)} data verification failed - skipping database log")
                    
                    # Log congestion data
                    if signal.congestion_level in ['HIGH', 'SEVERE']:
                        self.log_message(f"🚦 Signal {chr(65+i)}: {signal.congestion_level} congestion (Score: {signal.congestion_score:.1f})")
                
                self.log_message("Traffic data update cycle completed")
            
            # Performance monitoring - log if system is running slowly
            update_time = time.time() - update_start
            if update_time > 0.05:  # If update takes more than 50ms
                print(f"System update took {update_time:.3f}s (target: <0.05s)")
            
            # Force UI update for timers using QTimer.singleShot
            QTimer.singleShot(0, self.update_ui)
            
        except Exception as e:
            self.log_message(f"Error in system update: {str(e)}")

    def update_ui(self):
        if not self.is_running:
            return
            
        try:
            for i, signal in enumerate(self.signals):
                # Update signal status
                if signal.current_state == 'GREEN':
                    status_text = "🟢 GREEN"
                    color = 'green'
                elif signal.current_state == 'YELLOW':
                    status_text = "🟡 YELLOW"
                    color = 'orange'
                else:
                    status_text = "🔴 RED"
                    color = 'red'
                
                status_label = getattr(self, f'status_label_{i}')
                time_label = getattr(self, f'time_label_{i}')
                count_label = getattr(self, f'count_label_{i}')
                
                status_label.setText(status_text)
                status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                
                # Format remaining time with one decimal place
                remaining_time = max(0, signal.remaining_time)
                time_label.setText(f"Time: {remaining_time:.1f}s")
                
                count_label.setText(
                    f"Vehicles: {signal.vehicle_count} | Weight: {signal.traffic_weight:.1f}"
                )
            
            # Update system metrics
            total_vehicles = sum(signal.vehicle_count for signal in self.signals)
            self.total_vehicles_label.setText(f"Total Vehicles: {total_vehicles}")
            
            cycle_time = int(time.time() - self.cycle_start_time)
            self.cycle_time_label.setText(f"Cycle Time: {cycle_time}s")
            
            self.current_signal_label.setText(f"Active Signal: {chr(65 + self.current_signal)}")
            
        except Exception as e:
            self.log_message(f"Error updating UI: {str(e)}")

    def update_analytics(self):
        current_time = datetime.now()
        self.analytics_data['timestamps'].append(current_time)
        
        # Update analytics data for each signal
        for i in range(4):
            signal = self.signals[i]
            
            # Update vehicle counts with current data
            self.analytics_data['vehicle_counts'][i].append(signal.vehicle_count)
            
            # Update green times based on current state and calculated times
            if signal.current_state == 'GREEN':
                green_time = signal.calculated_green_time if hasattr(signal, 'calculated_green_time') else signal.default_green_time
            else:
                green_time = signal.pending_green_time if hasattr(signal, 'pending_green_time') and signal.pending_green_time > 0 else signal.default_green_time
            
            self.analytics_data['green_times'][i].append(green_time)

    def show_analytics(self):
        analytics_dialog = QDialog(self)
        analytics_dialog.setWindowTitle("Traffic Analytics Dashboard")
        analytics_dialog.setGeometry(200, 200, 1000, 800)
        
        layout = QVBoxLayout()
        
        # Loading indicator
        loading_label = QLabel("🔄 Loading Analytics Dashboard...")
        loading_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #3498db;
                padding: 20px;
            }
        """)
        loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(loading_label)
        
        # Create matplotlib canvas placeholder
        canvas = FigureCanvas(plt.figure(figsize=(12, 8)))  # Reduced figure size
        layout.addWidget(canvas)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        
        # Update button
        update_btn = QPushButton("Update Analytics")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        
        # Auto-update checkbox
        auto_update_check = QCheckBox("Auto Update")
        auto_update_check.setStyleSheet("""
            QCheckBox {
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        
        # Add buttons to layout
        button_layout.addWidget(update_btn)
        button_layout.addWidget(auto_update_check)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        analytics_dialog.setLayout(layout)
        
        # Track if dialog is visible to prevent updates when closed
        dialog_visible = True
        
        def update_analytics_display(data):
            """Update the canvas with new data."""
            if not data or not dialog_visible:
                return

            try:
                fig = canvas.figure
                fig.clear()
                
                # Re-create the layout based on the user's image
                gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.3, height_ratios=[1, 1, 1.2])
                ax_trends = fig.add_subplot(gs[0, 0])
                ax_green_opt = fig.add_subplot(gs[0, 1])
                ax_efficiency = fig.add_subplot(gs[1, 0])
                ax_traffic_dist = fig.add_subplot(gs[1, 1])
                ax_congestion = fig.add_subplot(gs[2, :])

                fig.suptitle('Traffic Analytics Dashboard', fontsize=14, fontweight='bold')

                colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
                signal_labels = [f'Signal {chr(65+i)}' for i in range(4)]
                
                # 1. Vehicle Count Trends
                ax_trends.set_title('Vehicle Count Trends', fontsize=11, fontweight='bold')
                timestamps = data.get('timestamps', [])
                times = list(range(len(timestamps)))
                
                def format_timedelta(x, pos):
                    if timestamps and len(timestamps) > int(x):
                        td = timestamps[int(x)] - timestamps[0]
                        return f"{int(td.total_seconds())}s"
                    return ""

                for i in range(4):
                    counts = data['vehicle_counts'][i]
                    if counts:
                        ax_trends.plot(times, counts, label=f'Signal {chr(65+i)}', color=colors[i], marker='o', markersize=3, linewidth=1.5)
                ax_trends.grid(True, linestyle='--', alpha=0.6)
                if times:
                    ax_trends.xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta))
                ax_trends.legend(fontsize='small')
                ax_trends.set_ylabel('Vehicle Count')

                # 2. Green Time Optimization
                ax_green_opt.set_title('Green Time Optimization', fontsize=11, fontweight='bold')
                for i in range(4):
                    green_times = data['green_times'][i]
                    if green_times:
                        ax_green_opt.plot(list(range(len(green_times))), green_times, label=f'Signal {chr(65+i)}', color=colors[i], marker='s', markersize=3, linewidth=1.5)
                ax_green_opt.grid(True, linestyle='--', alpha=0.6)
                ax_green_opt.legend(fontsize='small')
                ax_green_opt.set_ylabel('Green Time (s)')
                
                # 3. YOLO Model Accuracy (Bar Chart)
                ax_efficiency.set_title('YOLO Model Accuracy', fontsize=11, fontweight='bold')
                avg_confidences = data.get('avg_confidences', [0.0] * 4)
                accuracy_percentages = [c * 100 for c in avg_confidences]
                bars = ax_efficiency.bar(signal_labels, accuracy_percentages, color=colors)
                ax_efficiency.grid(True, axis='y', linestyle='--', alpha=0.6)
                ax_efficiency.set_ylim(0, 100)
                for bar in bars:
                    height = bar.get_height()
                    ax_efficiency.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                ax_efficiency.set_ylabel('Avg. Confidence (%)')

                # 4. Current Traffic Distribution (Pie Chart)
                ax_traffic_dist.set_title('Current Traffic Distribution', fontsize=11, fontweight='bold')
                distribution = data['vehicle_distribution']
                total_vehicles = sum(distribution)
                if total_vehicles > 0:
                    ax_traffic_dist.pie(distribution, labels=signal_labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
                else:
                    ax_traffic_dist.text(0.5, 0.5, "No traffic data", ha='center', va='center')
                ax_traffic_dist.axis('equal')

                # 5. Congestion Levels by Signal
                ax_congestion.set_title('Congestion Levels by Signal', fontsize=11, fontweight='bold')
                congestion_data = data['congestion_data']
                congestion_scores = [congestion_data.get(i, {}).get('score', 0) for i in range(4)]
                congestion_levels = [congestion_data.get(i, {}).get('level', 'LOW') for i in range(4)]
                congestion_colors = ['#2ecc71' if level == 'LOW' else '#f39c12' if level == 'MODERATE' else '#e74c3c' if level == 'HIGH' else '#c0392b' for level in congestion_levels]
                
                detailed_labels = [f"Signal {chr(65+i)}\n{congestion_levels[i]}" for i in range(4)]
                bars = ax_congestion.bar(detailed_labels, congestion_scores, color=congestion_colors)
                ax_congestion.grid(True, axis='y', linestyle='--', alpha=0.6)
                ax_congestion.set_ylabel('Congestion Score')
                for bar in bars:
                    height = bar.get_height()
                    ax_congestion.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

                canvas.draw_idle()
                loading_label.hide()
                
            except Exception as e:
                if dialog_visible:
                    loading_label.setText("❌ Error rendering analytics")
                print(f"Error updating analytics display: {e}")

        # Create and connect analytics thread
        analytics_thread = AnalyticsThread(
            self.analytics_data, 
            self.signals, 
            self.get_congestion_data_for_analytics
        )
        analytics_thread.analytics_data_ready.connect(update_analytics_display)
        analytics_thread.start()
        
        # Auto-update timer with longer interval to reduce lag
        auto_update_timer = QTimer()
        
        def toggle_auto_update(state):
            if state:
                auto_update_timer.start(5000)  # Update every 5 seconds instead of 2
            else:
                auto_update_timer.stop()
        
        auto_update_check.stateChanged.connect(toggle_auto_update)
        
        # Manual update button
        def manual_update():
            analytics_thread.trigger_update()
        
        update_btn.clicked.connect(manual_update)
        
        # Cleanup when dialog closes
        def cleanup():
            dialog_visible = False
            analytics_thread.stop()
            auto_update_timer.stop()
            analytics_dialog.deleteLater()

        analytics_dialog.finished.connect(cleanup)
        
        analytics_dialog.exec_()

    def show_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("System Settings")
        settings_dialog.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Signal Timing Settings")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Settings for each signal
        min_entries = []
        max_entries = []
        
        for i in range(4):
            group = QGroupBox(f"Signal {chr(65+i)}")
            group_layout = QHBoxLayout()
            
            # Min green time
            min_label = QLabel("Min Green Time:")
            min_entry = QLineEdit()
            min_entry.setText(str(self.signals[i].min_green_time))
            min_entries.append(min_entry)
            
            # Max green time
            max_label = QLabel("Max Green Time:")
            max_entry = QLineEdit()
            max_entry.setText(str(self.signals[i].max_green_time))
            max_entries.append(max_entry)
            
            group_layout.addWidget(min_label)
            group_layout.addWidget(min_entry)
            group_layout.addWidget(max_label)
            group_layout.addWidget(max_entry)
            
            group.setLayout(group_layout)
            layout.addWidget(group)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        
        def save_settings():
            try:
                for i in range(4):
                    self.signals[i].min_green_time = int(min_entries[i].text())
                    self.signals[i].max_green_time = int(max_entries[i].text())
                
                config = {}
                for i in range(4):
                    config[f'signal_{i}'] = {
                        'min_green_time': self.signals[i].min_green_time,
                        'max_green_time': self.signals[i].max_green_time
                    }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                QMessageBox.information(settings_dialog, "Success", 
                                      "Settings saved successfully!")
                settings_dialog.accept()
                
            except ValueError:
                QMessageBox.critical(settings_dialog, "Error",
                                   "Please enter valid numbers for timing values")
        
        save_btn.clicked.connect(save_settings)
        layout.addWidget(save_btn)
        
        settings_dialog.setLayout(layout)
        settings_dialog.exec_()

    def define_new_areas(self):
        if not self.video_sources:
            QMessageBox.critical(self, "Error", "No video sources configured!")
            return
        
        # Verify all configured video files exist and are accessible
        missing_videos = []
        for video_file in self.video_sources:
            if not os.path.exists(video_file):
                missing_videos.append(video_file)
        
        if missing_videos:
            QMessageBox.critical(self, "Error", 
                "Missing required video files:\n" + "\n".join(missing_videos) +
                "\n\nPlease ensure all video files are present and accessible.")
            return
        
        # Test each video file
        for video_path in self.video_sources:
            try:
                test_cap = cv2.VideoCapture(video_path)
                if not test_cap.isOpened():
                    QMessageBox.critical(self, "Error", f"Cannot open video file: {video_path}")
                    return
                # Read a test frame
                ret, frame = test_cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", f"Cannot read frames from: {video_path}")
                    return
                test_cap.release()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error testing video file {video_path}: {str(e)}")
                return
        
        # Update video sources with verified files
        self.video_sources = self.video_sources.copy()
        
        # Log the video sources
        self.log_message("Using video sources:")
        for i, video in enumerate(self.video_sources):
            self.log_message(f"Signal {chr(65+i)}: {video}")
        
        area_dialog = QDialog(self)
        area_dialog.setWindowTitle("Define Detection Areas")
        area_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # Debug info label
        debug_label = QLabel("")
        debug_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
        layout.addWidget(debug_label)
        
        # Instructions with signal-specific guidance
        instructions = QLabel(
            "Click to define 4 corners of detection area for each signal:\n"
            "Signal A & C: Define points from top to bottom\n"
            "Press 'Next Signal' when done with current signal\n"
            "Points Selected: 0/4"
        )
        instructions.setStyleSheet("font-size: 12px; color: #2c3e50; padding: 10px;")
        layout.addWidget(instructions)
        
        # Current signal indicator with specific instructions
        current_signal_label = QLabel("Defining area for Signal A\nDefine points from top-left to bottom-right")
        current_signal_label.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(current_signal_label)
        
        # Video display
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        video_frame.setLineWidth(2)
        video_layout = QVBoxLayout()
        
        class ClickableLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setMouseTracking(True)
                self.points = []
                self.display_size = [640, 480]
                self.setMinimumSize(640, 480)
            
            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton and len(self.points) < 4:
                    pos = event.pos()
                    if self.pixmap():
                        # Get the actual image rect
                        img_rect = self.get_image_rect()
                        if img_rect.contains(pos):
                            # Convert coordinates from widget space to image space
                            x = int((pos.x() - img_rect.x()) / img_rect.width() * self.display_size[0])
                            y = int((pos.y() - img_rect.y()) / img_rect.height() * self.display_size[1])
                            
                            # Validate point is not too close to existing points
                            too_close = False
                            for p in self.points:
                                dx = x - p[0]
                                dy = y - p[1]
                                if (dx * dx + dy * dy) < 100:  # Points too close together
                                    too_close = True
                                    break
                            
                            if not too_close:
                                point = (x, y)
                                self.points.append(point)
                                debug_label.setText(f"Added point {len(self.points)} at ({point[0]}, {point[1]})")
                                instructions.setText(
                                    "Click to define 4 corners of detection area for each signal:\n"
                                    "Signal A & C: Define points from top to bottom\n"
                                    "Press 'Next Signal' when done with current signal\n"
                                    f"Points Selected: {len(self.points)}/4"
                                )
                                self.update()
                            else:
                                debug_label.setText("Point too close to existing point. Please choose a different location.")
            
            def paintEvent(self, event):
                super().paintEvent(event)
                if not self.pixmap() or not self.points:
                    return
                    
                painter = QPainter(self)
                img_rect = self.get_image_rect()
                
                # Draw points and lines
                for i, point in enumerate(self.points):
                    # Convert image coordinates to widget coordinates
                    x = int(point[0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    y = int(point[1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    
                    # Draw point
                    painter.setPen(QPen(Qt.white, 3))  # White outline
                    painter.setBrush(QBrush(Qt.red))
                    painter.drawEllipse(x - 6, y - 6, 12, 12)
                    
                    # Draw number
                    painter.setPen(QPen(Qt.white, 2))
                    painter.setFont(QFont("Arial", 12, QFont.Bold))
                    painter.drawText(x - 8, y - 8, 16, 16, Qt.AlignCenter, str(i + 1))
                    
                    # Draw lines
                    if i > 0:
                        prev_x = int(self.points[i-1][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                        prev_y = int(self.points[i-1][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                        painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow lines
                        painter.drawLine(prev_x, prev_y, x, y)
                
                # Close the polygon if all points are selected
                if len(self.points) == 4:
                    first_x = int(self.points[0][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    first_y = int(self.points[0][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    last_x = int(self.points[-1][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    last_y = int(self.points[-1][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.drawLine(last_x, last_y, first_x, first_y)
            
            def get_image_rect(self):
                if self.pixmap():
                    # Get the scaled pixmap size
                    scaled_size = self.pixmap().size()
                    scaled_size.scale(self.size(), Qt.KeepAspectRatio)
                    
                    # Calculate position to center the image
                    x = (self.width() - scaled_size.width()) // 2
                    y = (self.height() - scaled_size.height()) // 2
                    
                    return QRect(x, y, scaled_size.width(), scaled_size.height())
                return QRect()
            
            def clear_points(self):
                self.points.clear()
                instructions.setText(
                    "Click to define 4 corners of detection area for each signal:\n"
                    "Signal A & C: Define points from top to bottom\n"
                    "Press 'Next Signal' when done with current signal\n"
                    "Points Selected: 0/4"
                )
                debug_label.setText("Points cleared")
                self.update()
        
        # Create and setup video label
        video_label = ClickableLabel()
        video_label.setMinimumSize(640, 480)
        video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #3498db;
                border-radius: 4px;
            }
        """)
        video_layout.addWidget(video_label, alignment=Qt.AlignCenter)
        video_frame.setLayout(video_layout)
        layout.addWidget(video_frame)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        next_btn = QPushButton("Next Signal")
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        clear_btn = QPushButton("Clear Points")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        
        button_layout.addWidget(next_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        area_dialog.setLayout(layout)
        
        # State variables
        current_signal_idx = [0]
        all_areas = []
        current_cap = [None]
        original_frame = [None]
        scale_factor = [1.0, 1.0]
        
        def init_video_capture():
            # Release previous capture if it exists
            if current_cap[0] is not None:
                current_cap[0].release()
                current_cap[0] = None
            
            if current_signal_idx[0] < len(self.video_sources):
                try:
                    # Get the correct video source for the current signal
                    video_path = self.video_sources[current_signal_idx[0]]
                    debug_label.setText(f"Opening video for Signal {chr(65+current_signal_idx[0])}: {video_path}")
                    
                    # Force reopen the video file
                    if os.path.exists(video_path):
                        # Try to release any system handles to the video
                        import gc
                        gc.collect()
                        
                        # Wait a bit to ensure file handle is released
                        time.sleep(0.1)
                        
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            debug_label.setText(f"Failed to open video: {video_path}")
                            QMessageBox.critical(area_dialog, "Error", f"Failed to open video: {video_path}")
                            return False
                        
                        current_cap[0] = cap
                        
                        # Set capture properties
                        current_cap[0].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        current_cap[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        current_cap[0].set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                        current_cap[0].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
                        
                        # Read first frame
                        ret, frame = current_cap[0].read()
                        if not ret or frame is None:
                            debug_label.setText(f"Failed to read frame from: {video_path}")
                            QMessageBox.critical(area_dialog, "Error", f"Failed to read frame from: {video_path}")
                            return False
                        
                        original_frame[0] = frame.copy()
                        h, w = frame.shape[:2]
                        scale_factor[0] = w / video_label.display_size[0]
                        scale_factor[1] = h / video_label.display_size[1]
                        
                        # Reset video label points when switching videos
                        video_label.clear_points()
                        
                        debug_label.setText(f"Successfully opened video for Signal {chr(65+current_signal_idx[0])}: {video_path} ({w}x{h})")
                        return True
                    else:
                        debug_label.setText(f"Video file not found: {video_path}")
                        QMessageBox.critical(area_dialog, "Error", f"Video file not found: {video_path}")
                        return False
                    
                except Exception as e:
                    error_msg = f"Error opening video source: {str(e)}"
                    debug_label.setText(error_msg)
                    QMessageBox.critical(area_dialog, "Error", error_msg)
                    return False
            return False
        
        def update_preview():
            if current_cap[0] is not None and current_cap[0].isOpened():
                try:
                    # Get the current video source
                    current_video = self.video_sources[current_signal_idx[0]]
                    debug_label.setText(f"Previewing Signal {chr(65+current_signal_idx[0])} - {current_video}")
                    
                    ret, frame = current_cap[0].read()
                    if not ret or frame is None:
                        # Try to reopen the video if we can't read a frame
                        current_cap[0].release()
                        current_cap[0] = cv2.VideoCapture(current_video)
                        ret, frame = current_cap[0].read()
                        if not ret or frame is None:
                            debug_label.setText(f"Failed to read frame from {current_video}")
                            return
                    
                    original_frame[0] = frame.copy()
                    display_frame = cv2.resize(frame, (video_label.display_size[0], video_label.display_size[1]))
                    
                    # Add text overlay showing which signal and video
                    cv2.putText(display_frame, 
                              f"Signal {chr(65+current_signal_idx[0])} - {current_video}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Convert to QImage and display
                    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    # Scale pixmap to fit label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    video_label.setPixmap(scaled_pixmap)
                    
                except Exception as e:
                    debug_label.setText(f"Error in update_preview: {str(e)}")
            
            if area_dialog.isVisible():
                QTimer.singleShot(30, update_preview)  # Update every 30ms
        
        def init_next_video():
            if not init_video_capture():
                QMessageBox.critical(area_dialog, "Error",
                                f"Could not open video source for Signal {chr(65+current_signal_idx[0])}")
                area_dialog.reject()
            else:
                update_preview()

        def next_signal():
            if len(video_label.points) == 4:
                # Get original video size for this signal
                video_path = self.video_sources[current_signal_idx[0]]
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    QMessageBox.critical(area_dialog, "Error", f"Could not read frame from {video_path}")
                    return
                original_height, original_width = frame.shape[:2]
                display_width, display_height = video_label.display_size
                video_points = []
                for x, y in video_label.points:
                    video_x = int((x / display_width) * original_width)
                    video_y = int((y / display_height) * original_height)
                    video_points.append([video_x, video_y])
                
                if not self.validate_area_shape(video_points, current_signal_idx[0]):
                    QMessageBox.warning(area_dialog, "Invalid Area",
                                    "Please define a valid area (points should form a proper polygon)")
                    return
                
                all_areas.append(video_points)
                self.log_message(f"Added area for Signal {chr(65+current_signal_idx[0])}: {video_points}")
                video_label.clear_points()
                current_signal_idx[0] += 1
                
                if current_signal_idx[0] < 4:
                    current_signal_label.setText(f"Defining area for Signal {chr(65+current_signal_idx[0])}\n" +
                                            ("Define points from top to bottom" if current_signal_idx[0] in [0, 2] else ""))
                    
                    if current_cap[0] is not None:
                        current_cap[0].release()
                        current_cap[0] = None
                    QTimer.singleShot(500, lambda: init_next_video())
                else:
                    try:
                        with open(self.areas_file, 'w') as f:
                            json.dump(all_areas, f, indent=4)
                        self.log_message("Areas saved to file successfully")
                        QMessageBox.information(area_dialog, "Success",
                                            "All detection areas saved successfully!")
                        self.areas = all_areas.copy()
                        area_dialog.accept()
                        self.log_message("New detection areas defined and saved")
                    except Exception as e:
                        self.log_message(f"Error saving areas: {e}")
                        QMessageBox.critical(area_dialog, "Error",
                                        f"Error saving areas: {e}")
            else:
                QMessageBox.warning(area_dialog, "Incomplete",
                                "Please define all 4 corners before proceeding.")
        
        def cleanup_resources():
            if current_cap[0] is not None:
                current_cap[0].release()
                current_cap[0] = None

        area_dialog.finished.connect(lambda: cleanup_resources())
        
        # Connect signals
        next_btn.clicked.connect(next_signal)
        clear_btn.clicked.connect(lambda: video_label.clear_points())
        cancel_btn.clicked.connect(lambda: area_dialog.reject())
        
        # Initialize first video capture
        if not init_video_capture():
            QMessageBox.critical(area_dialog, "Error", "Could not open first video source!")
            return
        
        # Start video preview
        update_preview()
        
        # Show dialog
        area_dialog.exec_()
        
        # Cleanup
        if current_cap[0] is not None:
            current_cap[0].release()

    def load_areas(self, show_message=True):
        """Load detection areas from JSON file."""
        try:
            if not os.path.exists(self.areas_file):
                self.log_message("⚠️ Areas file not found!")
                if show_message:
                    QMessageBox.warning(self, "Warning", "Areas file not found. Please define new areas first.")
                return False
            
            with open(self.areas_file, 'r') as f:
                loaded_areas = json.load(f)
            
            # Validate loaded areas
            if not isinstance(loaded_areas, list) or len(loaded_areas) != 4:
                self.log_message("⚠️ Invalid areas format - expected list of 4 areas")
                if show_message:
                    QMessageBox.warning(self, "Warning", "Invalid areas format in file. Please define new areas.")
                return False
            
            # Validate each area
            for i, area in enumerate(loaded_areas):
                if not isinstance(area, list) or len(area) != 4:
                    self.log_message(f"⚠️ Invalid format for area {i} - expected list of 4 points")
                    if show_message:
                        QMessageBox.warning(self, "Warning", f"Invalid format for Signal {chr(65+i)} area. Please define new areas.")
                    return False
                for point in area:
                    if not isinstance(point, list) or len(point) != 2:
                        self.log_message(f"⚠️ Invalid point format in area {i} - expected [x, y]")
                        if show_message:
                            QMessageBox.warning(self, "Warning", f"Invalid point format in Signal {chr(65+i)}. Please define new areas.")
                        return False
                    x, y = point
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        self.log_message(f"⚠️ Invalid coordinates in area {i} - expected numbers")
                        if show_message:
                            QMessageBox.warning(self, "Warning", f"Invalid coordinates in Signal {chr(65+i)}. Please define new areas.")
                        return False
                    
                # Convert to integer coordinates
                loaded_areas[i] = [[int(x), int(y)] for x, y in area]
                
                # Validate the area shape
                if not self.validate_area_shape(loaded_areas[i], i):
                    self.log_message(f"⚠️ Invalid area shape for Signal {chr(65+i)}")
                    if show_message:
                        QMessageBox.warning(self, "Warning", f"Invalid area shape for Signal {chr(65+i)}. Please define new areas.")
                    return False
            
            # Stop any running threads before updating areas
            if self.is_running:
                self.stop_system()
            
            self.areas = loaded_areas
            self.log_message("✅ Areas loaded successfully")
            self.log_message(f"Loaded areas: {self.areas}")
            if show_message:
                QMessageBox.information(self, "Success", "Areas loaded successfully!")
            return True
            
        except Exception as e:
            self.log_message(f"⚠️ Error loading areas: {e}")
            if show_message:
                QMessageBox.critical(self, "Error", f"Failed to load areas: {str(e)}")
            return False

    def validate_area_shape(self, points, signal_idx):
        """Validate the shape and size of an area"""
        try:
            if len(points) != 4:
                return False

            # Calculate area using shoelace formula
            area = 0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2

            # Only do basic validation for Signal A and D
            if signal_idx in [0, 3]:  # Signal A and D
                # Just ensure the area is not zero and points form a valid polygon
                return area > 0

            # More strict validation for other signals
            elif signal_idx == 2:  # Signal C
                # Validate trapezoid shape
                top_width = abs(points[1][0] - points[0][0])
                bottom_width = abs(points[3][0] - points[2][0])
                if top_width < 35 or bottom_width < 35:
                    return False
                
                # Check height
                height = max(p[1] for p in points) - min(p[1] for p in points)
                if height < 70:
                    return False

            # General validation - very minimal requirements
            min_area = 1000  # Greatly reduced minimum area
            if area < min_area:
                return False

            return True

        except Exception as e:
            self.log_message(f"Error in area validation: {str(e)}")
            return False

    def stop_system(self):
        """Stop the traffic management system"""
        if not self.is_running:
            return
            
        self.log_message("🛑 Stopping traffic management system...")
        self.is_running = False
        
        # Stop timers first
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'ui_timer'):
            self.ui_timer.stop()
        
        # Stop video threads
        for i, thread in enumerate(self.video_threads):
            if thread:
                self.log_message(f"Stopping video thread for Signal {chr(65 + i)}...")
                thread.stop()
                thread.wait()  # Ensure thread is fully stopped
                self.video_threads[i] = None
        
        # Stop congestion analysis thread
        if hasattr(self, 'congestion_thread'):
            self.log_message("Stopping congestion analysis thread...")
            self.congestion_thread.stop()
        
        # Stop analytics thread if running
        if hasattr(self, 'analytics_thread') and self.analytics_thread is not None:
            self.log_message("Stopping analytics thread...")
            self.analytics_thread.stop()
            self.analytics_thread = None
        
        # Reset all signals
        for signal in self.signals:
            signal.current_state = 'RED'
            signal.remaining_time = 0
            signal.vehicle_count = 0
            signal.traffic_weight = 0
        
        # Clear video displays
        for video_label in self.video_labels:
            if video_label:
                video_label.clear()
                video_label.setText("Video Stopped")
                video_label.setStyleSheet("background-color: #2c3e50; color: white; font-size: 12px;")
        
        self.log_message("System stopped and reset.")
        QMessageBox.information(self, "System Stopped", "Traffic management system has been stopped.")

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    for i, signal in enumerate(self.signals):
                        if f'signal_{i}' in config:
                            signal_config = config[f'signal_{i}']
                            signal.min_green_time = signal_config.get('min_green_time', 10)
                            signal.max_green_time = signal_config.get('max_green_time', 45)
        except Exception as e:
            self.log_message(f"Error loading config: {e}")

    def toggle_emergency_mode(self):
        self.emergency_mode = not self.emergency_mode
        status = "ACTIVATED" if self.emergency_mode else "DEACTIVATED"
        self.log_message(f"Emergency mode {status}")
        
        if self.emergency_mode:
            QMessageBox.information(
                self, "Emergency Mode",
                "Emergency mode activated!\nAll signals will prioritize emergency vehicles."
            )

    def test_database(self):
        """Test database functionality"""
        pass

    def show_detection_for_signal(self, signal_idx):
        """Run detection for the given signal and display the processed frame with bboxes and vehicle count"""
        frame = self.video_threads[signal_idx].current_frame
        if frame is not None and signal_idx < len(self.areas):
            vehicle_count, traffic_weight, processed_frame, vehicle_type_counts, avg_confidence = self.detector.detect_vehicles_in_area(
                frame, self.areas[signal_idx], draw_area=True
            )
            # Update the QLabel with processed_frame (with bboxes and count)
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_labels[signal_idx].size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_labels[signal_idx].setPixmap(pixmap)
            # Update vehicle count and weight in the UI
            count_label = getattr(self, f'count_label_{signal_idx}', None)
            if count_label:
                count_label.setText(f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}")
            # Update signal data
            signal = self.signals[signal_idx]
            signal.vehicle_count = vehicle_count
            signal.traffic_weight = traffic_weight
            signal.avg_confidence = avg_confidence
            # Calculate and set green time
            green_time = signal.calculate_adaptive_green_time(vehicle_count, traffic_weight, datetime.now())
            signal.pending_green_time = green_time

    def get_vehicle_class_counts(self, time_range='24h'):
        """Get counts of each vehicle class from the database"""
        try:
            import mysql.connector
            from datetime import datetime, timedelta
            
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Calculate time threshold based on time_range
            now = datetime.now()
            if time_range == '24h':
                threshold = now - timedelta(hours=24)
            elif time_range == '7d':
                threshold = now - timedelta(days=7)
            elif time_range == '30d':
                threshold = now - timedelta(days=30)
            else:
                threshold = now - timedelta(hours=24)  # Default to 24h
            
            # Query to get total counts for each vehicle class
            cursor.execute('''
                SELECT 
                    SUM(auto_count) as total_autos,
                    SUM(bike_count) as total_bikes,
                    SUM(bus_count) as total_buses,
                    SUM(car_count) as total_cars,
                    SUM(emergency_vehicles_count) as total_emergency_vehicles,
                    SUM(truck_count) as total_trucks,
                    COUNT(*) as total_records
                FROM traffic_data
                WHERE timestamp >= %s
            ''', (threshold,))
            
            total_counts = cursor.fetchone()
            
            # Query to get counts per signal
            cursor.execute('''
                SELECT 
                    signal_id,
                    SUM(auto_count) as total_autos,
                    SUM(bike_count) as total_bikes,
                    SUM(bus_count) as total_buses,
                    SUM(car_count) as total_cars,
                    SUM(emergency_vehicles_count) as total_emergency_vehicles,
                    SUM(truck_count) as total_trucks
                FROM traffic_data
                WHERE timestamp >= %s
                GROUP BY signal_id
                ORDER BY signal_id
            ''', (threshold,))
            
            signal_counts = cursor.fetchall()
            
            # Query to get hourly distribution
            cursor.execute('''
                SELECT 
                    HOUR(timestamp) as hour,
                    SUM(auto_count) as total_autos,
                    SUM(bike_count) as total_bikes,
                    SUM(bus_count) as total_buses,
                    SUM(car_count) as total_cars,
                    SUM(emergency_vehicles_count) as total_emergency_vehicles,
                    SUM(truck_count) as total_trucks
                FROM traffic_data
                WHERE timestamp >= %s
                GROUP BY HOUR(timestamp)
                ORDER BY hour
            ''', (threshold,))
            
            hourly_counts = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'total_counts': total_counts,
                'signal_counts': signal_counts,
                'hourly_counts': hourly_counts,
                'time_range': time_range,
                'from_time': threshold.strftime('%Y-%m-%d %H:%M:%S'),
                'to_time': now.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error getting vehicle class counts: {e}")
            return None
            
    def print_vehicle_statistics(self, time_range='24h'):
        """Print vehicle statistics in a readable format"""
        stats = self.get_vehicle_class_counts(time_range)
        if not stats:
            print("Failed to get vehicle statistics")
            return
            
        print("\n=== Vehicle Class Statistics ===")
        print(f"Time Range: {stats['time_range']}")
        print(f"From: {stats['from_time']}")
        print(f"To: {stats['to_time']}")
        
        print("\nTotal Counts:")
        print(f"Autos: {stats['total_counts']['total_autos']}")
        print(f"Bikes: {stats['total_counts']['total_bikes']}")
        print(f"Buses: {stats['total_counts']['total_buses']}")
        print(f"Cars: {stats['total_counts']['total_cars']}")
        print(f"Emergency Vehicles: {stats['total_counts']['total_emergency_vehicles']}")
        print(f"Trucks: {stats['total_counts']['total_trucks']}")
        print(f"Total Records: {stats['total_counts']['total_records']}")
        
        print("\nCounts per Signal:")
        for signal in stats['signal_counts']:
            print(f"\nSignal {chr(65 + signal['signal_id'])}:")
            print(f"  Autos: {signal['total_autos']}")
            print(f"  Bikes: {signal['total_bikes']}")
            print(f"  Buses: {signal['total_buses']}")
            print(f"  Cars: {signal['total_cars']}")
            print(f"  Emergency Vehicles: {signal['total_emergency_vehicles']}")
            print(f"  Trucks: {signal['total_trucks']}")
        
        print("\nHourly Distribution:")
        for hour in stats['hourly_counts']:
            print(f"\nHour {hour['hour']:02d}:00:")
            print(f"  Autos: {hour['total_autos']}")
            print(f"  Bikes: {hour['total_bikes']}")
            print(f"  Buses: {hour['total_buses']}")
            print(f"  Cars: {hour['total_cars']}")
            print(f"  Emergency Vehicles: {hour['total_emergency_vehicles']}")
            print(f"  Trucks: {hour['total_trucks']}")

    def log_signal_timing_change(self, signal_id, green_time, yellow_time, red_time, reason=None):
        """Log a signal timing change to the signal_timing_logs table."""
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_timing_logs (signal_id, green_time, yellow_time, red_time, reason)
                VALUES (%s, %s, %s, %s, %s)
            ''', (signal_id, green_time, yellow_time, red_time, reason))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error logging signal timing change: {e}")

    def log_congestion_event(self, signal_id, severity, cause, resolution_time=None):
        """Log a congestion spike or anomaly event to the congestion_events table."""
        try:
            import mysql.connector
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO congestion_events (signal_id, severity, cause, resolution_time)
                VALUES (%s, %s, %s, %s)
            ''', (signal_id, severity, cause, resolution_time))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error logging congestion event: {e}")
    
    def update_congestion_display(self, signal_idx, congestion_level, congestion_score, color):
        """Update congestion display from the dedicated analysis thread"""
        try:
            # Update congestion display in UI
            congestion_label = getattr(self, f'congestion_label_{signal_idx}', None)
            if congestion_label:
                congestion_label.setText(f"Congestion: {congestion_level} ({congestion_score:.1f})")
                congestion_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10px;")
            
            # Log significant congestion changes
            if congestion_level in ['HIGH', 'SEVERE']:
                self.log_message(f"⚠️ High congestion at Signal {chr(65+signal_idx)}: {congestion_level} (Score: {congestion_score:.1f})")
                # Log congestion event to database
                if hasattr(self, 'database') and hasattr(self.database, 'log_congestion_event'):
                    self.database.log_congestion_event(
                        signal_id=signal_idx,
                        severity=congestion_level,
                        cause=f"Congestion score {congestion_score:.1f}",
                        resolution_time=None
                    )
            elif congestion_level == 'MODERATE':
                self.log_message(f"📊 Moderate congestion at Signal {chr(65+signal_idx)}: {congestion_level} (Score: {congestion_score:.1f})")
                
        except Exception as e:
            print(f"Error updating congestion display: {e}")

    def get_congestion_data_for_analytics(self):
        """Get congestion data from the congestion analysis thread for analytics dashboard"""
        congestion_data = {}
        if hasattr(self, 'congestion_thread'):
            for signal_idx in range(4):
                if signal_idx in self.congestion_thread.signal_data:
                    data = self.congestion_thread.signal_data[signal_idx]
                    # Calculate congestion level using the same method as the thread
                    congestion_level, congestion_score, color = self.congestion_thread.calculate_congestion_level(
                        data['vehicle_count'], 
                        data['traffic_weight'], 
                        data['area_size']
                    )
                    congestion_data[signal_idx] = {
                        'level': congestion_level,
                        'score': congestion_score,
                        'color': color,
                        'vehicle_count': data['vehicle_count'],
                        'traffic_weight': data['traffic_weight']
                    }
                else:
                    # Default values if no data available
                    congestion_data[signal_idx] = {
                        'level': 'LOW',
                        'score': 0.0,
                        'color': 'green',
                        'vehicle_count': 0,
                        'traffic_weight': 0.0
                    }
        return congestion_data

class AnalyticsThread(QThread):
    analytics_data_ready = pyqtSignal(dict)

    def __init__(self, analytics_data, signals, get_congestion_data_func):
        super().__init__()
        self.running = True
        self.analytics_data = analytics_data
        self.signals = signals
        self.get_congestion_data_func = get_congestion_data_func

    def prepare_analytics_data(self):
        """Prepares all data required for the analytics plots."""
        try:
            # Vehicle counts and green times
            vehicle_counts_data = [list(self.analytics_data['vehicle_counts'][i]) for i in range(4)]
            green_times_data = [list(self.analytics_data['green_times'][i]) for i in range(4)]

            # Traffic distribution
            vehicle_distribution = [signal.vehicle_count for signal in self.signals]

            # Average confidence scores
            avg_confidences = [signal.avg_confidence for signal in self.signals]

            # Congestion data
            congestion_data = self.get_congestion_data_func()

            return {
                'timestamps': list(self.analytics_data['timestamps']),
                'vehicle_counts': vehicle_counts_data,
                'green_times': green_times_data,
                'vehicle_distribution': vehicle_distribution,
                'avg_confidences': avg_confidences,
                'congestion_data': congestion_data,
            }
        except Exception as e:
            print(f"Error preparing analytics data: {e}")
            return None

    def run(self):
        """Main analytics thread loop"""
        while self.running:
            try:
                data = self.prepare_analytics_data()
                if data:
                    self.analytics_data_ready.emit(data)
                self.msleep(2000)  # Update every 2 seconds
            except Exception as e:
                print(f"Error in analytics thread: {e}")
                self.msleep(5000)  # Wait longer on error

    def trigger_update(self):
        """Trigger a manual update of the analytics"""
        try:
            data = self.prepare_analytics_data()
            if data:
                self.analytics_data_ready.emit(data)
        except Exception as e:
            print(f"Error in manual analytics update: {e}")

    def stop(self):
        """Stop the analytics thread"""
        self.running = False
        self.wait()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = EnhancedTrafficManagementSystem()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()