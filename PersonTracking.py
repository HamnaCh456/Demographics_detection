"""
Enhanced People Counter with Person Tracking
Counts UNIQUE people throughout the video, not just per-frame counts
Uses DeepSORT or simple centroid tracking for person identification
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import math

try:
    from ultralytics import YOLO
except ImportError:
    print("Required packages not installed. Please run: pip install ultralytics opencv-python")
    exit(1)


class CentroidTracker:
    """
    Simple centroid-based tracker for unique person identification
    Tracks objects by matching centroids across frames
    """
    
    def __init__(self, max_disappeared=50, max_distance=100):
        """
        Args:
            max_disappeared: Maximum frames an object can disappear before removal
            max_distance: Maximum pixel distance to consider same object
        """
        self.next_object_id = 0
        self.objects = {}  # {object_id: centroid}
        self.disappeared = {}  # {object_id: disappeared_count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track appearance info for each person
        self.first_seen = {}  # {object_id: frame_number}
        self.last_seen = {}   # {object_id: frame_number}
        self.total_frames_visible = {}  # {object_id: count}
    
    def register(self, centroid, frame_number):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = frame_number
        self.last_seen[self.next_object_id] = frame_number
        self.total_frames_visible[self.next_object_id] = 1
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object that has disappeared for too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections, frame_number):
        """
        Update tracked objects with new detections
        
        Args:
            detections: List of bounding boxes [[x1,y1,x2,y2], ...]
            frame_number: Current frame number
            
        Returns:
            Dictionary of {object_id: centroid}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Calculate centroids of current detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], frame_number)
        
        # Otherwise, match existing objects with new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between each pair
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i, j] = np.linalg.norm(obj_centroid - input_centroid)
            
            # Match objects to detections
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match existing objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.last_seen[object_id] = frame_number
                self.total_frames_visible[object_id] += 1
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched objects (disappeared)
            unused_rows = set(range(distances.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(distances.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], frame_number)
        
        return self.objects
    
    def get_unique_count(self):
        """Get total number of unique people seen"""
        return max(self.next_object_id, len(self.first_seen))
    
    def get_tracking_info(self):
        """Get detailed tracking information"""
        return {
            'total_unique_people': self.get_unique_count(),
            'currently_tracked': len(self.objects),
            'tracking_details': {
                str(obj_id): {
                    'first_seen_frame': self.first_seen.get(obj_id, -1),
                    'last_seen_frame': self.last_seen.get(obj_id, -1),
                    'total_frames_visible': self.total_frames_visible.get(obj_id, 0)
                }
                for obj_id in self.first_seen.keys()
            }
        }


class EnhancedPeopleCounter:
    """
    Enhanced people counter with tracking to identify unique individuals
    """
    
    def __init__(self, yolo_model='yolov8n.pt', confidence_threshold=0.5, 
                 tracking_distance=100, max_disappeared=30):
        """
        Initialize the enhanced counter with tracking
        
        Args:
            yolo_model: YOLOv8 model variant
            confidence_threshold: Minimum confidence for detections
            tracking_distance: Maximum distance to consider same person (pixels)
            max_disappeared: Frames before considering person left
        """
        print("Initializing Enhanced People Counter with Tracking...")
        print(f"Loading YOLOv8 model: {yolo_model}")
        
        self.yolo = YOLO(yolo_model)
        self.confidence_threshold = confidence_threshold
        
        # Initialize tracker
        self.tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=tracking_distance
        )
        
        print(f"✅ System ready! Tracking enabled (max distance: {tracking_distance}px)")
    
    def detect_people(self, frame):
        """
        Detect people in a frame
        
        Returns:
            List of bounding boxes [[x1,y1,x2,y2], ...]
        """
        results = self.yolo(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 0 is 'person' in COCO dataset
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
        
        return detections
    
    def draw_tracking(self, frame, detections, tracked_objects, frame_number):
        """
        Draw bounding boxes with tracking IDs
        
        Args:
            frame: Image frame
            detections: List of detection bounding boxes
            tracked_objects: Dictionary of {object_id: centroid}
            frame_number: Current frame number
        """
        # Create a mapping of centroids to object IDs
        centroid_to_id = {}
        for obj_id, centroid in tracked_objects.items():
            centroid_to_id[tuple(centroid)] = obj_id
        
        # Draw each detection
        for (x1, y1, x2, y2) in detections:
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Find closest tracked object
            min_dist = float('inf')
            matched_id = None
            for obj_id, obj_centroid in tracked_objects.items():
                dist = np.linalg.norm(np.array([cx, cy]) - obj_centroid)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            
            # Draw bounding box
            color = self._get_color_for_id(matched_id if matched_id else 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and info
            if matched_id is not None:
                label = f"Person #{matched_id}"
                first_seen = self.tracker.first_seen.get(matched_id, frame_number)
                duration = frame_number - first_seen
                
                # Label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw centroid
                cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    def _get_color_for_id(self, object_id):
        """Generate consistent color for each tracked ID"""
        np.random.seed(object_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color
    
    def process_video(self, video_path, output_path=None, show_preview=False, 
                     save_stats=True, progress_interval=30):
        """
        Process video with person tracking
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (None to skip)
            show_preview: Show live preview window
            save_stats: Save statistics to JSON
            progress_interval: Print progress every N frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print("\n" + "="*70)
        print("PROCESSING VIDEO WITH PERSON TRACKING")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print("="*70)
        
        # Initialize video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"📹 Output video will be saved to: {output_path}")
        
        # Reset tracker
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=100)
        
        # Statistics
        stats = {
            'frame_data': [],
            'people_per_frame': [],
            'max_people_in_frame': 0,
            'max_people_frame_number': 0
        }
        
        frame_count = 0
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections, frame_count)
            
            # Track statistics
            current_people = len(detections)
            stats['people_per_frame'].append(current_people)
            
            if current_people > stats['max_people_in_frame']:
                stats['max_people_in_frame'] = current_people
                stats['max_people_frame_number'] = frame_count
            
            stats['frame_data'].append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'people_count': current_people,
                'unique_people_so_far': self.tracker.get_unique_count(),
                'currently_tracked': len(tracked_objects)
            })
            
            # Draw tracking
            if output_path or show_preview:
                annotated_frame = self.draw_tracking(
                    frame.copy(), detections, tracked_objects, frame_count
                )
                
                # Add info overlay
                unique_count = self.tracker.get_unique_count()
                info_text = f"Frame: {frame_count}/{total_frames} | Current: {current_people} | Total Unique: {unique_count}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if output_path:
                    out.write(annotated_frame)
                
                if show_preview:
                    cv2.imshow('People Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️  Processing stopped by user")
                        break
            
            # Progress
            if frame_count % progress_interval == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                unique = self.tracker.get_unique_count()
                print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                      f"Current: {current_people} | Unique Total: {unique} | "
                      f"Processing: {fps_processing:.1f} fps")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get tracking info
        tracking_info = self.tracker.get_tracking_info()
        
        # Compile final statistics
        # Convert numpy types to native Python types for JSON serialization
        unique_counts, frequencies = np.unique(stats['people_per_frame'], return_counts=True)
        people_count_freq = {int(k): int(v) for k, v in zip(unique_counts, frequencies)}
        
        results = {
            'video_info': {
                'path': str(video_path),
                'resolution': [width, height],
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': float(duration)
            },
            'summary': {
                'total_unique_people': int(tracking_info['total_unique_people']),
                'max_people_in_single_frame': int(stats['max_people_in_frame']),
                'max_people_frame_number': int(stats['max_people_frame_number']),
                'max_people_timestamp': float(stats['max_people_frame_number'] / fps),
                'average_people_per_frame': float(np.mean(stats['people_per_frame'])),
                'total_frames_processed': int(frame_count),
                'processing_time_seconds': float(processing_time),
                'processing_fps': float(frame_count / processing_time)
            },
            'tracking': tracking_info,
            'distribution': {
                'people_count_frequency': people_count_freq
            },
            'frame_data': stats['frame_data']
        }
        
        # Print summary
        self._print_summary(results)
        
        # Save statistics
        if save_stats:
            stats_path = Path(video_path).stem + '_tracking_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Statistics saved to: {stats_path}")
        
        if output_path:
            print(f"📹 Annotated video saved to: {output_path}")
        
        return results
    
    def _print_summary(self, results):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        print("\n📊 SUMMARY:")
        print(f"  • Total frames processed: {results['summary']['total_frames_processed']}")
        print(f"  • Processing time: {results['summary']['processing_time_seconds']:.2f} seconds")
        print(f"  • Processing speed: {results['summary']['processing_fps']:.1f} fps")
        
        print("\n👥 UNIQUE PEOPLE TRACKING:")
        print(f"  • TOTAL UNIQUE PEOPLE DETECTED: {results['summary']['total_unique_people']}")
        print(f"  • Maximum people in single frame: {results['summary']['max_people_in_single_frame']}")
        print(f"  • Occurred at frame {results['summary']['max_people_frame_number']} "
              f"(timestamp: {results['summary']['max_people_timestamp']:.2f}s)")
        print(f"  • Average people per frame: {results['summary']['average_people_per_frame']:.2f}")
        
        print("\n📈 INDIVIDUAL TRACKING:")
        for person_id, info in results['tracking']['tracking_details'].items():
            print(f"  Person #{person_id}:")
            print(f"    - First seen: frame {info['first_seen_frame']} "
                  f"({info['first_seen_frame']/results['video_info']['fps']:.2f}s)")
            print(f"    - Last seen: frame {info['last_seen_frame']} "
                  f"({info['last_seen_frame']/results['video_info']['fps']:.2f}s)")
            print(f"    - Visible in {info['total_frames_visible']} frames")
        
        print("\n" + "="*70)
        print("✅ Processing complete!")
        print("="*70)


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced People Counter with Tracking')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO model (yolov8n/s/m/l/x)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--tracking-distance', type=int, default=100,
                       help='Max pixel distance for tracking same person')
    parser.add_argument('--max-disappeared', type=int, default=30,
                       help='Max frames before person considered left')
    parser.add_argument('--preview', action='store_true', help='Show preview')
    parser.add_argument('--no-stats', action='store_true', help='Do not save stats')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = EnhancedPeopleCounter(
        yolo_model=args.model,
        confidence_threshold=args.confidence,
        tracking_distance=args.tracking_distance,
        max_disappeared=args.max_disappeared
    )
    
    # Process video
    counter.process_video(
        video_path=args.input,
        output_path=args.output,
        show_preview=args.preview,
        save_stats=not args.no_stats
    )


if __name__ == "__main__":
    main()
