import click
import json
import cv2
import numpy as np
from src.agent import DetectionAgent
from src.config import Config

def print_progress(progress):
    """Print progress bar."""
    click.echo(f"\rProgress: {progress:.1f}%", nl=False)

def draw_results(frame, results, config):
    """Draw detection results on frame."""
    frame_copy = frame.copy()
    
    for result in results:
        # Get coordinates
        box = result['box']
        h, w = frame.shape[:2]
        box_pixels = [(int(x * w), int(y * h)) for x, y in box]
        x_coords = [x for x, y in box_pixels]
        y_coords = [y for x, y in box_pixels]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        # Choose color based on object type
        color = {
            'Person': (0, 255, 0),  # Green
            'Car': (255, 0, 0),     # Blue
            'Truck': (255, 0, 0),
            'Bus': (255, 0, 0),
            'Motorcycle': (255, 0, 0),
            'Bicycle': (255, 165, 0) # Orange
        }.get(result['name'], (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        texts = [result['name']]
        if config.SHOW_CONFIDENCE:
            texts.append(f"{result['score']:.2f}")
        
        if result['name'] == 'Person':
            if result.get('age'):
                texts.append(f"Age: {result['age']:.1f}")
            if result.get('gender'):
                gender_text = f"{result['gender'].title()}"
                if config.SHOW_CONFIDENCE:
                    gender_text += f" ({result['gender_confidence']:.2f})"
                texts.append(gender_text)
        elif result['name'] in {'Car', 'Truck', 'Bus', 'Motorcycle'}:
            if result.get('make') and result.get('model'):
                vehicle_text = f"{result['make']} {result['model']}"
                if result.get('category'):
                    vehicle_text += f" ({result['category']})"
                if config.SHOW_CONFIDENCE:
                    vehicle_text += f" ({result['confidence']:.2f})"
                texts.append(vehicle_text)
        
        # Draw text
        for i, text in enumerate(texts):
            y = y1 + 20 + (i * 20)
            cv2.putText(frame_copy, text, (x1, y),
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                       color, 2)
    
    return frame_copy

@click.command()
@click.option('--video', help='Path to video file')
@click.option('--images', help='Path to folder containing images')
@click.option('--webcam', is_flag=True, help='Use webcam')
@click.option('--camera-id', default=0, help='Camera device ID')
@click.option('--output', default='results.json', help='Output JSON file path')
@click.option('--display/--no-display', default=True, help='Show detection results')
def main(video, images, webcam, camera_id, output, display):
    """Run object detection and classification on video, images, or webcam."""
    if sum([bool(video), bool(images), webcam]) != 1:
        click.echo("Error: Must specify exactly one of: --video, --images, or --webcam")
        return
        
    try:
        # Initialize agent
        config = Config(
            video_path=video,
            image_folder=images,
            use_webcam=webcam,
            camera_id=camera_id
        )
        agent = DetectionAgent(config)
        
        click.echo("Starting detection...")
        
        # Create display window if needed
        if display:
            cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
        
        # Prepare results storage
        all_results = []
        
        def handle_display(frame, results):
            if display:
                # Draw results on frame
                annotated_frame = draw_results(frame, results, config)
                
                # Resize if needed
                if config.DISPLAY_SCALE != 1.0:
                    h, w = annotated_frame.shape[:2]
                    new_h = int(h * config.DISPLAY_SCALE)
                    new_w = int(w * config.DISPLAY_SCALE)
                    annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))
                
                # Show frame
                cv2.imshow('Detection Results', annotated_frame)
                
                # Check for quit command
                key = cv2.waitKey(1)
                if key == ord('q'):
                    return False
            return True
        
        # Run detection
        results = agent.run(
            progress_callback=None if webcam else print_progress,
            display_callback=handle_display
        )
        
        # Save results
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
            
        if not webcam:
            click.echo(f"\nDone! Results saved to {output}")
        
    except Exception as e:
        click.echo(f"\nError: {str(e)}")
        raise
    finally:
        if display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 