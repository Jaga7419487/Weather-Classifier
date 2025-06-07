import os
import cv2
import torch
import torchvision.transforms as T

from PIL import Image

# Class groupings
grouped_dict = {
    'foggy': ['fogsmog', 'sandstorm'],
    'cold': ['snow', 'rime', 'frost', 'glaze'],
    'cloudy': ['cloudy'],
    'dew': ['dew'],
    'hail': ['hail'],
    'lightning': ['lightning'],
    'rain': ['rain'],
    'shine': ['shine'],
    'sunrise': ['sunrise']
}

all_classes = ['cloudy', 'dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rime', 'sandstorm', 'shine', 'snow', 'sunrise']
grouped_classes = list(grouped_dict.keys())
cold_classes = ['frost', 'glaze', 'rime', 'snow']
foggy_classes = ['fogsmog', 'sandstorm']

videos_dir = "test videos"

def load_grouped_models(model_dir='models'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    grouped_model_path = os.path.join(model_dir, 'hierarchical_model.pth')
    foggy_model_path = os.path.join(model_dir, 'foggy_model_9419.pth')
    cold_model_path = os.path.join(model_dir, 'cold_model.pth')
    
    # Add safe globals for torchvision models
    torch.serialization.add_safe_globals(['torchvision.models.resnet.ResNet'])
    
    grouped_model = torch.load(grouped_model_path, map_location=device, weights_only=False)
    foggy_model = torch.load(foggy_model_path, map_location=device, weights_only=False)
    cold_model = torch.load(cold_model_path, map_location=device, weights_only=False)
    
    grouped_model.eval()
    foggy_model.eval()
    cold_model.eval()
    
    print("Hierarchical models loaded successfully!")
    return grouped_model, foggy_model, cold_model, device

def load_single_model(model_dir='models', model_filename='direct_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(model_dir, model_filename)
    
    # Add safe globals for torchvision models
    torch.serialization.add_safe_globals(['torchvision.models.resnet.ResNet'])
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print("Single CNN model loaded successfully!")
    return model, device

def predict_frame_hierarchical(frame, grouped_model, foggy_model, cold_model, transform, device):
    # Convert BGR to RGB (OpenCV uses BGR, the model expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(rgb_frame)
    
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # First layer prediction
        first_output = grouped_model(tensor)
        first_probs = torch.softmax(first_output, dim=1)
        confidence1, predicted_idx = torch.max(first_probs, 1)
        first_prediction = grouped_classes[predicted_idx.item()]
        confidence_score1 = confidence1.item() * 100
        
        final_prediction = first_prediction
        confidence_score = confidence_score1
        
        # Second layer prediction if needed
        if predicted_idx.item() == 0:  # foggy
            second_output = foggy_model(tensor)
            prob = torch.sigmoid(second_output).item()
            
            if prob > 0.5:
                final_prediction = 'sandstorm'
                confidence_score = prob * 100
            else:
                final_prediction = 'fogsmog'
                confidence_score = (1.0 - prob) * 100
                
        elif predicted_idx.item() == 1:  # cold
            second_output = cold_model(tensor)
            second_probs = torch.softmax(second_output, dim=1)
            confidence2, predicted_idx2 = torch.max(second_probs, 1)
            final_prediction = cold_classes[predicted_idx2.item()]
            confidence_score = confidence2.item() * 100
            
    return final_prediction, confidence_score

def predict_frame_single(frame, model, transform, device):
    # Convert BGR to RGB (OpenCV uses BGR, the model expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(rgb_frame)
    
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        prediction = all_classes[predicted_idx.item()]
        confidence_score = confidence.item() * 100
            
    return prediction, confidence_score

def process_video(video_path, transform, device, use_hierarchical=True, **models):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup output file
    os.makedirs('output', exist_ok=True)
    base_name = os.path.basename(video_path)
    model_type = 'hierarchical' if use_hierarchical else 'single'
    output_path = os.path.join('output', f"{model_type}_{base_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height + 80))
    print(f"Saving output to {output_path}")

    # Limit prediction frequency to improve performance
    predict_every_n_frames = 15
    frame_counter = 0
    
    # Store last prediction to avoid flickering
    last_prediction = ""
    last_confidence = 0
    
    if use_hierarchical:
        grouped_model = models.get('grouped_model')
        foggy_model = models.get('foggy_model')
        cold_model = models.get('cold_model')
        predict_func = lambda frame: predict_frame_hierarchical(
            frame, grouped_model, foggy_model, cold_model, transform, device
        )
    else:
        single_model = models.get('single_model')
        predict_func = lambda frame: predict_frame_single(
            frame, single_model, transform, device
        )
    
    # Real-time video classification display
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video")
            break
        
        if frame_counter % predict_every_n_frames == 0:
            try:
                prediction, confidence = predict_func(frame)
                last_prediction = prediction
                last_confidence = confidence
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        frame_counter += 1
        
        # Create a black bar at the bottom for text
        result_frame = cv2.copyMakeBorder(
            frame, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        model_info = "Hierarchical" if use_hierarchical else "Single CNN"
        text = f"{model_info}: {last_prediction} ({last_confidence:.2f}%)"
        cv2.putText(
            result_frame, text, (10, frame_height + 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        out.write(result_frame)
        
        # Try to display the frame, but don't fail if GUI isn't available
        try:
            cv2.imshow("Weather Classification", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass  # Skip display if not available
    
    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    print(f"Video processed and saved to {output_path}")

def video_demo():
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video files:")
    for i, file in enumerate(video_files):
        print(f"{i+1}. {file}")
    
    print("\nSelect classification approach:")
    print("1. Hierarchical model (grouped + specialized models)")
    print("2. Single CNN model (direct classification)")
    
    try:
        approach = int(input("Enter choice (1 or 2): "))
        use_hierarchical = approach == 1
        
        if not use_hierarchical:
            single_model_filename = 'single_cnn_model.pth'
            print(f"Using single CNN model: {single_model_filename}")
        else:
            single_model_filename = None
    except ValueError:
        print("Invalid input, defaulting to hierarchical approach")
        use_hierarchical = True
        single_model_filename = None
    
    if use_hierarchical:
        grouped_model, foggy_model, cold_model, device = load_grouped_models()
        models_dict = {
            'grouped_model': grouped_model,
            'foggy_model': foggy_model,
            'cold_model': cold_model
        }
    else:
        single_model, device = load_single_model(model_filename=single_model_filename)
        models_dict = {'single_model': single_model}
    
    while True:
        try:
            selection = int(input("\nEnter the number of the video to process (0 to exit): "))
            if selection == 0:
                break
            
            if 1 <= selection <= len(video_files):
                video_path = os.path.join(videos_dir, video_files[selection-1])
                print(f"Processing {video_path}...")
                process_video(
                    video_path, 
                    transform, 
                    device, 
                    use_hierarchical=use_hierarchical, 
                    **models_dict
                )
            else:
                print("Invalid selection, please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_demo()
