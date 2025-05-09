import cv2
import numpy as np
import torch
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class ECGDigitizer:
    def __init__(self):
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    def preprocess_image(self, image):
        """Preprocess the ECG image for better digitization"""
        print("\n=== Image Preprocessing ===")
        print(f"Input image shape: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Converted to grayscale")
        else:
            gray = image
            print("Already grayscale")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("Applied Gaussian blur")
        
        # Apply adaptive thresholding with larger block size
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 2
        )
        print("Applied adaptive thresholding")
        
        # Remove grid lines with larger kernel
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        print("Removed grid lines")
        
        return binary
    
    def detect_leads(self, binary_image):
        """Detect and separate individual leads from the ECG image"""
        print("\n=== Lead Detection Debug ===")
        print(f"Binary image shape: {binary_image.shape}")
        
        # Find contours with simplified approximation
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"Found {len(contours)} total contours")
        
        # Filter contours by area, aspect ratio, width, and height
        lead_contours = []
        min_width = 100  # Minimum width for a lead contour
        min_height = 30  # Minimum height for a lead contour
        max_height = binary_image.shape[0] // 6  # No lead should be taller than 1/6th of the image
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            print(f"\nContour {i}:")
            print(f"  Position: x={x}, y={y}, w={w}, h={h}")
            print(f"  Aspect ratio: {aspect_ratio:.2f}")
            print(f"  Area: {area:.2f}")
            
            # Filtering
            if (
                0.05 < aspect_ratio < 20 and
                area > 500 and
                w > min_width and
                min_height < h < max_height
            ):
                lead_contours.append((x, y, w, h))
                print("  ✓ Accepted as lead")
            else:
                print("  ✗ Rejected as lead")
        
        print(f"\nAfter filtering: {len(lead_contours)} valid leads")
        
        # If more than 12, select the 12 largest by width
        if len(lead_contours) > 12:
            print("More than 12 leads detected, selecting 12 largest by width...")
            lead_contours = sorted(lead_contours, key=lambda c: c[2], reverse=True)[:12]
        
        # Sort leads by vertical position
        lead_contours.sort(key=lambda x: x[1])
        print(f"\nSelected {len(lead_contours)} leads (sorted by vertical position):")
        for idx, (x, y, w, h) in enumerate(lead_contours):
            print(f"  Lead {idx+1}: x={x}, y={y}, w={w}, h={h}")
        
        # If we still don't have 12 leads, try to split large ones (as before)
        if len(lead_contours) < 12:
            print("Too few leads detected, attempting to split large ones...")
            split_contours = []
            for x, y, w, h in lead_contours:
                if h > 100:  # If lead is too tall, split it
                    num_splits = min(2, 12 - len(lead_contours) + 1)
                    split_height = h // num_splits
                    for i in range(num_splits):
                        split_contours.append((x, y + i * split_height, w, split_height))
                else:
                    split_contours.append((x, y, w, h))
            lead_contours = split_contours
            print(f"After splitting: {len(lead_contours)} leads")
        
        return lead_contours
    
    def extract_lead_signal(self, binary_image, x, y, w, h):
        """Extract the ECG signal from a single lead"""
        # Extract the lead region
        lead_region = binary_image[y:y+h, x:x+w]
        
        # Find the center line of the signal
        center_line = []
        for col in range(lead_region.shape[1]):
            # Find white pixels in this column
            white_pixels = np.where(lead_region[:, col] == 255)[0]
            if len(white_pixels) > 0:
                # Use weighted average of white pixels as the center
                weights = np.exp(-0.1 * np.abs(white_pixels - h/2))  # Gaussian weighting
                center = np.average(white_pixels, weights=weights)
                center_line.append(center)
            else:
                # If no white pixels, use the previous center
                if center_line:
                    center_line.append(center_line[-1])
                else:
                    center_line.append(h/2)
        
        # Convert to signal
        signal = np.array(center_line)
        signal = signal - np.mean(signal)  # Center the signal
        
        # Apply smoothing
        signal = np.convolve(signal, np.ones(3)/3, mode='same')
        
        return signal
    
    def digitize_ecg(self, image_path):
        """Convert ECG image to digital signal data in format expected by ML model"""
        try:
            print("\n=== ECG Digitization ===")
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            print(f"Successfully read image from {image_path}")
            
            # Preprocess image
            binary_image = self.preprocess_image(image)
            
            # Detect leads
            lead_contours = self.detect_leads(binary_image)
            
            if len(lead_contours) != 12:
                print(f"\nWARNING: Expected 12 leads, found {len(lead_contours)}")
                # Try adjusting preprocessing parameters
                print("Attempting with adjusted preprocessing...")
                # Increase kernel size for better grid removal
                kernel = np.ones((3,3), np.uint8)
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
                
                # Try detecting leads again
                lead_contours = self.detect_leads(binary_image)
                if len(lead_contours) != 12:
                    raise ValueError(f"Expected 12 leads, found {len(lead_contours)} after adjustment")
            
            # Extract signals for each lead
            ecg_signals = {}
            for i, (x, y, w, h) in enumerate(lead_contours):
                if i < len(self.lead_names):
                    lead_name = self.lead_names[i]
                    signal = self.extract_lead_signal(binary_image, x, y, w, h)
                    ecg_signals[lead_name] = signal
                    print(f"Extracted signal for lead {lead_name}, length: {len(signal)}")
            
            # Convert to numpy array format
            max_length = max(len(signal) for signal in ecg_signals.values())
            target_length = 512  # Fixed length for model compatibility
            ecg_array = np.zeros((12, target_length))
            
            for i, lead_name in enumerate(self.lead_names):
                signal = ecg_signals[lead_name]
                # Resample to target_length
                signal = np.interp(
                    np.linspace(0, len(signal) - 1, target_length),
                    np.arange(len(signal)),
                    signal
                )
                ecg_array[i] = signal
            
            # Transform to format expected by ML model
            # Convert to torch tensor and add batch dimension
            ecg_tensor = torch.FloatTensor(ecg_array).unsqueeze(0)  # Shape: (1, 12, 512)
            print(f"Final tensor shape: {ecg_tensor.shape}")
            
            # Generate text description
            text_description = f"ECG recording with 12 leads. "
            text_description += f"Sampling rate: 500 Hz. "
            text_description += f"Leads: {', '.join(self.lead_names)}."
            
            return {
                'signal': ecg_tensor,  # Shape: (1, 12, 512)
                'text_description': text_description,
                'metadata': {
                    'num_leads': 12,
                    'sampling_rate': 500,
                    'lead_names': self.lead_names,
                    'signal_shape': ecg_tensor.shape
                }
            }
            
        except Exception as e:
            print(f"Error digitizing ECG image: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def digitize_ecg_image(image_path):
    """Helper function to digitize an ECG image"""
    digitizer = ECGDigitizer()
    return digitizer.digitize_ecg(image_path) 