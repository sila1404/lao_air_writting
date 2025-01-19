import cv2
from utils import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor()

# Read image with Lao text
image = cv2.imread("datasets/lao_text.jpg")

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    try:
        # Recognize text with bounding boxes
        text, boxes = ocr.recognize_text(image, return_bbox=True)

        # Check if any text was recognized
        if text:
            print(f"Recognized Text: {text}")

            # Visualize results
            result = ocr.visualize_results(image, text, boxes)
            cv2.imshow("Lao OCR Results", result)
            cv2.waitKey(0)
        else:
            print("No text recognized in the image.")
    except Exception as e:
        print(f"Error during OCR processing: {e}")
