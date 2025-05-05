import cv2
import numpy as np
from typing import List, Tuple, Optional
from .model import CharacterRecognitionModel
from PIL import Image, ImageDraw, ImageFont
import os


class OCRProcessor:
    """Optical Character Recognition Processor for Lao Handwriting."""

    def __init__(
        self,
        model_path: str = "model/hand_drawn_character_model.keras",
        label_map_path: str = "model/label_map.json",
    ) -> None:
        """
        Initialize the OCR processor with a trained model and font settings.

        Args:
            model_path (str): Path to the trained character recognition model.
            label_map_path (str): Path to the label map (character mappings).

        Raises:
            RuntimeError: If the model fails to load.
        """
        self.recognizer = CharacterRecognitionModel()
        if not self.recognizer.load_model(model_path, label_map_path):
            raise RuntimeError("Failed to load character recognition model")

        # Parameters optimized for Lao script
        self.min_contour_area: int = 25
        self.min_aspect_ratio: float = 0.1
        self.max_aspect_ratio: float = 4.0
        self.cluster_distance_threshold: int = 10

        # Load font
        possible_fonts = [
            "src\\assets\\fonts\\Phetsarath_OT.ttf",
            "C:\\Windows\\Fonts\\Phetsarath_OT.ttf",
            "\\usr\\share\\fonts\\truetype\\lao\\Phetsarath_OT.ttf",
            os.path.join(os.path.dirname(__file__), "fonts", "Phetsarath_OT.ttf"),
        ]

        self.font_path: Optional[str] = None
        for font in possible_fonts:
            if os.path.exists(font):
                self.font_path = font
                break

        if self.font_path is None:
            print("Warning: No suitable font found. Text display may be limited.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better character segmentation.

        Args:
            image (np.ndarray): Input image (color or grayscale).

        Returns:
            np.ndarray: Preprocessed binary image.

        Raises:
            ValueError: If the input image is invalid.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Check if image is already grayscale (1 channel)
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # For binary images, we might not need thresholding
        if np.max(gray) == 255 and np.min(gray) in [0, 255]:
            binary = gray
        else:
            # Apply adaptive thresholding for other cases
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

        return binary

    def prepare_char_image(self, char_region: np.ndarray) -> np.ndarray:
        """
        Prepare a character region image for model input.

        Args:
            char_region (np.ndarray): Cropped character region.

        Returns:
            np.ndarray: Resized (128x128) input image.

        Raises:
            ValueError: If the character region is invalid.
        """
        if char_region is None or char_region.size == 0:
            raise ValueError("Invalid character region")

        h, w = char_region.shape
        max_dim = max(w, h)

        # Create a square black background
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

        # Calculate padding to center the character
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2

        # Place the character region in the center
        square_img[pad_y : pad_y + h, pad_x : pad_x + w] = char_region

        # Add padding around the square (20% on each side)
        padding = int(max_dim * 0.2)
        padded_img = np.zeros(
            (max_dim + 2 * padding, max_dim + 2 * padding), dtype=np.uint8
        )
        padded_img[padding : padding + max_dim, padding : padding + max_dim] = (
            square_img
        )

        # Resize to model input size (128x128)
        resized = cv2.resize(padded_img, (128, 128), interpolation=cv2.INTER_AREA)
        return resized

    def find_connected_components(
        self, binary_image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Find connected components in a binary image.

        Args:
            binary_image (np.ndarray): Binary preprocessed image.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
                List of components with their bounding boxes.
        """
        contours, _ = cv2.findContours(
            binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        components = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            if (
                area >= self.min_contour_area
                and self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            ):
                # Extract the component with boundary checking
                if (
                    y >= 0
                    and x >= 0
                    and (y + h) <= binary_image.shape[0]
                    and (x + w) <= binary_image.shape[1]
                ):
                    comp = binary_image[y : y + h, x : x + w]
                    if comp.size > 0:
                        components.append((comp, (x, y, w, h)))

        return components

    def group_components(
        self, components: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> List[List[int]]:
        """
        Group components that likely belong to the same character.

        Args:
            components (List[Tuple[np.ndarray, Tuple[int, int, int, int]]]): List of components.

        Returns:
            List[List[int]]: Groups of component indices.
        """
        if not components:
            return []

        groups = []
        used = set()

        # Sort components by x-coordinate to process left-to-right
        sorted_components = sorted(enumerate(components), key=lambda x: x[1][1][0])

        for i, (_, box1) in sorted_components:
            if i in used:
                continue

            current_group = [i]
            x1, y1, w1, h1 = box1

            # Calculate center point x only
            center1_x = x1 + w1 / 2

            for j, (_, box2) in sorted_components:
                if j in used or j == i:
                    continue

                x2, y2, w2, h2 = box2
                center2_x = x2 + w2 / 2

                # Check if component 2 is significantly smaller (likely a vowel/tone mark)
                is_diacritic = (w2 * h2) < (w1 * h1 * 0.5)

                # Different grouping rules for diacritics vs regular characters
                if is_diacritic:
                    # For diacritics, check if it's above/below and within horizontal bounds
                    x_overlap = x1 <= center2_x <= x1 + w1
                    y_distance = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))

                    if x_overlap and y_distance < self.cluster_distance_threshold:
                        continue  # Don't group diacritics with base character
                else:
                    # For regular characters, use stricter horizontal distance
                    x_distance = center2_x - center1_x
                    if 0 < x_distance < self.cluster_distance_threshold / 2:
                        current_group.append(j)
                        used.add(j)

            used.add(i)
            groups.append(current_group)

        return groups

    def recognize_text(
        self, image: np.ndarray, return_bbox: bool = False, min_confidence: float = 0.1
    ) -> Tuple[str, List[Tuple[int, int, int, int]], List[float], bool]:
        """
        Recognize text from an image.

        Args:
            image (np.ndarray): Input image.
            return_bbox (bool, optional): Whether to return bounding boxes. Defaults to False.
            min_confidence (float, optional): Minimum confidence to include results. Defaults to 0.1.

        Returns:
            Union[str, Tuple[str, List[Tuple[int, int, int, int]], List[float], bool]]:
                - Recognized text
                - Bounding boxes (if return_bbox=True)
                - Confidence scores for each character
                - Flag indicating if any content was detected
        """
        if image is None or image.size == 0:
            return ("", [], [], False) if return_bbox else ("", [], False)

        try:
            binary = self.preprocess_image(image)
            components = self.find_connected_components(binary)
            if not components:
                # No components detected at all
                return ("", [], [], False) if return_bbox else ("", [], False)

            groups = self.group_components(components)

            recognized_chars = []
            bounding_boxes = []
            confidence_scores = []
            has_content = False  # Flag to indicate if any content was detected

            for group in groups:
                # Get the bounding box that contains all components in the group
                boxes = [components[i][1] for i in group]
                x = min(box[0] for box in boxes)
                y = min(box[1] for box in boxes)
                max_x = max(box[0] + box[2] for box in boxes)
                max_y = max(box[1] + box[3] for box in boxes)
                w = max_x - x
                h = max_y - y

                # Extract and validate the region
                if (
                    x >= 0
                    and y >= 0
                    and w > 0
                    and h > 0
                    and x + w <= binary.shape[1]
                    and y + h <= binary.shape[0]
                ):
                    char_region = binary[y : y + h, x : x + w]
                    has_content = True  # We have found some visual content

                    try:
                        char_input = self.prepare_char_image(char_region)
                        predicted_char, confidence = self.recognizer.predict(char_input)

                        # Include all results above min_confidence
                        if confidence > min_confidence:
                            recognized_chars.append(predicted_char)
                            bounding_boxes.append((x, y, w, h))
                            confidence_scores.append(confidence)
                    except Exception as e:
                        print(f"Error processing character region: {e}")
                        continue

            text = "".join(recognized_chars)
            if return_bbox:
                return text, bounding_boxes, confidence_scores, has_content
            else:
                return text, confidence_scores, has_content

        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ("", [], [], False) if return_bbox else ("", [], False)

    def visualize_results(
        self,
        image: np.ndarray,
        text: str,
        bounding_boxes: List[Tuple[int, int, int, int]],
        confidence_scores: List[float] = None,
    ) -> np.ndarray:
        """
        Visualize OCR results by drawing bounding boxes and recognized text.

        Args:
            image (np.ndarray): Original image.
            text (str): Recognized text.
            bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes.
            confidence_scores (List[float], optional): Confidence scores for each character.

        Returns:
            np.ndarray: Image with drawn results.
        """
        if len(image.shape) == 2:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_image)

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, 22)
                small_font = ImageFont.truetype(self.font_path, 16)
            else:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except Exception as e:
            print(f"Font error: {e}")
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Determine if we have confidence scores to display
        has_confidence = confidence_scores and len(confidence_scores) == len(text)

        # Draw bounding boxes and labels
        for i, ((x, y, w, h), char) in enumerate(zip(bounding_boxes, text)):
            # Determine color based on confidence (if available)
            if has_confidence:
                conf = confidence_scores[i]
                # Color gradient from red (low) to yellow to green (high)
                if conf < 0.3:  # Low confidence
                    box_color = (255, 0, 0)  # Red
                elif conf < 0.7:  # Medium confidence
                    box_color = (255, 255, 0)  # Yellow
                else:  # High confidence
                    box_color = (0, 255, 0)  # Green
            else:
                box_color = (0, 255, 0)  # Default green if no confidence scores

            # Draw rectangle with confidence-based color
            draw.rectangle([(x, y), (x + w, y + h)], outline=box_color, width=2)

            # Calculate text position (above the box)
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Center text above box
            text_x = x + (w - text_width) // 2
            text_y = max(0, y - text_height - 5)

            # Draw text with border for better visibility
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((text_x + dx, text_y + dy), char, font=font, fill=(0, 0, 0))

            # Draw main text
            draw.text((text_x, text_y), char, font=font, fill=(255, 0, 0))

            # Draw confidence value if available
            if has_confidence:
                conf_text = f"{confidence_scores[i]:.0%}"
                conf_bbox = draw.textbbox((0, 0), conf_text, font=small_font)
                conf_width = conf_bbox[2] - conf_bbox[0]

                # Position confidence below the box
                conf_x = x + (w - conf_width) // 2
                conf_y = y + h - 15

                # Draw confidence text with background for readability
                text_bg_color = (50, 50, 50, 180)  # Semi-transparent dark background
                conf_bg_padding = 2
                draw.rectangle(
                    [
                        (conf_x - conf_bg_padding, conf_y - conf_bg_padding),
                        (
                            conf_x + conf_width + conf_bg_padding,
                            conf_y + (conf_bbox[3] - conf_bbox[1]) + conf_bg_padding,
                        ),
                    ],
                    fill=text_bg_color,
                )

                # Draw confidence value in white
                draw.text(
                    (conf_x, conf_y), conf_text, font=small_font, fill=(255, 255, 255)
                )

        # Convert back to OpenCV format (RGB to BGR)
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result
